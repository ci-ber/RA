from core.Trainer import Trainer
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import MultiStepLR

from time import time
import wandb
from dl_utils.config_utils import *
import logging
from model_zoo.ra import *
from optim.losses.image_losses import EmbeddingLoss
import matplotlib.pyplot as plt
import copy

class PTrainer(Trainer):
    """
    Code based on: https://github.com/taldatech/soft-intro-vae-pytorch/blob/main/soft_intro_vae/

    Added Reversed Embedding loss as in: Bercea, Cosmin I., et al. "Generalizing Unsupervised Anomaly Detection:
    Towards Unbiased Pathology Screening." Medical Imaging with Deep Learning. 2023.

    https://openreview.net/forum?id=8ojx-Ld3yjR
    """
    def __init__(self, training_params, model, data, device, log_wandb=True):
        self.optimizer_e = Adam(model.encoder.parameters(), lr=training_params['optimizer_params']['lr'])
        self.optimizer_d = Adam(model.decoder.parameters(), lr=training_params['optimizer_params']['lr'])

        self.e_scheduler = MultiStepLR(self.optimizer_e, milestones=(100,), gamma=0.1)
        self.d_scheduler = MultiStepLR(self.optimizer_d, milestones=(100,), gamma=0.1)

        self.scale = 1 / (training_params['input_size'][1] ** 2)  # normalize by images size (channels * height * width)
        self.gamma_r = 1e-8
        self.beta_kl = 1.0
        self.beta_rec = 0.5
        self.beta_neg = 128.0
        self.z_dim = 128

        self.embedding_loss = EmbeddingLoss()
        super(PTrainer, self).__init__(training_params, model, data, device, log_wandb)


    def train(self, model_state=None, opt_state=None, start_epoch=0):
        """
        Train local client
        :param model_state: weights
            weights of the global model
        :param opt_state: state
            state of the optimizer
        :param start_epoch: int
            start epoch
        :return:
            self.model.state_dict():
        """
        if model_state is not None:
            self.model.load_state_dict(model_state)  # load weights

        epoch_losses = []
        self.early_stop = False

        for epoch in range(self.training_params['nr_epochs']):
            if start_epoch > epoch:
                continue
            if self.early_stop is True:
                logging.info("[Trainer::test]: ################ Finished training (early stopping) ################")
                break
            start_time = time()

            diff_kls, batch_kls_real, batch_kls_fake, batch_kls_rec, batch_rec_errs, batch_exp_elbo_f,\
            batch_exp_elbo_r, batch_emb, count_images = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0

            for data in self.train_ds:
                # Input
                images = data[0].to(self.device)
                transformed_images = self.transform(images) if self.transform is not None else images
                b, c, w, h = images.shape

                count_images += b

                noise_batch = torch.randn(size=(b, self.z_dim)).to(self.device)
                real_batch = transformed_images.to(self.device)

                # =========== Update E ================
                for param in self.model.encoder.parameters():
                    param.requires_grad = True
                for param in self.model.decoder.parameters():
                    param.requires_grad = False

                fake = self.model.sample(noise_batch)
                # cloned_encoder = copy.deepcopy(self.model.encoder)
                real_mu, real_logvar, anomaly_embeddings = self.model.encode(real_batch)
                z = reparameterize(real_mu, real_logvar)
                rec = self.model.decoder(z)

                _, _, healthy_embeddings = self.model.encode(rec.detach())

                loss_emb = self.embedding_loss(anomaly_embeddings['embeddings'], healthy_embeddings['embeddings'])

                loss_rec = calc_reconstruction_loss(real_batch, rec, loss_type="mse", reduction="mean")
                lossE_real_kl = calc_kl(real_logvar, real_mu, reduce="mean")
                # loss_emb = loss_rec
                rec_rec, z_dict = self.model(rec.detach(), deterministic=False)
                rec_mu, rec_logvar, z_rec = z_dict['z_mu'], z_dict['z_logvar'], z_dict['z']
                rec_fake, z_dict_fake = self.model(fake.detach(), deterministic=False)
                fake_mu, fake_logvar, z_fake = z_dict_fake['z_mu'], z_dict_fake['z_logvar'], z_dict_fake['z']

                kl_rec = calc_kl(rec_logvar, rec_mu, reduce="none")
                kl_fake = calc_kl(fake_logvar, fake_mu, reduce="none")

                loss_rec_rec_e = calc_reconstruction_loss(rec, rec_rec, loss_type="mse", reduction='none')
                while len(loss_rec_rec_e.shape) > 1:
                    loss_rec_rec_e = loss_rec_rec_e.sum(-1)
                loss_rec_fake_e = calc_reconstruction_loss(fake, rec_fake, loss_type="mse", reduction='none')
                while len(loss_rec_fake_e.shape) > 1:
                    loss_rec_fake_e = loss_rec_fake_e.sum(-1)

                expelbo_rec = (-2 * self.scale * (self.beta_rec * loss_rec_rec_e + self.beta_neg * kl_rec)).exp().mean()
                expelbo_fake = (-2 * self.scale * (self.beta_rec * loss_rec_fake_e + self.beta_neg * kl_fake)).exp().mean()

                lossE_fake = 0.25 * (expelbo_rec + expelbo_fake)
                lossE_real = self.scale * (self.beta_rec * loss_rec + self.beta_kl * lossE_real_kl)

                lossE = lossE_real + lossE_fake + 0.005 * loss_emb
                self.optimizer_e.zero_grad()
                lossE.backward()
                self.optimizer_e.step()

                # ========= Update D ==================
                for param in self.model.encoder.parameters():
                    param.requires_grad = False
                for param in self.model.decoder.parameters():
                    param.requires_grad = True

                fake = self.model.sample(noise_batch)
                rec = self.model.decoder(z.detach())
                loss_rec = calc_reconstruction_loss(real_batch, rec, loss_type="mse", reduction="mean")

                rec_mu, rec_logvar,_ = self.model.encode(rec)
                z_rec = reparameterize(rec_mu, rec_logvar)

                fake_mu, fake_logvar,_ = self.model.encode(fake)
                z_fake = reparameterize(fake_mu, fake_logvar)

                rec_rec = self.model.decode(z_rec.detach())
                rec_fake = self.model.decode(z_fake.detach())

                loss_rec_rec = calc_reconstruction_loss(rec.detach(), rec_rec, loss_type="mse", reduction="mean")
                loss_fake_rec = calc_reconstruction_loss(fake.detach(), rec_fake, loss_type="mse", reduction="mean")

                lossD_rec_kl = calc_kl(rec_logvar, rec_mu, reduce="mean")
                lossD_fake_kl = calc_kl(fake_logvar, fake_mu, reduce="mean")


                lossD = self.scale * (loss_rec * self.beta_rec + (
                        lossD_rec_kl + lossD_fake_kl) * 0.5 * self.beta_kl + self.gamma_r * 0.5 * self.beta_rec * (
                                         loss_rec_rec + loss_fake_rec))

                self.optimizer_d.zero_grad()
                lossD.backward()
                self.optimizer_d.step()
                if torch.isnan(lossD) or torch.isnan(lossE):
                    print('is non for D')
                    raise SystemError
                if torch.isnan(lossE):
                    print('is non for E')
                    raise SystemError

                diff_kls += -lossE_real_kl.data.cpu().item() + lossD_fake_kl.data.cpu().item() * images.shape[0]
                batch_kls_real += lossE_real_kl.data.cpu().item() * images.shape[0]
                batch_kls_fake += lossD_fake_kl.cpu().item() * images.shape[0]
                batch_kls_rec += lossD_rec_kl.data.cpu().item() * images.shape[0]
                batch_rec_errs += loss_rec.data.cpu().item() * images.shape[0]

                batch_exp_elbo_f += expelbo_fake.data.cpu() * images.shape[0]
                batch_exp_elbo_r += expelbo_rec.data.cpu() * images.shape[0]

                batch_emb += loss_emb.cpu().item() * images.shape[0]

            epoch_loss_d_kls = diff_kls / count_images if count_images > 0 else diff_kls
            epoch_loss_kls_real = batch_kls_real / count_images if count_images > 0 else batch_kls_real
            epoch_loss_kls_fake = batch_kls_fake / count_images if count_images > 0 else batch_kls_fake
            epoch_loss_kls_rec = batch_kls_rec / count_images if count_images > 0 else batch_kls_rec
            epoch_loss_rec_errs = batch_rec_errs / count_images if count_images > 0 else batch_rec_errs
            epoch_loss_exp_f = batch_exp_elbo_f / count_images if count_images > 0 else batch_exp_elbo_f
            epoch_loss_exp_r = batch_exp_elbo_r / count_images if count_images > 0 else batch_exp_elbo_r
            epoch_loss_emb = batch_emb / count_images if count_images > 0 else batch_emb
            epoch_losses.append(epoch_loss_rec_errs)

            end_time = time()
            print('Epoch: {} \tTraining Loss: {:.6f} , computed in {} seconds for {} samples'.format(
                epoch, epoch_loss_rec_errs, end_time - start_time, count_images))
            wandb.log({"Train/Loss_DKLS": epoch_loss_d_kls, '_step_': epoch})
            wandb.log({"Train/Loss_REAL": epoch_loss_kls_real, '_step_': epoch})
            wandb.log({"Train/Loss_FAKE": epoch_loss_kls_fake, '_step_': epoch})
            wandb.log({"Train/Loss_REC": epoch_loss_kls_rec, '_step_': epoch})
            wandb.log({"Train/Loss_REC_ERRS": epoch_loss_rec_errs, '_step_': epoch})
            wandb.log({"Train/Loss_EXP_F": epoch_loss_exp_f, '_step_': epoch})
            wandb.log({"Train/Loss_EXP_R": epoch_loss_exp_r, '_step_': epoch})
            wandb.log({"Train/Loss_EMB": epoch_loss_emb, '_step_': epoch})


            # Save latest model
            torch.save({'model_weights': self.model.state_dict(), 'optimizer_weights': self.optimizer.state_dict()
                           , 'epoch': epoch}, self.client_path + '/latest_model.pt')


            img = transformed_images[0].cpu().detach().numpy()
            # print(np.min(img), np.max(img))
            rec_ = rec[0].cpu().detach().numpy()
            # print(f'rec: {np.min(rec)}, {np.max(rec)}')
            elements = [img, rec_, np.abs(rec_ - img)]
            v_maxs = [1, 1, 0.5]
            diffp, axarr = plt.subplots(1, len(elements), gridspec_kw={'wspace': 0, 'hspace': 0})
            diffp.set_size_inches(len(elements) * 4, 4)
            for i in range(len(axarr)):
                axarr[i].axis('off')
                v_max = v_maxs[i]
                c_map = 'gray' if v_max == 1 else 'inferno'
                axarr[i].imshow(elements[i].transpose(1, 2, 0), vmin=0, vmax=v_max, cmap=c_map)

            wandb.log({'Train/Example_': [
                wandb.Image(diffp, caption="Iteration_" + str(epoch))]})


            self.test(self.model.state_dict(), self.val_ds, 'Val', [self.optimizer_e.state_dict(),
                                                                    self.optimizer_d.state_dict()], epoch)

        return self.best_weights, self.best_opt_weights

    def test(self, model_weights, test_data, task='Val', opt_weights=None, epoch=0):
        """
        :param model_weights: weights of the global model
        :return: dict
            metric_name : value
            e.g.:
             metrics = {
                'test_loss_rec': 0,
                'test_total': 0
            }
        """
        self.test_model.load_state_dict(model_weights)
        self.test_model.to(self.device)
        self.test_model.eval()
        metrics = {
            task + '_loss_rec': 0,
            task + '_loss_mse': 0,
            task + '_loss_pl': 0,
        }
        test_total = 0
        with torch.no_grad():
            for data in test_data:
                x = data[0]
                b, c, h, w = x.shape
                test_total += b
                x = x.to(self.device)

                # Forward pass
                x_, z_rec = self.test_model(x)
                loss_rec = self.criterion_MSE(x_, x)
                loss_mse = self.criterion_MSE(x_, x)
                loss_pl = self.criterion_PL(x_, x)

                metrics[task + '_loss_rec'] += loss_rec.item() * x.size(0)
                metrics[task + '_loss_mse'] += loss_mse.item() * x.size(0)
                metrics[task + '_loss_pl'] += loss_pl.item() * x.size(0)

        img = x[0].cpu().detach().numpy()
        # print(np.min(img), np.max(img))
        rec_ = x_[0].cpu().detach().numpy()
        # print(f'rec: {np.min(rec)}, {np.max(rec)}')
        elements = [img, rec_, np.abs(rec_ - img)]
        v_maxs = [1, 1, 0.5]
        diffp, axarr = plt.subplots(1, len(elements), gridspec_kw={'wspace': 0, 'hspace': 0})
        diffp.set_size_inches(len(elements) * 4, 4)
        for i in range(len(axarr)):
            axarr[i].axis('off')
            v_max = v_maxs[i]
            c_map = 'gray' if v_max == 1 else 'inferno'
            axarr[i].imshow(elements[i].transpose(1, 2, 0), vmin=0, vmax=v_max, cmap=c_map)

        wandb.log({task + '/Example_': [
                wandb.Image(diffp, caption="Iteration_" + str(epoch))]})

        for metric_key in metrics.keys():
            metric_name = task + '/' + str(metric_key)
            metric_score = metrics[metric_key] / test_total
            wandb.log({metric_name: metric_score, '_step_': epoch})
        wandb.log({'lr': self.optimizer_e.param_groups[0]['lr'], '_step_': epoch})
        epoch_val_loss = metrics[task + '_loss_mse'] / test_total
        if task == 'Val':
            if epoch_val_loss < self.min_val_loss:
                self.min_val_loss = epoch_val_loss
                self.best_weights = model_weights
                self.best_opt_weights = opt_weights
                torch.save({'model_weights': model_weights, 'optimizer_e_weights': opt_weights[0],
                            'optimizer_d_weights': opt_weights[1], 'epoch': epoch},
                           self.client_path + '/best_model.pt')
            self.early_stop = self.early_stopping(epoch_val_loss)
            self.e_scheduler.step(epoch_val_loss)
            self.d_scheduler.step(epoch_val_loss)
