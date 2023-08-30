import logging
from torch.nn import L1Loss
import copy
import torch
import numpy as np
import cv2
#
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
logging.getLogger("matplotlib").setLevel(logging.WARNING)

import wandb
import plotly.graph_objects as go
import seaborn as sns
import umap.umap_ as umap
#
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import roc_auc_score, roc_curve
from skimage import exposure
from scipy.ndimage.filters import gaussian_filter
#
import lpips
from model_zoo.vgg import VGGEncoder
#
from optim.metrics import *
from core.DownstreamEvaluator import DownstreamEvaluator
#


class PDownstreamEvaluator(DownstreamEvaluator):
    """
    Federated Downstream Tasks
        - run tasks training_end, e.g. anomaly detection, reconstruction fidelity, disease classification, etc..
    """
    def __init__(self, name, model, device, test_data_dict, checkpoint_path, global_=True):
        super(PDownstreamEvaluator, self).__init__(name, model, device, test_data_dict, checkpoint_path)

        self.criterion_rec = L1Loss().to(self.device)
        self.compute_scores = True
        self.vgg_encoder = VGGEncoder().to(self.device)
        self.l_pips_sq = lpips.LPIPS(pretrained=True, net='squeeze', use_dropout=True, eval_mode=True, spatial=True, lpips=True).to(self.device)
        self.global_ = False

    def start_task(self, global_model):
        """
        Function to perform analysis after training is complete, e.g., call downstream tasks routines, e.g.
        anomaly detection, classification, etc..

        :param global_model: dict
                   the model weights
        """

        th = 0.009 # RA
        # _ = self.thresholding(global_model)

        if self.global_:
            self.global_detection(global_model)
        else:
            self.object_localization(global_model, th)
        #
        # self.umap_plot(global_model)
        # self.pseudo_healthy(global_model)

    def compute_residual(self, x_rec, x):
        x_rec = torch.clamp(x_rec, 0, 1)
        x = torch.clamp(x,0,1)
        saliency = self.get_saliency(x_rec, x)
        x_rescale = exposure.equalize_adapthist(x.cpu().detach().numpy())
        x_rec_rescale = exposure.equalize_adapthist(x_rec.cpu().detach().numpy())
        saliency2 = self.get_saliency(torch.Tensor(x_rec_rescale).to(x_rec.device), torch.Tensor(x_rescale).to(x_rec.device))
        saliency = saliency * saliency2
        # x_res = np.abs(x_rec.cpu().detach().numpy() - x.cpu().detach().numpy())
        x_res = np.abs(x_rec_rescale - x_rescale)
        return x_res, saliency

    def lpips_loss(self, ph_img, anomaly_img, mode=0):
        def ano_cam(ph_img_, anomaly_img_, mode=0):
            anomaly_img_.requires_grad_(True)
            loss_lpips = self.l_pips_sq(anomaly_img_, ph_img_, normalize=True, retPerLayer=False)
            return loss_lpips.cpu().detach().numpy()
        import lpips

        if len(ph_img.shape) == 2:
            ph_img = torch.unsqueeze(torch.unsqueeze(ph_img, 0), 0)
            anomaly_img = torch.unsqueeze(torch.unsqueeze(anomaly_img, 0), 0)
        ano_map = ano_cam(ph_img, anomaly_img, mode)
        return ano_map

    def get_saliency(self, x_rec, x):
        saliency = self.lpips_loss(x_rec, x)
        saliency = gaussian_filter(saliency, sigma=2)
        return saliency

    def global_detection(self, global_model):
        """
        Validation of downstream tasks
        Logs results to wandb

        :param global_model:
            Global parameters
        """
        logging.info("################ MANIFOLD LEARNING TEST #################")
        lpips_alex = lpips.LPIPS(net='alex')  # best forward scores
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

        self.model.load_state_dict(global_model)
        self.model.eval()
        metrics = {
            'MSE': [],
            'LPIPS': [],
            'SSIM': []
        }
        pred_dict = dict()
        for dataset_key in self.test_data_dict.keys():
            pred_ = []
            label_ = []
            dataset = self.test_data_dict[dataset_key]
            test_metrics = {
                'MSE': [],
                'LPIPS': [],
                'SSIM': []
            }
            logging.info('DATASET: {}'.format(dataset_key))
            for idx, data in enumerate(dataset):
                if type(data) is dict and 'images' in data.keys():
                    data0 = data['images']
                else:
                    data0 = data[0]
                x = data0.to(self.device)
                x_rec, x_rec_dict = self.model(x)
                for i in range(len(x)):
                    count = str(idx * len(x) + i)
                    x_i = x[i][0]
                    x_rec_i = x_rec[i][0]
                    x_res_i = np.abs(x_rec_i.cpu().detach().numpy() - x_i.cpu().detach().numpy()) # default
                    saliency_i = None
                    if 'embeddings' in x_rec_dict.keys():
                        x_res_i, saliency_i = self.compute_residual(x_rec_i, x_i)
                        x_res_orig_i = copy.deepcopy(x_res_i)
                        x_res_i = x_res_i * saliency_i
                    #
                    loss_mse = self.criterion_rec(x_rec_i, x_i)
                    test_metrics['MSE'].append(loss_mse.item())
                    loss_lpips = np.squeeze(lpips_alex(x_i.cpu(), x_rec_i.cpu()).detach().numpy())
                    test_metrics['LPIPS'].append(loss_lpips)
                    #
                    x_ = x_i.cpu().detach().numpy()
                    x_rec_ = x_rec_i.cpu().detach().numpy()
                    # np.save(self.checkpoint_path + '/' + dataset_key + '_' + str(count) + '_rec.npy', x_rec_)

                    ssim_ = ssim(x_rec_, x_, data_range=1.)
                    test_metrics['SSIM'].append(ssim_)

                    res_pred = np.mean(x_res_i)
                    label = 0 if 'Normal' in dataset_key else 1
                    pred_.append(res_pred)
                    label_.append(label)

                    if int(count) % 3 == 0 or int(count) in [67]:
                        elements = [x_, x_rec_, x_res_i]
                        v_maxs = [1, 1, 0.5]
                        titles = ['Input', 'Rec', str(res_pred)]
                        if 'embeddings' in x_rec_dict.keys():
                            elements = [x_, x_rec_, x_res_orig_i, saliency_i, x_res_i]
                            v_maxs = [1, 1, 0.5, 0.5, 0.25]  # , 0.99, 0.25]
                            titles = ['Input', 'Rec', 'Res', 'SAl ' + str(np.max(saliency_i)), 'Combined ' + str(np.max(x_res_i))]

                        diffp, axarr = plt.subplots(1, len(elements), gridspec_kw={'wspace': 0, 'hspace': 0})
                        diffp.set_size_inches(len(elements) * 4, 4)
                        for idx_arr in range(len(axarr)):
                            axarr[idx_arr].axis('off')
                            v_max = v_maxs[idx_arr]
                            c_map = 'gray' if v_max == 1 else 'plasma'
                            axarr[idx_arr].imshow(np.squeeze(elements[idx_arr]), vmin=0, vmax=v_max, cmap=c_map)
                            axarr[idx_arr].set_title(titles[idx_arr])

                            wandb.log({'Anomaly/Example_' + dataset_key + '_' + str(count): [
                                wandb.Image(diffp, caption="Sample_" + str(count))]})

            pred_dict[dataset_key] = (pred_, label_)

            for metric in test_metrics:
                logging.info('{} mean: {} +/- {}'.format(metric, np.nanmean(test_metrics[metric]),
                                                         np.nanstd(test_metrics[metric])))
                metrics[metric].append(test_metrics[metric])

        if self.compute_scores:
            normal_key = 'Normal'
            for key in pred_dict.keys():
                if 'Normal' in key:
                    normal_key = key
                    break
            pred_cxr, label_cxr = pred_dict[normal_key]
            for dataset_key in self.test_data_dict.keys():
                print(f'Running evaluation for {dataset_key}')
                if dataset_key == normal_key:
                    continue
                pred_ood, label_ood = pred_dict[dataset_key]
                predictions = np.asarray(pred_cxr + pred_ood)
                labels = np.asarray(label_cxr + label_ood)
                print('Negative Classes: {}'.format(len(np.argwhere(labels == 0))))
                print('Positive Classes: {}'.format(len(np.argwhere(labels == 1))))
                print('total Classes: {}'.format(len(labels)))
                print('Shapes {} {} '.format(labels.shape, predictions.shape))

                auprc = average_precision_score(labels, predictions)
                print('[ {} ]: AUPRC: {}'.format(dataset_key, auprc))
                auroc = roc_auc_score(labels, predictions)
                print('[ {} ]: AUROC: {}'.format(dataset_key, auroc))

                fpr, tpr, ths = roc_curve(labels, predictions)
                th_95 = np.squeeze(np.argwhere(tpr >= 0.95)[0])
                th_99 = np.squeeze(np.argwhere(tpr >= 0.99)[0])
                fpr95 = fpr[th_95]
                fpr99 = fpr[th_99]
                print('[ {} ]: FPR95: {} at th: {}'.format(dataset_key, fpr95, ths[th_95]))
                print('[ {} ]: FPR99: {} at th: {}'.format(dataset_key, fpr99, ths[th_99]))
        logging.info('Writing plots...')

        for metric in metrics:
            fig_bp = go.Figure()
            x = []
            y = []
            for idx, dataset_values in enumerate(metrics[metric]):
                dataset_name = list(self.test_data_dict)[idx]
                for dataset_val in dataset_values:
                    y.append(dataset_val)
                    x.append(dataset_name)

            fig_bp.add_trace(go.Box(
                y=y,
                x=x,
                name=metric,
                boxmean='sd'
            ))
            title = 'score'
            fig_bp.update_layout(
                yaxis_title=title,
                boxmode='group',  # group together boxes of the different traces for each value of x
                yaxis=dict(range=[0, 1]),
            )
            fig_bp.update_yaxes(range=[0, 1], title_text='score', tick0=0, dtick=0.1, showgrid=False)

            wandb.log({"Reconstruction_Metrics(Healthy)_" + self.name + '_' + str(metric): fig_bp})

    def thresholding(self, global_model):
        """
                Validation of downstream tasks
                Logs results to wandb

                :param global_model:
                    Global parameters
                """
        logging.info("################ Threshold Search #################")
        self.model.load_state_dict(global_model)
        self.model.eval()
        ths = np.linspace(0, 1, endpoint=False, num=1000)
        fprs = dict()
        for th_ in ths:
            fprs[th_] = []
        im_scale = 128 * 128
        for dataset_key in self.test_data_dict.keys():
            dataset = self.test_data_dict[dataset_key]
            logging.info('DATASET: {}'.format(dataset_key))
            for idx, data in enumerate(dataset):
                if type(data) is dict and 'images' in data.keys():
                    data0 = data['images']
                else:
                    data0 = data[0]
                x = data0.to(self.device)
                x_rec, x_rec_dict = self.model(x)
                for i in range(len(x)):
                    x_i = x[i][0]
                    x_rec_i = x_rec[i][0]
                    x_res_i = np.abs(x_rec_i.cpu().detach().numpy() - x_i.cpu().detach().numpy()) # default
                    if 'embeddings' in x_rec_dict.keys():
                        x_res_i, saliency_i = self.compute_residual(x_rec_i, x_i)
                        x_res_i = x_res_i * saliency_i
                    for th_ in ths:
                        fpr = (np.count_nonzero(x_res_i > th_) * 100) / im_scale
                        fprs[th_].append(fpr)
        mean_fprs = []
        for th_ in ths:
            mean_fprs.append(np.mean(fprs[th_]))
        mean_fprs = np.asarray(mean_fprs)
        sorted_idxs = np.argsort(mean_fprs)
        th_1, th_2, best_th = 0, 0, 0
        fpr_1, fpr_2, best_fpr = 0, 0, 0
        for sort_idx in sorted_idxs:
            th_ = ths[sort_idx]
            fpr_ = mean_fprs[sort_idx]
            if fpr_ <= 1:
                th_1 = th_
                fpr_1 = fpr_
            if fpr_ <= 2:
                th_2 = th_
                fpr_2 = fpr_
            if fpr_ <= 5:
                best_th = th_
                best_fpr = fpr_
            else:
                break
        print(f'Th_1: [{th_1}]: {fpr_1} || Th_2: [{th_2}]: {fpr_2} || Th_5: [{best_th}]: {best_fpr}')
        return best_th

    def object_localization(self, global_model, th=0):
        """
        Validation of downstream tasks
        Logs results to wandb

        :param global_model:
            Global parameters
        """
        logging.info("################ Object Localzation TEST #################" + str(th))
        lpips_alex = lpips.LPIPS(net='alex')  # best forward scores
        self.model.load_state_dict(global_model)
        self.model.eval()
        metrics = {
            'MSE': [],
            'LPIPS': [],
            'SSIM': [],
            'TP': [],
            'FP': [],
            'Precision': [],
            'Recall': [],
            'F1': [],
        }
        for dataset_key in self.test_data_dict.keys():

            dataset = self.test_data_dict[dataset_key]
            logging.info(f"#################################### {dataset} ############################################")

            test_metrics = {
                'MSE': [],
                'LPIPS': [],
                'SSIM': [],
                'TP': [],
                'FP': [],
                'Precision': [],
                'Recall': [],
                'F1': [],
            }
            logging.info('DATASET: {}'.format(dataset_key))
            tps, fns, fps = 0, 0, []
            for idx, data in enumerate(dataset):
                if type(data) is dict and 'images' in data.keys():
                    data0 = data['images']
                else:
                    data0 = data[0]
                    data1 = data[1].shape
                x = data0.to(self.device)
                masks_bool = True if len(data1) > 2 else False
                nr_batches, nr_slices, width, height = data0.shape
                # x = data0.view(nr_batches * nr_slices, 1, width, height)
                x = data0.to(self.device)
                masks = data[1][:, 0, :, :].view(nr_batches, 1, width, height).to(self.device)\
                    if masks_bool else None
                neg_masks = data[1][:, 1, :, :].view(nr_batches, 1, width, height).to(self.device)
                neg_masks[neg_masks>0.5] = 1
                neg_masks[neg_masks<1] = 0

                x_rec, x_rec_dict = self.model(x)
                for i in range(len(x)):
                    count = str(idx * len(x) + i)
                    x_i = x[i][0]
                    x_rec_i = x_rec[i][0]
                    x_res_i = np.abs(x_rec_i.cpu().detach().numpy() - x_i.cpu().detach().numpy())  # default
                    saliency_i = None
                    if 'embeddings' in x_rec_dict.keys():
                        x_res_i, saliency_i = self.compute_residual(x_rec_i, x_i)
                        x_res_orig_i = copy.deepcopy(x_res_i)
                        x_res_i = x_res_i * saliency_i

                    mask_ = masks[i][0].cpu().detach().numpy() if masks_bool else None
                    neg_mask_ = neg_masks[i][0].cpu().detach().numpy() if masks_bool else None
                    bboxes = cv2.cvtColor(neg_mask_*255, cv2.COLOR_GRAY2RGB)
                    # thresh_gt = cv2.threshold((mask_*255).astype(np.uint8), 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                    cnts_gt = cv2.findContours((mask_*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cnts_gt = cnts_gt[0] if len(cnts_gt) == 2 else cnts_gt[1]
                    gt_box = []
                    for c_gt in cnts_gt:
                        x, y, w, h = cv2.boundingRect(c_gt)
                        gt_box.append([x, y, x+w, y+h])
                        cv2.rectangle(bboxes, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    #
                    loss_mse = self.criterion_rec(x_rec_i, x_i)
                    test_metrics['MSE'].append(loss_mse.item())
                    loss_lpips = np.squeeze(lpips_alex(x_i.cpu(), x_rec_i.cpu()).detach().numpy())
                    test_metrics['LPIPS'].append(loss_lpips)
                    #
                    x_ = x_i.cpu().detach().numpy()
                    x_rec_ = x_rec_i.cpu().detach().numpy()
                    # np.save(self.checkpoint_path + '/' + dataset_key + '_' + str(count) + '_rec.npy', x_rec_)

                    ssim_ = ssim(x_rec_, x_, data_range=1.)
                    test_metrics['SSIM'].append(ssim_)

                    x_combo = copy.deepcopy(x_res_i)
                    x_combo[x_combo < th] = 0

                    x_pos = x_combo * mask_
                    x_neg = x_combo * neg_mask_
                    res_anomaly = np.sum(x_pos)
                    res_healthy = np.sum(x_neg)
                    # print(np.sum(x_neg), np.sum(x_pos))

                    amount_anomaly = np.count_nonzero(x_pos)
                    amount_mask = np.count_nonzero(mask_)

                    tp = 1 if amount_anomaly > 0.1 * amount_mask else 0
                    tps += tp
                    fn = 1 if tp == 0 else 0
                    fns += fn

                    fp = int(res_healthy / max(res_anomaly,1)) #[i for i in ious if i < 0.1]
                    fps.append(fp)
                    precision = tp / max((tp+fp), 1)
                    test_metrics['TP'].append(tp)
                    test_metrics['FP'].append(fp)
                    test_metrics['Precision'].append(precision)
                    test_metrics['Recall'].append(tp)
                    test_metrics['F1'].append(2 * (precision * tp) / (precision + tp + 1e-8))

                    ious = [res_anomaly, res_healthy]

                    if (idx % 5) == 0: # and (i % 5 == 0) or int(count)==13600 or int(count)==40:
                        elements = [x_, x_rec_, x_res_i, x_combo]
                        v_maxs = [1, 1, 0.5, 0.5]
                        titles = ['Input', 'Rec', str(ious), '5%FPR']
                        if 'embeddings' in x_rec_dict.keys():
                            elements = [x_, x_rec_, x_res_orig_i, saliency_i, x_res_i, x_combo]
                            v_maxs = [1, 1, 0.5, 0.75, 0.35, 0.35]#, 0.99, 0.25]
                            titles = ['Input', 'Rec', 'Residual', 'Perc. Res', 'Combined', 'Thresholded']

                        if masks_bool:
                            elements.append(bboxes.astype(np.int64))
                            elements.append(x_pos)
                            elements.append(x_neg)
                            v_maxs.append(1)
                            v_maxs.append(np.max(x_res_i))
                            v_maxs.append(np.max(x_res_i))
                            titles.append('GT')
                            titles.append(str(np.round(res_anomaly, 2)) + ', TP: ' + str(tp))
                            titles.append(str(np.round(res_healthy, 2)) + ', FP: ' + str(fp))
                        diffp, axarr = plt.subplots(1, len(elements), gridspec_kw={'wspace': 0, 'hspace': 0})
                        diffp.set_size_inches(len(elements) * 4, 4)
                        for idx_arr in range(len(axarr)):
                            axarr[idx_arr].axis('off')
                            v_max = v_maxs[idx_arr]
                            c_map = 'gray' if v_max == 1 else 'plasma'
                            axarr[idx_arr].imshow(np.squeeze(elements[idx_arr]), vmin=0, vmax=v_max, cmap=c_map)
                            axarr[idx_arr].set_title(titles[idx_arr])

                            wandb.log({'Anomaly/Example_' + dataset_key + '_' + str(count): [
                                wandb.Image(diffp, caption="Sample_" + str(count))]})

            for metric in test_metrics:
                logging.info('{} mean: {} +/- {}'.format(metric, np.nanmean(test_metrics[metric]),
                                                         np.nanstd(test_metrics[metric])))
                if metric == 'TP':
                    logging.info(f'TP: {np.sum(test_metrics[metric])} of {len(test_metrics[metric])} detected')
                if metric == 'FP':
                    logging.info(f'FP: {np.sum(test_metrics[metric])} missed')
                metrics[metric].append(test_metrics[metric])

            logging.info("################################################################################")

        logging.info('Writing plots...')

        for metric in metrics:
            fig_bp = go.Figure()
            x = []
            y = []
            for idx, dataset_values in enumerate(metrics[metric]):
                dataset_name = list(self.test_data_dict)[idx]
                for dataset_val in dataset_values:
                    y.append(dataset_val)
                    x.append(dataset_name)

            fig_bp.add_trace(go.Box(
                y=y,
                x=x,
                name=metric,
                boxmean='sd'
            ))
            title = 'score'
            fig_bp.update_layout(
                yaxis_title=title,
                boxmode='group',  # group together boxes of the different traces for each value of x
                yaxis=dict(range=[0, 1]),
            )
            fig_bp.update_yaxes(range=[0, 1], title_text='score', tick0=0, dtick=0.1, showgrid=False)

            wandb.log({"Reconstruction_Metrics(Healthy)_" + self.name + '_' + str(metric): fig_bp})

    def umap_plot(self, global_model):
        compute_umap = True
        plot_saliency = False
        """
                Validation of downstream tasks
                Logs results to wandb

                :param global_model:
                    Global parameters
                """
        logging.info("################ TSNE TEST #################")
        self.model.load_state_dict(global_model)
        self.model.eval()
        z = None
        z_ = None
        labels = []
        labels_ = []
        idx_global = 0
        for dataset_key in self.test_data_dict.keys():
            dataset = self.test_data_dict[dataset_key]
            logging.info('DATASET: {}'.format(dataset_key))
            for idx, data in enumerate(dataset):
                count = 0
                if 'dict' in str(type(data)) and 'images' in data.keys():
                    data0 = data['images']
                    data1 = [0]
                else:
                    data0 = data[0]
                    data1 = data[1].shape
                # print(data[1].shape)
                nr_batches, nr_slices, width, height = data0.shape
                # x = data0.view(nr_batches * nr_slices, 1, width, height)
                x = data0.to(self.device)
                x_rec, x_rec_dict = self.model(x, deterministic=False)
                embedding = x_rec_dict['z']

                if compute_umap:
                    z = torch.flatten(embedding, start_dim=1).cpu().detach().numpy() if z is None else \
                        np.concatenate((z, torch.flatten(embedding, start_dim=1).cpu().detach().numpy()), axis=0)
                    for i in range(len(embedding)):
                        labels.append(dataset_key)
                x_rec_rec, x_rec_rec_dict = self.model(x_rec.detach())

                embedding_ph = x_rec_rec_dict['z']
                if compute_umap:
                    z = torch.flatten(embedding_ph, start_dim=1).cpu().detach().numpy() if z is None else \
                        np.concatenate((z, torch.flatten(embedding_ph, start_dim=1).cpu().detach().numpy()), axis=0)
                    for i in range(len(embedding_ph)):
                        labels.append(dataset_key + '_PH')
                # if compute_umap:
                #     z_ = torch.flatten(torch.abs(embedding_ph-embedding), start_dim=1).cpu().detach().numpy() if z is None else \
                #         np.concatenate((z, torch.flatten(torch.abs(embedding_ph-embedding), start_dim=1).cpu().detach().numpy()), axis=0)
                #     labels_.append(dataset_key)
                elements = [x.cpu().detach().numpy(), x_rec.cpu().detach().numpy()]
                v_maxs = [1, 1]
                titles = ['Input_' + str(idx_global), 'Rec_' + str(idx_global+1)]

                diffp, axarr = plt.subplots(1, len(elements), gridspec_kw={'wspace': 0, 'hspace': 0})
                diffp.set_size_inches(len(elements) * 4, 4)
                for idx_arr in range(len(axarr)):
                    axarr[idx_arr].axis('off')
                    v_max = v_maxs[idx_arr]
                    c_map = 'gray' if v_max == 1 else 'inferno'
                    axarr[idx_arr].imshow(np.squeeze(elements[idx_arr]), vmin=0, vmax=v_max, cmap=c_map)
                    axarr[idx_arr].set_title(titles[idx_arr])

                    wandb.log({'Anomaly/Example_' + dataset_key + '_' + str(idx_global): [
                        wandb.Image(diffp, caption="Sample_" + str(idx_global))]})

                idx_global += 1

        if compute_umap:
            z = np.asarray(z)
            labels = np.asarray(labels)
            reducer = umap.UMAP(min_dist=0.6, n_neighbors=25, metric='euclidean', init='random')
            umap_dim = reducer.fit_transform(z)

            for idx, label  in  enumerate(labels):
                print(f' {str(idx)}: ({label}),  [{str(umap_dim[idx, 1])}, {str(umap_dim[idx, 0])}]')
            sns.set_style("whitegrid")

            # sns.set(context='paper', style='darkgrid', rc={'figure.facecolor':'white'}, font_scale=1.2)
            colors = ['#acce81', '#8c9fc5', '#d5121b', '#4766a2', '#ff7373', '#76b6b6', '#800000', '#468499']
            # colors = ['#acce81', '#d5121b', '#ff7373', '#800000']

            # fig = plt.figure()
            # sns.color_palette(colors)
            fig, ax = plt.subplots()
            sns_plot = sns.jointplot(x=umap_dim[:, 1], y=umap_dim[:, 0], hue=labels, palette=sns.color_palette(colors),
                                     s=40)
            sns_plot.ax_joint.legend(loc='center right', bbox_to_anchor=(-0.2, 0.5))
            sns_plot.savefig(self.checkpoint_path + "/output_umap.pdf")
            wandb.log({"umap_plot": fig})
            logging.info('DONE')
            # wandb.log({"umap_image": [wandb.Image(sns_plot, caption="UMAP_Image")]})

            # sns_plot = sns.jointplot(x=tsne[:,1], y=tsne[:,0], hue=labels, palette="deep", s=50)

