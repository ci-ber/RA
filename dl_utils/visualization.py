import matplotlib.pyplot as plt
import numpy as np
import torch
import umap.umap_ as umap
import seaborn as sns
import logging
import wandb


def umap_plot(self, global_model):
    """
    Validation on all clients after a number of rounds
    Logs results to wandb

    :param global_model:
        Global parameters
    """
    logging.info("################ UMAP PLOT #################")
    self.model.load_state_dict(global_model)
    self.model.eval()
    z = None
    labels = []
    for dataset_key in self.test_data_dict.keys():
        dataset = self.test_data_dict[dataset_key]
        logging.info('DATASET: {}'.format(dataset_key))
        for idx, data in enumerate(dataset):
            nr_batches, nr_slices, width, height = data[0].shape
            x = data[0].view(nr_batches * nr_slices, 1, width, height)
            x = x.to(self.device)
            x_rec, x_rec_dict = self.model(x)
            z = torch.flatten(x_rec_dict['z'], start_dim=1).cpu().detach().numpy() if z is None else \
                np.concatenate((z, torch.flatten(x_rec_dict['z'], start_dim=1).cpu().detach().numpy()), axis=0)
            for i in range(len(x_rec_dict['z'])):
                labels.append(dataset_key)
    z = np.asarray(z)
    labels = np.asarray(labels)
    reducer = umap.UMAP(min_dist=0.6, n_neighbors=25, metric='euclidean', init='random')
    umap_dim = reducer.fit_transform(z)
    sns.set_style("whitegrid")

    # sns.set(context='paper', style='darkgrid', rc={'figure.facecolor':'white'}, font_scale=1.2)
    colors = ['#ec7062', '#af7ac4', '#5599c7', '#48c9b0', '#f5b041']
    # fig = plt.figure()
    # sns.color_palette(colors)
    fig, ax = plt.subplots()
    sns_plot = sns.jointplot(x=umap_dim[:, 1], y=umap_dim[:, 0], hue=labels, palette=sns.color_palette(colors), s=40)
    sns_plot.ax_joint.legend(loc='center right', bbox_to_anchor=(-0.2, 0.5))
    sns_plot.savefig(self.checkpoint_path + "/output_umap.pdf")
    wandb.log({"umap_plot": fig})
    # wandb.log({"umap_image": [wandb.Image(sns_plot, caption="UMAP_Image")]})

    # sns_plot = sns.jointplot(x=tsne[:,1], y=tsne[:,0], hue=labels, palette="deep", s=50)
