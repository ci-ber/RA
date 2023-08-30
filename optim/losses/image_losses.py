import torch
import torch.nn.functional as F
import numpy as np
import math
from model_zoo.vgg import VGGEncoder
from torch.nn.modules.loss import _Loss
from torchvision import transforms
from skimage import color
from scipy.ndimage.filters import gaussian_filter


class PerceptualLoss(_Loss):
    """
    """

    def __init__(
        self,
        reduction: str = 'mean',
        device: str = 'gpu') -> None:
        """
        Args
            reduction: str, {'none', 'mean', 'sum}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.
                - 'none': no reduction will be applied.
                - 'mean': the sum of the output will be divided by the number of elements in the output.
                - 'sum': the output will be summed.
        """
        super().__init__()
        self.device = device
        self.reduction = reduction
        self.loss_network = VGGEncoder().eval().to(self.device)

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        Args:
            input: (N,*),
                where N is the batch size and * is any number of additional dimensions.
            target (N,*),
                same shape as input.
        """
        input_features = self.loss_network(input.repeat(1, 3, 1, 1)) if input.shape[1] == 1 else input
        output_features = self.loss_network(target.repeat(1, 3, 1, 1)) if target.shape[1] == 1 else target

        loss_pl = 0
        for output_feature, input_feature in zip(output_features, input_features):
            loss_pl += F.mse_loss(output_feature, input_feature)
        return loss_pl


class EmbeddingLoss(torch.nn.Module):
    def __init__(self):
        super(EmbeddingLoss, self).__init__()
        self.criterion = torch.nn.MSELoss()
        self.similarity_loss = torch.nn.CosineSimilarity()

    def forward(self, teacher_embeddings, student_embeddings):
        # print(f'LEN {len(output_real)}')
        layer_id = 0
        # teacher_embeddings = teacher_embeddings[:-1]
        # student_embeddings = student_embeddings[3:-1]
        # print(f' Teacher: {len(teacher_embeddings)}, Student: {len(student_embeddings)}')
        for teacher_feature, student_feature in zip(teacher_embeddings, student_embeddings):
            if layer_id == 0:
                total_loss = 0.5 * self.criterion(teacher_feature, student_feature)
            else:
                total_loss += 0.5 * self.criterion(teacher_feature, student_feature)
            total_loss += torch.mean(1 - self.similarity_loss(teacher_feature.view(teacher_feature.shape[0], -1),
                                                         student_feature.view(student_feature.shape[0], -1)))
            layer_id += 1
        return total_loss