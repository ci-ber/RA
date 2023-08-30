import torch.nn as nn
import torchvision


class VGGEncoder(nn.Module):
    """
    VGG Encoder used to extract feature representations for e.g., perceptual losses
    """
    def __init__(self, layers=[4, 9, 16, 23]):
    # def __init__(self, layers=[1, 6, 11, 20]):
        super(VGGEncoder, self).__init__()
        vgg = torchvision.models.vgg16(pretrained=True).features
        self.encoder = nn.ModuleList()
        temp_seq = nn.Sequential()
        for i in range(max(layers) + 1):
            temp_seq.add_module(str(i), vgg[i])
            if i in layers:
                # print(str(i) + ': ' + str(type(vgg[i])))
                self.encoder.append(temp_seq)
                temp_seq = nn.Sequential()

        # resnet = torchvision.models.inception_v3(pretrained=True)
        # modules = list(resnet.children())[:-1]  # delete the last fc layer.
        # self.encoder = nn.Sequential(*modules)

    def forward(self, x):
        features = []
        for layer in self.encoder:
            x = layer(x)
            features.append(x)
        return features
