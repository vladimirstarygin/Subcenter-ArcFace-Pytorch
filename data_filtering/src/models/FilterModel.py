import torch
from torch import nn 
import torchvision.models as models

class Filtermodel(nn.Module):
    '''
        small embedding dim due to million classes
    '''

    def __init__(self,
                 n_classes,
                 embedding_dim = 2048,
                 backbone='resnet50',
                 pseudolabels=False):

        super(Filtermodel, self).__init__()

        self.pseudolabels = pseudolabels


        if backbone == 'resnet50':
            net = models.resnet50(pretrained=True)  
        elif backbone == 'resnet101':
            net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)

        self.embedder = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
            net.maxpool,
            net.layer1,
            net.layer2,
            net.layer3,
            net.layer4,
            net.avgpool
        )

    def forward(self,x):
        features = self.embedder(x).squeeze(-1).squeeze(-1)
        return features
