
import numpy as np

import torch
import torch.nn as nn
import torchvision


class StNet(nn.Module):
    
    def __init__(self, num_genes=1788, backbone='densenet'):
        super(StNet, self).__init__()
        
        self.model = torchvision.models.__dict__['densenet121'](pretrained=True)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_genes)

    def forward(self, img):
        
        out = self.model(img)
    
        return out
