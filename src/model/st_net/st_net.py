
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class StNet(nn.Module):
    
    def __init__(self, num_outputs=1788, backbone='densenet'):
        super(StNet, self).__init__()
        
        self.model = torchvision.models.__dict__['densenet121'](pretrained=True)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_outputs)

    def forward(self, img, label):
        
        output = self.model(img)
        loss = F.mse_loss(output, label)

        return {'loss': loss, 'logits': output}
