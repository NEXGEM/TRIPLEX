
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class StNet(nn.Module):
    
    def __init__(self, num_outputs=1788, max_batch_size=1024):
        super(StNet, self).__init__()
        
        self.max_batch_size = max_batch_size
        self.model = torchvision.models.__dict__['densenet121'](pretrained=True)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_outputs)

    def forward(self, img, label=None, **kwargs):
        phase = kwargs.get('phase', 'train')
        
        if phase == 'train':
            output = self.model(img)
            
        else:
            if img.shape[0] > self.max_batch_size:
                imgs = img.split(self.max_batch_size, dim=0)
                output = [self.model(img) for img in imgs]
                output = torch.cat(output, dim=0)
            else:
                output = self.model(img)
        
        output = torch.clamp(output, 0) 
        
        if label is not None:
            loss = F.mse_loss(output, label)
            return {'loss': loss, 'logits': output}
        else:
            return {'logits': output}
