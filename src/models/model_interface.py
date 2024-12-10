import sys
import numpy as np
import inspect
import importlib
import random
import pandas as pd

#---->
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torchmetrics.regression import ( PearsonCorrCoef, 
                                    SpearmanCorrCoef,
                                    ConcordanceCorrCoef, 
                                    MeanSquaredError,
                                    MeanAbsoluteError,
                                    ExplainedVariance )  

#---->
import pytorch_lightning as pl


class  ModelInterface(pl.LightningModule):

    #---->init
    def __init__(self, **kargs):
        super(ModelInterface, self).__init__()
        self.save_hyperparameters()
        self.load_model()
        self.loss = nn.MSELoss()
        
        self.num_output = kargs['num_output']
        # self.log_path = kargs['log']
        
        self.validation_step_outputs = []
        self.test_step_outputs = []
        # self.best_loss = 100000

        metrics = torchmetrics.MetricCollection([PearsonCorrCoef(num_output = self.num_output),
                                                SpearmanCorrCoef(num_output = self.num_output),
                                                ConcordanceCorrCoef(num_output = self.num_output),
                                                MeanSquaredError(num_output = self.num_output),
                                                MeanAbsoluteError(num_output = self.num_output),
                                                ExplainedVariance()
                                                ])
        
        self.valid_metrics = metrics.clone(prefix = 'val_')
        self.test_metrics = metrics.clone(prefix = 'test_')        

    #---->remove v_num
    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def training_step(self, batch, batch_idx):
        #---->Forward
        results_dict = self.model(batch)
        logits = results_dict['logits']
        
        #---->Loss
        label = batch['st']
        loss = self.loss(logits, label)  

        return {'loss': loss} 

    def validation_step(self, batch, batch_idx):
        #---->Forward
        results_dict = self.model(batch)
        logits = results_dict['logits']
        
        label = batch['st']
        outputs = {'logits' : logits, 'label' : label}
        
        self.validation_step_outputs.append(outputs)
        
        return outputs

    def on_validation_epoch_end(self):
        val_step_outputs = self.validation_step_outputs
        
        logits = torch.cat([x['logits'] for x in val_step_outputs], dim = 0)
        target = torch.stack([x['label'] for x in val_step_outputs], dim = 0)
        
        val_metric = self.valid_metrics(logits, target)
        self.log_dict(val_metric, on_epoch = True, logger = True)


    def test_step(self, batch, batch_idx):
        #---->Forward
        results_dict = self.model(batch)
        logits = results_dict['logits']
        
        label = batch['st']
        outputs = {'logits' : logits, 'label' : label}
        
        self.test_step_outputs.append(outputs)
        
        return outputs

    def on_test_epoch_end(self):
        test_step_outputs = self.test_step_outputs
        
        logits = torch.cat([x['logits'] for x in test_step_outputs], dim = 0)
        target = torch.stack([x['label'] for x in test_step_outputs], dim = 0)
        
        test_metric = self.test_metrics(logits, target)
        self.log_dict(test_metric, on_epoch = True, logger = True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=0.1,
            patience=5,
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
    
    def load_model(self):
        name = self.hparams.model.name
        # Change the `trans_unet.py` file name to `TransUnet` class name.
        # Please always name your model file name as `trans_unet.py` and
        # class name or funciton name corresponding `TransUnet`.
        if '_' in name:
            camel_name = ''.join([i.capitalize() for i in name.split('_')])
        else:
            camel_name = name
        try:
            Model = getattr(importlib.import_module(
                f'models.{name}'), camel_name)
        except:
            raise ValueError('Invalid Module File Name or Invalid Class Name!')
        self.model = self.instancialize(Model)
        pass

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.model.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams.model, arg)
        args1.update(other_args)
        return Model(**args1)