import sys
import os
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
                                    ConcordanceCorrCoef, 
                                    MeanSquaredError,
                                    MeanAbsoluteError,
                                    ExplainedVariance )  

#---->
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter


class  ModelInterface(pl.LightningModule):

    #---->init
    def __init__(self, model_name=None, **kwargs):
        super(ModelInterface, self).__init__()
        self.save_hyperparameters()
        
        self.model_name = model_name
        self.config = kwargs['config']
        self.model_config = self.config.MODEL
        num_outputs = self.model_config.num_outputs
        
        self.load_model()
        # self.log_path = kargs['log']
        
        # self.validation_step_outputs = []
        # self.test_step_outputs = []
        # self.best_loss = 100000

        metrics = torchmetrics.MetricCollection([PearsonCorrCoef(num_outputs = num_outputs),
                                                ConcordanceCorrCoef(num_outputs = num_outputs),
                                                MeanSquaredError(num_outputs = num_outputs),
                                                MeanAbsoluteError(num_outputs = num_outputs),
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
        if self.model_name == 'TRIPLEX':
            dataset = self._trainer.train_dataloader.dataset
            batch['dataset'] = dataset
            
        results_dict = self.model(**batch)
        
        #---->Loss
        loss = results_dict['loss']
        
        self.log("train_loss", loss)

        return {'loss': loss} 

    def validation_step(self, batch, batch_idx):
        #---->Forward
        results_dict = self.model(**batch)
        
        #---->Loss
        logits = results_dict['logits']
        label = batch['label'].squeeze(0)
        
        val_metric = self.valid_metrics(logits, label)
        # val_metric = {k:v.mean() for k,v in val_metric.items() if len(v.shape) > 0 else k:v}
        val_metric = {k: v.mean() if len(v.shape) > 0 else v for k, v in val_metric.items()}
        self.log_dict(val_metric, on_epoch = True, logger = True, sync_dist=True)
        outputs = {'logits': logits, 'label': label}
        
        # self.validation_step_outputs.append(outputs)
        
        return outputs

    # def on_validation_epoch_end(self):
    #     val_step_outputs = self.validation_step_outputs
        
    #     logits = torch.cat([x['logits'] for x in val_step_outputs], dim = 0)
    #     targets = torch.stack([x['label'] for x in val_step_outputs], dim = 0)
        
    #     val_metric = self.valid_metrics(logits, targets)
    #     self.log_dict(val_metric, on_epoch = True, logger = True)

    def test_step(self, batch, batch_idx):
        #---->Forward
        results_dict = self.model(**batch)
        
        #---->Loss
        logits = results_dict['logits']
        label = batch['label'].squeeze(0)
        
        test_metric = self.test_metrics(logits, label)
        # test_metric = {k:v.mean() for k,v in test_metric.items() if len(v.shape) > 0}
        val_metric = {k: v.mean() if len(v.shape) > 0 else v for k, v in val_metric.items()}
        self.log_dict(test_metric, on_epoch = True, logger = True, sync_dist=True)
        outputs = {'logits': logits, 'label': label}
        # self.validation_step_outputs.append(outputs)
        
        return outputs

    # def on_test_epoch_end(self):
    #     test_step_outputs = self.test_step_outputs
        
    #     logits = torch.cat([x['logits'] for x in test_step_outputs], dim = 0)
    #     targets = torch.stack([x['label'] for x in test_step_outputs], dim = 0)
        
    #     test_metric = self.test_metrics(logits, targets)
    #     self.log_dict(test_metric, on_epoch = True, logger = True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.TRAINING.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode=self.config.TRAINING.lr_scheduler.mode,
            factor=self.config.TRAINING.lr_scheduler.factor,
            patience=self.config.TRAINING.lr_scheduler.patience
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.config.TRAINING.lr_scheduler.monitor
            }
        }
    
    def load_model(self):
        # Change the `trans_unet.py` file name to `TransUnet` class name.
        # Please always name your model file name as `trans_unet.py` and
        # class name or funciton name corresponding `TransUnet`.
        if '_' in self.model_name:
            camel_name = ''.join([i.capitalize() for i in self.model_name.split('_')])
        else:
            camel_name = self.model_name
        try:
            Model = getattr(importlib.import_module(
                f'models.{self.model_name}'), camel_name)
        except:
            raise ValueError('Invalid Module File Name or Invalid Class Name!')
        self.model = self.instancialize(Model)
        pass

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.model_config dictionary. You can also input any args
            to overwrite the corresponding value in self.model_config.
        """
        class_args = inspect.getfullargspec(Model.__init__).args[1:]
        inkeys = self.model_config.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.model_config, arg)
        args1.update(other_args)
        return Model(**args1)
    
    
class CustomWriter(BasePredictionWriter):
    def __init__(self, pred_dir, write_interval, emb_dir=None, names=None):
        super().__init__(write_interval)
        self.pred_dir = pred_dir
        self.emb_dir = emb_dir
        self.names = names

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        for i, batch in enumerate(batch_indices[0]):
            torch.save(predictions[0][i][0], os.path.join(self.pred_dir, f"{self.names[i]}.pt"))
            torch.save(predictions[0][i][1], os.path.join(self.emb_dir, f"{self.names[i]}.pt"))

        # optionally, you can also save `batch_indices` to get the information about the data index
        # from your prediction data
        # torch.save(batch_indices, os.path.join(self.output_dir, f"batch_indices_{trainer.global_rank}.pt"))


