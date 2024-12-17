from cProfile import label
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
        
        self.model_name = model_name
        self.config = kwargs['config']
        self.model_config = self.config.MODEL
        
        num_outputs = self.config.DATA.num_outputs
        target = self.config.TRAINING.monitor
        
        if self.config.DATA.mode == 'cv':
            self.save_hyperparameters()
        
        self.load_model()

        metrics = torchmetrics.MetricCollection([PearsonCorrCoef(num_outputs = num_outputs),
                                                ConcordanceCorrCoef(num_outputs = num_outputs),
                                                MeanSquaredError(num_outputs = num_outputs),
                                                MeanAbsoluteError(num_outputs = num_outputs),
                                                ExplainedVariance()
                                                ])
        self.test_metrics = metrics.clone(prefix = 'test_')        
        
        metrics['target'] = metrics.pop(target)
        idx_target = {v[0]: k for k,v in metrics.compute_groups.items()}[target]
        metrics.compute_groups[idx_target] = ['target']
        self.valid_metrics = metrics.clone(prefix = 'val_')
        

    #---->remove v_num
    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items
    
    def _preprocess_inputs(self, inputs):
        if len(inputs['img'].shape) == 5:
            inputs['img'] = inputs['img'].squeeze(0)
        if len(inputs['label'].shape) == 3:
            inputs['label'] = inputs['label'].squeeze(0)
        if 'mask' in inputs and len(inputs['mask'].shape) == 3:
            inputs['mask'] = inputs['mask'].squeeze(0)
        if 'neighbor_emb' in inputs and len(inputs['neighbor_emb'].shape) == 4:
            inputs['neighbor_emb'] = inputs['neighbor_emb'].squeeze(0)
        if 'position' in inputs and len(inputs['position'].shape) == 3:
            inputs['position'] = inputs['position'].squeeze(0)
        if 'pid' in inputs and len(inputs['pid'].shape) == 2:
            inputs['pid'] = inputs['pid'].view(-1)
        if 'sid' in inputs and len(inputs['sid'].shape) == 2:
            inputs['sid'] = inputs['sid'].view(-1)
        return inputs

    def training_step(self, batch, batch_idx):
        batch = self._preprocess_inputs(batch)
        
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
        batch = self._preprocess_inputs(batch)
        
        #---->Forward
        results_dict = self.model(**batch)
        
        #---->Loss
        if 'logits' in results_dict:
            logits = results_dict['logits']
            label = batch['label']
            
            val_metric = self.valid_metrics(logits, label)
            # val_metric = {k:v.mean() for k,v in val_metric.items() if len(v.shape) > 0 else k:v}
            val_metric = {k: v.mean() if len(v.shape) > 0 else v for k, v in val_metric.items()}
            self.log_dict(val_metric, on_epoch = True, logger = True, sync_dist=True)
            outputs = {'logits': logits, 'label': label}
        else:
            loss = results_dict['loss']
            self.log_dict({'val_target': loss}, on_epoch = True, logger = True, sync_dist=True)
            outputs = {'loss': loss}
        
        # self.validation_step_outputs.append(outputs)
        
        return outputs

    def test_step(self, batch, batch_idx):
        batch = self._preprocess_inputs(batch)
        
        #---->Forward
        if self.model_name == 'BLEEP':
            dataset = self._trainer.test_dataloaders.dataset
            batch['dataset'] = dataset
            
        results_dict = self.model(**batch)
        
        #---->Loss
        logits = results_dict['logits']
        label = batch['label']
        
        test_metric = self.test_metrics(logits, label)
        # val_metric = {k:v.mean() for k,v in val_metric.items() if len(v.shape) > 0 else k:v}
        test_metric = {k: v.mean() if len(v.shape) > 0 else v for k, v in test_metric.items()}
        self.log_dict(test_metric, on_epoch = True, logger = True, sync_dist=True)
        outputs = {'logits': logits, 'label': label}
        
        return outputs

    def predict_step(self, batch, batch_idx):
        batch = self._preprocess_inputs(**batch)
        
        #---->Forward
        results_dict = self.model(**batch)
        
        dataset = self._trainer.predict_dataloaders.dataset
        _id = dataset.int2id[batch_idx]
        
        #---->Loss
        preds = results_dict['logits']
        
        return preds, _id

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.TRAINING.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode=self.config.TRAINING.mode,
            factor=self.config.TRAINING.lr_scheduler.factor,
            patience=self.config.TRAINING.lr_scheduler.patience
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": 'val_target'
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
                f'model.{self.model_name}'), camel_name)
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
    def __init__(self, pred_dir, write_interval):
        super().__init__(write_interval)
        self.pred_dir = pred_dir
        
    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        
        for i, _ in enumerate(batch_indices[0]):
            pred = predictions[i][0]
            name = predictions[i][1]
            torch.save(pred, os.path.join(self.pred_dir, f"{name}.pt"))


