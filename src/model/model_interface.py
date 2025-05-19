
import os
import inspect
import importlib

#---->
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torchmetrics
from torchmetrics.regression import ( PearsonCorrCoef, 
                                    ConcordanceCorrCoef, 
                                    MeanSquaredError,
                                    MeanAbsoluteError,
                                    ExplainedVariance )  

#---->
import pytorch_lightning as pl

from utils.metrics import RVDMetric


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
            
        if self.config.DATA.mode == 'inference':
            self.predictions = []
        
        self.load_model()

        metrics = torchmetrics.MetricCollection([PearsonCorrCoef(num_outputs = num_outputs),
                                                ConcordanceCorrCoef(num_outputs = num_outputs),
                                                MeanSquaredError(num_outputs = num_outputs),
                                                MeanAbsoluteError(num_outputs = num_outputs),
                                                ExplainedVariance()
                                                ])
        self.test_metrics = metrics.clone(prefix = 'test_')        
        if target:
            metrics['target'] = metrics.pop(target)
            idx_target = {v[0]: k for k,v in metrics.compute_groups.items()}[target]
            metrics.compute_groups[idx_target] = ['target']
        self.valid_metrics = metrics.clone(prefix = 'val_')
        
        if os.path.exists(f"{self.config.DATA.output_path}/idx_top.npy"):
            num_hpg = np.load(f"{self.config.DATA.output_path}/idx_top.npy").shape[0]
            self.test_metrics_hpg = torchmetrics.MetricCollection([PearsonCorrCoef(num_outputs = num_hpg),
                                                    ConcordanceCorrCoef(num_outputs = num_hpg),
                                                    MeanSquaredError(num_outputs = num_hpg),
                                                    MeanAbsoluteError(num_outputs = num_hpg),
                                                    ExplainedVariance()
                                                    ]).clone(prefix = 'test_', postfix='_hpg')
        
        self.avg_pcc = torch.zeros(num_outputs)
        
    #---->remove v_num
    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items
    
    def _preprocess_inputs(self, inputs):
        if len(inputs['img'].shape) == 5:
            inputs['img'] = inputs['img'].squeeze(0)
        if 'label' in inputs and len(inputs['label'].shape) == 3:
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
        if 'ei' in inputs and len(inputs['ei'].shape) == 4:
            inputs['ei'] = inputs['ei'].squeeze(0)
        if 'ej' in inputs and len(inputs['ej'].shape) == 4:
            inputs['ej'] = inputs['ej'].squeeze(0)
        if 'yj' in inputs and len(inputs['yj'].shape) == 4:
            inputs['yj'] = inputs['yj'].squeeze(0)
        if 'img_emb' in inputs and len(inputs['img_emb'].shape) == 3:
            inputs['img_emb'] = inputs['img_emb'].squeeze(0)
        if 'coord' in inputs and len(inputs['coord'].shape) == 3:
            inputs['coord'] = inputs['coord'].squeeze(0)
        if 'pred' in inputs and len(inputs['pred'].shape) == 3:
            inputs['pred'] = inputs['pred'].squeeze(0)
        
        return inputs

    def training_step(self, batch, batch_idx):
        batch = self._preprocess_inputs(batch)
        
        #---->Forward
        if self.model_name == 'TRIPLEX':
            dataset = self._trainer.train_dataloader.dataset
            batch['dataset'] = dataset
                
        results_dict = self.model(**batch, phase='train')
        
        #---->Loss
        loss = results_dict['loss']        
        self.log("train_loss", loss)
        
        return {'loss': loss} 
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        # for name, param in self.named_parameters():
        #     if param.grad is None:
        #         print(f"Parameter {name} is unused (no grad)")
        return super().on_train_batch_end(outputs, batch, batch_idx)
    
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        
        if self.model_name == 'Stem':
            self.model.update_ema()
            
        if getattr(self.model, 'ema', None) is not None:
            self.model.ema.update()

    def validation_step(self, batch, batch_idx):
        batch = self._preprocess_inputs(batch)
        
        #---->Forward
        results_dict = self.model(**batch, phase='val')
        
        label = batch['label']
        #---->Loss
        if 'logits' in results_dict:
            logits = results_dict['logits']
            
            val_metric = self.valid_metrics(logits, label)
            val_metric = {k: v.nanmean() if len(v.shape) > 0 else v for k, v in val_metric.items()}
            
            self.log_dict(val_metric, on_epoch = True, logger = True, sync_dist=True, batch_size = label.shape[0])
            outputs = {'logits': logits, 'label': label}
        else:
            loss = results_dict['loss']
            self.log_dict({'val_target': loss}, on_epoch = True, logger = True, sync_dist=True, batch_size = label.shape[0])
            outputs = {'loss': loss}
        
        return outputs

    def test_step(self, batch, batch_idx):
        batch = self._preprocess_inputs(batch)
        
        #---->Forward
        if self.model_name == 'BLEEP':
            dataset = self._trainer.test_dataloaders.dataset
            batch['dataset'] = dataset
            
        results_dict = self.model(**batch, phase='test')
        
        #---->Loss
        logits = results_dict['logits']
        label = batch['label']

        test_metric = self.test_metrics(logits, label)
        if os.path.exists(f"{self.config.DATA.output_path}/idx_top.npy"):
            idx_top = np.load(f"{self.config.DATA.output_path}/idx_top.npy")
            idx_top = torch.tensor(idx_top).to(logits.device)
            test_metric_hpg = self.test_metrics_hpg(logits[:, idx_top], label[:, idx_top])
            test_metric_hpg = {k: v.nanmean() if len(v.shape) > 0 else v for k, v in test_metric_hpg.items()}
            test_metric_hpg['epoch'] = self.step_epoch
            self.log_dict(test_metric_hpg, 
                          on_epoch = True, 
                          logger=True,
                          sync_dist=True, 
                          batch_size = label.shape[0])
        else:
            self.avg_pcc += test_metric['test_PearsonCorrCoef'].cpu()
            
        test_metric = {k: v.nanmean() if len(v.shape) > 0 else v for k, v in test_metric.items()}
        test_metric['epoch'] = self.step_epoch
        self.log_dict(test_metric, on_epoch = True, logger = True, sync_dist=True, batch_size = label.shape[0])
        
        if self.config.GENERAL.save_predictions:
            self.save_predictions(logits, batch_idx)
        
        outputs = {'logits': logits, 'label': label}
        
        return outputs

    def on_test_epoch_end(self):
        parent_dir = "/".join(self.config.DATA.output_path.split('/')[:-1])
        if not os.path.exists(f"{parent_dir}/idx_top.npy"):
            print("on_test_epoch_end")
            pcc_rank = torch.argsort(torch.argsort(self.avg_pcc, dim=-1), dim=-1) + 1
            np.save(f"{self.config.DATA.output_path}/pcc_rank.npy", pcc_rank.numpy())
    
    def predict_step(self, batch, batch_idx):
        dataset = self._trainer.predict_dataloaders.dataset
        # _id = dataset.int2id[batch_idx]
        _id = self.config.DATA.data_id
        
        batch = self._preprocess_inputs(batch)
        if self.model_name == 'TRIPLEX':
            batch['position'] = dataset.position.clone().to(batch['img'].device)
            batch['global_emb'] = dataset.global_emb.clone().to(batch['img'].device).unsqueeze(0)
        
        #---->Forward
        results_dict = self.model(**batch, phase='test')
        
        #---->Loss
        pred = results_dict['logits']
        
        self.predictions.append(pred)
        
        return pred, _id
    
    def on_predict_epoch_end(self):
        preds = torch.cat(self.predictions, 0)
        self.save_predictions(preds)
        
        self.predictions = []

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
        
    def save_predictions(self, preds, batch_idx=None):
        if self.config.DATA.mode == 'inference':
            name = self._trainer.predict_dataloaders.dataset.name
            output_path = self.config.DATA.output_path
        else:
            int2id = self._trainer.test_dataloaders.dataset.int2id
            name = int2id[batch_idx]
            genes = self._trainer.test_dataloaders.dataset.genes
            output_path = f"{self.config.DATA.output_path}/{name}.h5ad"
        
        preds = preds.detach().cpu().numpy().astype(np.float32)
        
        adata_pred = sc.AnnData(
            X=preds,
            var=pd.DataFrame(index=genes) if self.config.DATA.mode != 'inference' else None
        )
        adata_pred.write(output_path)
    
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
    
