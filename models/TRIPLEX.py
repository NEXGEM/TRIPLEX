
import os 
import inspect
import importlib

from pathlib import Path
import wget
import numpy as np
from scipy.stats import pearsonr
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import torch.nn.functional as F
from einops import rearrange

from models.module import GlobalEncoder, NeighborEncoder, FusionEncoder


def load_model_weights(path: str):       
        """Load pretrained ResNet18 model without final fc layer.

        Args:
            path (str): path_for_pretrained_weight

        Returns:
            torchvision.models.resnet.ResNet: ResNet model with pretrained weight
        """
        
        resnet = torchvision.models.__dict__['resnet18'](weights=None)
        
        ckpt_dir = Path('./weights')
        if not ckpt_dir.exists():
            ckpt_dir.mkdir()
        ckpt_path = ckpt_dir / 'tenpercent_resnet18.ckpt'
        
        # prepare the checkpoint
        if not ckpt_path.exists():
            ckpt_url='https://github.com/ozanciga/self-supervised-histopathology/releases/download/tenpercent/tenpercent_resnet18.ckpt'
            wget.download(ckpt_url, out=ckpt_dir)
            
        state = torch.load(path)
        state_dict = state['state_dict']
        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)
        
        model_dict = resnet.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        if state_dict == {}:
            print('No weight could be loaded..')
        model_dict.update(state_dict)
        resnet.load_state_dict(model_dict)
        resnet.fc = nn.Identity()

        return resnet
    

class TRIPLEX(pl.LightningModule):
    """Model class for TRIPLEX
    """
    def __init__(self, 
                num_genes=250,
                emb_dim=512,
                depth1=2,
                depth2=2,
                depth3=2,
                num_heads1=8,
                num_heads2=8,
                num_heads3=8,
                mlp_ratio1=2.0,
                mlp_ratio2=2.0,
                mlp_ratio3=2.0,
                dropout1=0.1,
                dropout2=0.1,
                dropout3=0.1,
                kernel_size=3,
                res_neighbor=(5,5),
                learning_rate= 0.0001):
        """TRIPLEX model 

        Args:
            num_genes (int): Number of genes to predict.
            emb_dim (int): Embedding dimension for images. Defaults to 512.
            depth1 (int): Depth of FusionEncoder. Defaults to 2.
            depth2 (int): Depth of GlobalEncoder. Defaults to 2.
            depth3 (int): Depth of NeighborEncoder. Defaults to 2.
            num_heads1 (int): Number of heads for FusionEncoder. Defaults to 8.
            num_heads2 (int): Number of heads for GlobalEncoder. Defaults to 8.
            num_heads3 (int): Number of heads for NeighborEncoder. Defaults to 8.
            mlp_ratio1 (float): mlp_ratio (MLP dimension/emb_dim) for FusionEncoder. Defaults to 1.0.
            mlp_ratio2 (float): mlp_ratio (MLP dimension/emb_dim) for GlobalEncoder. Defaults to 512.
            mlp_ratio3 (float): mlp_ratio (MLP dimension/emb_dim) for NeighborEncoder. Defaults to 512.
            dropout1 (float): Dropout rate for FusionEncoder. Defaults to 0.1.
            dropout2 (float): Dropout rate for GlobalEncoder. Defaults to 0.1.
            dropout3 (float): Dropout rate for NeighborEncoder. Defaults to 0.1.
            kernel_size (int): Kernel size of convolution layer in PEGH. Defaults to 3.
        """
        
        super().__init__()
        
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        
        # Initialize best metrics
        self.best_loss = np.inf
        self.best_cor = -1
        
        self.num_genes = num_genes
        self.alpha = 0.3
        self.num_n = res_neighbor[0]
    
        # Target Encoder
        resnet18 = load_model_weights("weights/tenpercent_resnet18.ckpt")
        module=list(resnet18.children())[:-2]
        self.target_encoder = nn.Sequential(*module)
        self.fc_target = nn.Linear(emb_dim, num_genes)

        # Neighbor Encoder
        self.neighbor_encoder = NeighborEncoder(emb_dim, depth3, num_heads3, mlp_ratio3, dropout = dropout3, resolution=res_neighbor)
        self.fc_neighbor = nn.Linear(emb_dim, num_genes)

        # Global Encoder        
        self.global_encoder = GlobalEncoder(emb_dim, depth2, num_heads2, emb_dim*mlp_ratio2, dropout2, kernel_size)
        self.fc_global = nn.Linear(emb_dim, num_genes)
    
        # Fusion Layer
        self.fusion_encoder = FusionEncoder(emb_dim, depth1, num_heads1, emb_dim*mlp_ratio1, dropout1)    
        self.fc = nn.Linear(emb_dim, num_genes)
    
    
    def forward(self, x, x_total, position, neighbor, mask, pid=None, sid=None):
        """Forward pass of TRIPLEX

        Args:
            x (torch.Tensor): Target spot image (batch_size x 3 x 224 x 224)
            x_total (list): Extracted features of all the spot images in the patient. (batch_size * (num_spot x 512))
            position (list): Relative position coordinates of all the spots. (batch_size * (num_spot x 2))
            neighbor (torch.Tensor): Neighbor spot features. (batch_size x num_neighbor x 512)
            mask (torch.Tensor): Masking table for neighbor spot. (batch_size x num_neighbor)
            pid (torch.LongTensor, optional): Patient index. Defaults to None. (batch_size x 1)
            sid (torch.LongTensor, optional): Spot index of the patient. Defaults to None. (batch_size x 1)

        Returns:
            tuple:
                out: Prediction of fused feature
                out_target: Prediction of TEM
                out_neighbor: Prediction of NEM
                out_global: Prediction of GEM
        """
        
        # Target tokens
        target_token = self.target_encoder(x) # B x 512 x 7 x 7
        _, dim, w, h = target_token.shape
        target_token = rearrange(target_token, 'b d h w -> b (h w) d', d = dim, w=w, h=h)
    
        # Neighbor tokens
        neighbor_token = self.neighbor_encoder(neighbor, mask) # B x 26 x 384
        
        # Global tokens
        if pid == None:
            global_token = self.global_encoder(x_total, position.squeeze()).squeeze()  # N x 384
            if sid != None:
                global_token = global_token[sid]
        else:
            pid = pid.view(-1)
            sid = sid.view(-1)
            global_token = torch.zeros((len(x_total), x_total[0].shape[1])).to(x.device)
            
            pid_unique = pid.unique()
            for pu in pid_unique:
                ind = int(torch.argmax((pid == pu).int()))
                x_g = x_total[ind].unsqueeze(0) # 1 x N x 384
                pos = position[ind]
                
                emb = self.global_encoder(x_g, pos).squeeze() 
                global_token[pid == pu] = emb[sid[pid == pu]].float()
    
        # Fusion tokens
        fusion_token = self.fusion_encoder(target_token, neighbor_token, global_token, mask=mask) # B x 384
            
        output = self.fc(fusion_token) # B x num_genes
        out_target = self.fc_target(target_token.mean(1)) # B x num_genes
        out_neighbor = self.fc_neighbor(neighbor_token.mean(1)) # B x num_genes
        out_global = self.fc_global(global_token) # B x num_genes

        return output, out_target, out_neighbor, out_global
    
    
    def training_step(self, batch, batch_idx):
        """Train the model. Transfer knowledge from fusion to each module.

        """
        patch, exp, pid, sid, wsi, position, neighbor, mask  = batch
        
        outputs = self(patch, wsi, position, neighbor, mask, pid, sid)
        
        # Fusion loss
        loss = F.mse_loss(outputs[0].view_as(exp), exp)                   # Supervised loss for Fusion
        
        # Target loss
        loss += F.mse_loss(outputs[1].view_as(exp), exp) * (1-self.alpha) # Supervised loss for Target
        loss += F.mse_loss(outputs[0], outputs[1]) * self.alpha           # Distillation loss for Target
    
        # Neighbor loss
        loss += F.mse_loss(outputs[2].view_as(exp), exp) * (1-self.alpha) # Supervised loss for Neighbor
        loss += F.mse_loss(outputs[0], outputs[2]) * self.alpha           # Distillation loss for Neighbor
    
        # Global loss
        loss += F.mse_loss(outputs[3].view_as(exp), exp) * (1-self.alpha) # Supervised loss for Global
        loss += F.mse_loss(outputs[0], outputs[3]) * self.alpha           # Distillation loss for Global
            
        self.log('train_loss', loss, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validating the model in a sample. Calucate MSE and PCC for all spots in the sample.

        Returns:
            dict: 
                val_loss: MSE loss between pred and label
                corr: PCC between pred and label (across genes)
        """
        patch, exp, _, wsi, position, name, neighbor, mask = batch
        patch, exp, neighbor, mask = patch.squeeze(), exp.squeeze(), neighbor.squeeze(), mask.squeeze()
        
        outputs = self(patch, wsi, position, neighbor, mask)
        
        pred = outputs[0]
        loss = F.mse_loss(pred.view_as(exp), exp)

        pred=pred.cpu().numpy().T
        exp=exp.cpu().numpy().T        
        r=[]
        for g in range(self.num_genes):
            r.append(pearsonr(pred[g],exp[g])[0])
        rr = torch.Tensor(r)
        
        self.get_meta(name)
        
        return {"val_loss":loss, "corr":rr}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack(
            [x["val_loss"] for x in outputs]).mean()
        
        avg_corr = torch.stack(
            [x["corr"] for x in outputs])
        
        os.makedirs(f"results/{self.__class__.__name__}/{self.data}", exist_ok=True)
        if self.best_cor < avg_corr.mean():
            torch.save(avg_corr.cpu(), f"results/{self.__class__.__name__}/{self.data}/R_{self.patient}")
            torch.save(avg_loss.cpu(), f"results/{self.__class__.__name__}/{self.data}/loss_{self.patient}")
            
            self.best_cor = avg_corr.mean()
            self.best_loss = avg_loss
        
        self.log('valid_loss', avg_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('R', avg_corr.nanmean(), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        """Testing the model in a sample. 
        Calucate MSE, MAE and PCC for all spots in the sample.

        Returns:
            dict:
                MSE: MSE loss between pred and label
                MAE: MAE loss between pred and label
                corr: PCC between pred and label (across genes)
        """
        patch, exp, sid, wsi, position, name, neighbor, mask = batch
        patch, exp, sid, neighbor, mask = patch.squeeze(), exp.squeeze(), sid.squeeze(), neighbor.squeeze(), mask.squeeze()
        
        if '10x_breast' in name[0]:
            wsi = wsi[0].unsqueeze(0)
            position = position[0]
            
            outputs = self(patch, wsi, position, neighbor.squeeze(), mask.squeeze(), sid=sid)
            pred = outputs[0]
            
            ind_match = np.load(f'/data/temp/TRIPLEX/data/test/{name[0]}/ind_match.npy', allow_pickle=True)
            self.num_genes = len(ind_match)
            pred = pred[:,ind_match]
            
        else:        
            outputs = self(patch, wsi, position, neighbor.squeeze(), mask.squeeze())
            pred = outputs[0]
            
        mse = F.mse_loss(pred.view_as(exp), exp)
        mae = F.l1_loss(pred.view_as(exp), exp)
        
        pred=pred.cpu().numpy().T
        exp=exp.cpu().numpy().T
        
        r=[]
        for g in range(self.num_genes):
            r.append(pearsonr(pred[g],exp[g])[0])
        rr = torch.Tensor(r)
        
        self.get_meta(name)
        
        os.makedirs(f"final/{self.__class__.__name__}_{self.num_n}/{self.data}/{self.patient}", exist_ok=True)
        np.save(f"final/{self.__class__.__name__}_{self.num_n}/{self.data}/{self.patient}/{name[0]}", pred.T)
        
        return {"MSE":mse, "MAE":mae, "corr":rr}
    
    def test_epoch_end(self, outputs):
        avg_mse = torch.stack(
            [x["MSE"] for x in outputs]).nanmean()

        avg_mae = torch.stack(
            [x["MAE"] for x in outputs]).nanmean()

        avg_corr = torch.stack(
            [x["corr"] for x in outputs]).nanmean(0)

        os.makedirs(f"final/{self.__class__.__name__}_{self.num_n}/{self.data}/{self.patient}", exist_ok=True)
        torch.save(avg_mse.cpu(), f"final/{self.__class__.__name__}_{self.num_n}/{self.data}/{self.patient}/MSE")
        torch.save(avg_mae.cpu(), f"final/{self.__class__.__name__}_{self.num_n}/{self.data}/{self.patient}/MAE")
        torch.save(avg_corr.cpu(), f"final/{self.__class__.__name__}_{self.num_n}/{self.data}/{self.patient}/cor")
    
    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optim=torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        StepLR = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.9)
        optim_dict = {'optimizer': optim, 'lr_scheduler': StepLR}
        return optim_dict
    
    def get_meta(self, name):
        if '10x_breast' in name[0]:
            self.patient = name[0]
            self.data = "test"
        else:
            name = name[0]
            self.data = name.split("+")[1]
            self.patient = name.split("+")[0]
            
            if self.data == 'her2st':
                self.patient = self.patient[0]
            elif self.data == 'stnet':
                self.data = "stnet"
                patient = self.patient.split('_')[0]
                if patient in ['BC23277', 'BC23287', 'BC23508']:
                    self.patient = 'BC1'
                elif patient in ['BC24105', 'BC24220', 'BC24223']:
                    self.patient = 'BC2'
                elif patient in ['BC23803', 'BC23377', 'BC23895']:
                    self.patient = 'BC3'
                elif patient in ['BC23272', 'BC23288', 'BC23903']:
                    self.patient = 'BC4'
                elif patient in ['BC23270', 'BC23268', 'BC23567']:
                    self.patient = 'BC5'
                elif patient in ['BC23269', 'BC23810', 'BC23901']:
                    self.patient = 'BC6'
                elif patient in ['BC23209', 'BC23450', 'BC23506']:
                    self.patient = 'BC7'
                elif patient in ['BC23944', 'BC24044']:
                    self.patient = 'BC8'
            elif self.data == 'skin':
                self.patient = self.patient.split('_')[0]
    
    def load_model(self):
        name = self.hparams.MODEL.name
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

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.MODEL.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams.MODEL, arg)
        args1.update(other_args)
        return Model(**args1)

    