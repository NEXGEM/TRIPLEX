
import os
from glob import glob
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd 
from sklearn.model_selection import KFold
from PIL import ImageFile, Image
import torch
import torchvision
import torchvision.transforms as transforms
import scprep as scp

from utils import smooth_exp

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class BaselineDataset(torch.utils.data.Dataset):
    """Some Information about baselines"""
    def __init__(self):
        super(BaselineDataset, self).__init__()
        
        self.external_test = False

        self.train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomApply([torchvision.transforms.RandomRotation((90, 90))]),
            transforms.ToTensor(),
        ])
        
        self.test_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

    def get_img(self, name: str):
        """Load whole slide image of a sample.

        Args:
            name (str): name of a sample

        Returns:
            PIL.Image: return whole slide image.
        """
        img_dir = self.data_dir+'/ST-imgs'
        if self.data == 'her2st':
            pre = img_dir+'/'+name[0]+'/'+name
            fig_name = os.listdir(pre)[0]
            path = pre+'/'+fig_name
        elif self.data == 'stnet' or '10x_breast' in self.data:
            path = glob(img_dir+'/*'+name+'.tif')[0]
        else:
            path = glob(img_dir+'/*'+name+'.jpg')[0]
        
        im = Image.open(path)
        return im
    
    def get_cnt(self, name: str):
        """Load gene expression data of a sample.

        Args:
            name (str): name of a sample

        Returns:
            pandas.DataFrame: return gene expression. 
        """
        path = self.data_dir+'/ST-cnts/'+name+'_sub.parquet'
        df = pd.read_parquet(path)

        return df

    def get_pos(self, name: str):
        """Load position information of a sample.
        The 'id' column is for matching against the gene expression table.

        Args:
            name (str): name of a sample

        Returns:
            pandas.DataFrame: return DataFrame with position information.
        """
        path = self.data_dir+'/ST-spotfiles/'+name+'_selection.tsv'
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id

        return df

    def get_meta(self, name: str):
        """Load both gene expression and postion data and merge them.

        Args:
            name (str): name of a sample

        Returns:
            pandas.DataFrame: return merged table (gene exp + position)
        """
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join(pos.set_index('id'),how='inner')
        
        if self.external_test:
            meta = meta.sort_values(['x', 'y'])
        else:
            meta = meta.sort_values(['y', 'x'])

        return meta


class STDataset(BaselineDataset):
    """Dataset to load ST data for TRIPLEX
    """
    def __init__(self, train: bool, fold: int=0, external_test: bool=False, extract: str=None, test_data=None, **kwargs):
        """

        Args:
            train (bool): In training mode or not.
            fold (int): Number of fold for cross validation.
            external_test (bool, optional): In external test model or not. Defaults to False.
            extract (bool, optional): Source to extract features. Defaults to None.
            test_data (str, optional): Test data name. Defaults to None.
        """
        super().__init__()
        
        self.train = train
        self.extract = extract
        self.external_test = external_test
            
        self.gt_dir = kwargs['t_global_dir']
        self.num_neighbors = kwargs['num_neighbors']
        self.neighbor_dir = f"{kwargs['neighbor_dir']}_{self.num_neighbors}_224"
        
        self.r = kwargs['radius']//2
        
        if external_test:
            self.data = test_data
            self.data_dir = f"{kwargs['data_dir']}/test/{self.data}"
        else:
            self.data = kwargs['type']
            self.data_dir = f"{kwargs['data_dir']}/{self.data}"
        
        names = os.listdir(self.data_dir+'/ST-spotfiles')
        names.sort()
        names = [i.split('_selection.tsv')[0] for i in names]
        
        if not external_test:
            if self.data == 'stnet':
                kf = KFold(8, shuffle=True, random_state=2021)
                patients = ['BC23209','BC23270','BC23803','BC24105','BC24220','BC23268','BC23269','BC23272','BC23277','BC23287','BC23288','BC23377','BC23450','BC23506','BC23508','BC23567','BC23810','BC23895','BC23901','BC23903','BC23944','BC24044','BC24223']
                patients = np.array(patients)
                _, ind_val = [i for i in kf.split(patients)][fold]
                paients_val = patients[ind_val]
                
                te_names = []
                for pp in paients_val:
                    te_names += [i for i in names if pp in i]
                
            elif self.data == 'her2st':
                patients = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
                te_names = [i for i in names if patients[fold] in i]
            elif self.data == 'skin':
                patients = ['P2', 'P5', 'P9', 'P10']
                te_names = [i for i in names if patients[fold] in i]
                
            tr_names = list(set(names)-set(te_names))

            if train:
                self.names = tr_names
            else:
                self.names = te_names
        else:
            self.names = names

        self.img_dict = {i:np.array(self.get_img(i)) for i in self.names}
        self.meta_dict = {i:self.get_meta(i) for i in self.names}
        
        gene_list = list(np.load(self.data_dir + f'/genes_{self.data}.npy', allow_pickle=True))    
        
        self.exp_dict = {i:scp.transform.log(scp.normalize.library_size_normalize(m[gene_list])) for i,m in self.meta_dict.items()}
        
        # Smoothing data 
        self.exp_dict = {i:smooth_exp(m).values for i,m in self.exp_dict.items()}
        
        if external_test:
            self.center_dict = {i:np.floor(m[['pixel_y','pixel_x']].values).astype(int) for i,m in self.meta_dict.items()}
            self.loc_dict = {i:m[['x','y']].values for i,m in self.meta_dict.items()}
        else:
            self.center_dict = {i:np.floor(m[['pixel_x','pixel_y']].values).astype(int) for i,m in self.meta_dict.items()}
            self.loc_dict = {i:m[['y','x']].values for i,m in self.meta_dict.items()}

        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(self.names))

    def __getitem__(self, index):
        """Return one piece of data for training, and all data within a patient for testing.

        Returns:
            tuple: 
                patches (torch.Tensor): Target spot images
                exps (torch.Tensor): Gene expression of the target spot.
                pid (torch.LongTensor): patient index
                sid (torch.LongTensor): spot index
                wsi (torch.Tensor): Features extracted from all spots for the patient
                position (torch.LongTensor): Relative position of spots 
                neighbors (torch.Tensor): Features extracted from neighbor regions of the target spot.
                maks_tb (torch.Tensor): Masking table for neighbor features
        """
        if self.train or (self.external_test and self.extract in ['g_target','neighbor']):
        # if self.train:
            i = 0
            while index>=self.cumlen[i]:
                i += 1
            idx = index
            if i > 0:
                idx = index - self.cumlen[i-1]
                
            name = self.id2name[i]
                
            im = self.img_dict[name]
            center = self.center_dict[name][idx]
            x, y = center
            
            mask_tb =  self.make_masking_table(x, y, im.shape)
                
            if (self.external_test and self.extract == 'neighbor'):
                y_start = y-self.r*self.num_neighbors
                x_start = x-self.r*self.num_neighbors
                
                n = 0
                patches = torch.zeros((3,2*self.r*self.num_neighbors,2*self.r*self.num_neighbors))
                for k in range(self.num_neighbors):
                    for m in range(self.num_neighbors):
                        if mask_tb[n] != 0:
                            tmp = im[y_start+self.r*2*k:y_start+self.r*2*(k+1), x_start+self.r*2*m:x_start+self.r*2*(m+1),:]    
                            tmp = self.test_transforms(tmp)
                            patches[:,self.r*2*k:self.r*2*(k+1), self.r*2*m:self.r*2*(m+1)] = tmp
                        n+=1
                return patches
            else:
                patches = im[y-self.r:y+self.r, x-self.r:x+self.r, :] 
                if (self.external_test and self.extract == 'g_target'):
                    return self.test_transforms(patches)
                
                if self.external_test:
                    patches = self.test_transforms(patches)
                else:
                    patches = self.train_transforms(patches)
                
                exps = self.exp_dict[name][idx]
                exps = torch.Tensor(exps)
                
                sid = torch.LongTensor([idx])
                
                neighbors = torch.load(self.data_dir + f"/{self.neighbor_dir}/{name}.pt")[idx]
        else:
            i = index
            name = self.id2name[i]
            
            im = self.img_dict[name]    
            centers = self.center_dict[name]
            
            n_patches = len(centers)    
            if self.extract == 'neighbor':
                patches = torch.zeros((n_patches,3,2*self.r*self.num_neighbors,2*self.r*self.num_neighbors))
            else:
                patches = torch.zeros((n_patches,3,2*self.r,2*self.r))
            mask_tb = torch.ones((n_patches, self.num_neighbors**2))
            for j in range(n_patches):
                center = centers[j]
                x, y = center
                
                mask_tb[j] = self.make_masking_table(x, y, im.shape)
                
                if self.extract == 'neighbor':
                    y_start = y-self.r*self.num_neighbors
                    x_start = x-self.r*self.num_neighbors
                    
                    n = 0
                    patch = torch.zeros((3,2*self.r*self.num_neighbors,2*self.r*self.num_neighbors))
                    for k in range(self.num_neighbors):
                        for m in range(self.num_neighbors):
                            if mask_tb[j,n] != 0:
                                tmp = im[y_start+self.r*2*k:y_start+self.r*2*(k+1), x_start+self.r*2*m:x_start+self.r*2*(m+1),:]    
                                tmp = self.test_transforms(tmp)
                                patch[:,self.r*2*k:self.r*2*(k+1), self.r*2*m:self.r*2*(m+1)] = tmp
                            n+=1
                else:
                    patch = im[y-self.r:y+self.r,x-self.r:x+self.r,:]
                    patch = self.test_transforms(patch)
                
                patches[j] = patch

            exps = self.exp_dict[name]
            exps = torch.Tensor(exps)
            
            sid = torch.arange(n_patches)
            
            if self.extract in ['g_neighbor', 'g_target', 'neighbor']:
                return patches, exps
            
            neighbors = torch.load(self.data_dir +  f"/{self.neighbor_dir}/{name}.pt")
        
        wsi = torch.load(self.data_dir +  f"/{self.gt_dir}/{name}.pt")
        
        pid = torch.LongTensor([i])
        pos = self.loc_dict[name]
        position = torch.LongTensor(pos)
        
        if not self.external_test:
            name += f"+{self.data}"
        
        if self.train:
            return patches, exps, pid, sid, wsi, position, neighbors, mask_tb
        else:
            return patches, exps, sid, wsi, position, name, neighbors, mask_tb
        
    def __len__(self):
        if self.train or (self.external_test and self.extract in ['g_target','neighbor']):
            return self.cumlen[-1]
        else:
            return len(self.exp_dict)
        
    def make_masking_table(self, x: int, y: int, img_shape: tuple):
        """Generate masking table for neighbor encoder.

        Args:
            x (int): x coordinate of target spot
            y (int): y coordinate of target spot
            img_shape (tuple): Shape of whole slide image

        Raises:
            Exception: if self.num_neighbors is bigger than 5, raise error.

        Returns:
            torch.Tensor: masking table
        """
        
        # Make masking table for neighbor encoding module
        mask_tb = torch.ones(self.num_neighbors**2)
        
        def create_mask(ind, mask_tb, window):
            if y-self.r*window < 0:
                mask_tb[self.num_neighbors*ind:self.num_neighbors*ind+self.num_neighbors] = 0 
            if y+self.r*window > img_shape[0]:
                mask_tb[(self.num_neighbors**2-self.num_neighbors*(ind+1)):(self.num_neighbors**2-self.num_neighbors*ind)] = 0 
            if x-self.r*window < 0:
                mask = [i+ind for i in range(self.num_neighbors**2) if i % self.num_neighbors == 0]
                mask_tb[mask] = 0 
            if x+self.r*window > img_shape[1]:
                mask = [i-ind for i in range(self.num_neighbors**2) if i % self.num_neighbors == (self.num_neighbors-1)]
                mask_tb[mask] = 0 
                
            return mask_tb
        
        ind = 0
        window = self.num_neighbors
        while window >= 3: 
            mask_tb = create_mask(ind, mask_tb, window)
            ind += 1 
            window -= 2   
         
        return mask_tb
