
from glob import glob

import h5py
import scanpy as sc
import torch
import torchvision.transforms as transforms


class STDataset(torch.utils.data.Dataset):
    """Some Information about baselines"""
    def __init__(self, 
                img_dir: str, 
                st_dir: str):
        super(STDataset, self).__init__()
        
        self.img_dir = img_dir
        self.st_dir = st_dir
        
        self.train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([transforms.RandomRotation((90, 90))]),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        
        self.test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def load_img(self, name: str):
        """Load whole slide image of a sample.

        Args:
            name (str): name of a sample

        Returns:
            PIL.Image: return whole slide image.
        """
        
        # img_dir = self.img_dir+'/ST-imgs'
        # path = glob(img_dir+'/*' + name + self.img_ext)[0]
        path = glob(f"{self.img_dir}/{name}.h5")[0]
        
        with h5py.File(path, 'r') as f:
            img = f['img'][:]
            
        return img
    
    def load_st(self, name: str):
        """Load gene expression data of a sample.

        Args:
            name (str): name of a sample

        Returns:
            pandas.DataFrame: return gene expression. 
        """
        path = glob(f"{self.st_dir}/{name}.h5ad")[0]
        
        adata = sc.read_h5ad(path)
    
        return adata
