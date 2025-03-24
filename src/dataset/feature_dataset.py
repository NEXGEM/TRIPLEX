import os 
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import h5py
import cv2
import tifffile as tifi
from openslide import OpenSlide
from hestcore.wsi import wsi_factory, OpenSlideWSI


class H5TileDataset(Dataset):
    def __init__(self, 
                h5_path, 
                wsi_dir=None, 
                ext='.tif', 
                level=1, 
                img_transform=None, 
                num_n=1, 
                radius=112, 
                chunk_size=1000, 
                num_workers=6):

        self.h5_path = h5_path
        self.chunk_size = chunk_size
        self.level = level
        self.wsi_loaded = 0
        self.num_workers = num_workers
        self.use_openslide = False if 'tif' in ext else False
        
        sample_id = os.path.basename(h5_path).split('.h5')[0]
        
        if wsi_dir is not None:
            if os.path.isfile(f"{wsi_dir}/{sample_id}{ext}"):
                wsi_path = f"{wsi_dir}/{sample_id}{ext}"
            else:
                wsi_path = glob(f"{wsi_dir}/{sample_id}/*{ext}*")[0]
                
            self.wsi = self._load_wsi(wsi_path)
        
        self.img_transform = img_transform
        self.transformed = False
        
        self.num_n = num_n
        assert num_n % 2 == 1, "num_n must be odd number"
        
        self.r = radius
        
        with h5py.File(h5_path, 'r') as f:
            self.n_chunks = int(np.ceil(len(f['coords']) / chunk_size))        
    
    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start_idx = idx * self.chunk_size
        end_idx = (idx + 1) * self.chunk_size
        with h5py.File(self.h5_path, 'r') as f:
            # barcodes = f['barcode'][start_idx:end_idx].flatten().tolist()
            if 'barcode' in f.keys():
                barcodes = f['barcode'][start_idx:end_idx].flatten().tolist()
            else:
                barcodes = torch.zeros(self.n_chunks)
                
            coords = f['coords'][start_idx:end_idx]
            
            if 'img' in f.keys():
                imgs = f['img'][start_idx:end_idx]
                
            else:
                if self.num_n == 1:
                    imgs = [self.wsi.read_region(coord, self.level, (self.r*2, self.r*2)) for coord in coords]
                    imgs = [np.array(img) for img in imgs]
                    imgs = np.stack(imgs)
                else:
                    imgs, mask_tb = self.get_neighbor(coords, self.wsi) 
                    self.transformed = True
        
        
        if self.num_n == 1:
            if not self.transformed:
                imgs = torch.stack([self.img_transform(Image.fromarray(img)) for img in imgs])
            return {'imgs': imgs, 'barcodes': barcodes, 'coords': coords}
        else:
            if not self.transformed:
                
                imgs_transformed = torch.zeros((imgs.shape[0], imgs.shape[3], imgs.shape[1], imgs.shape[2]))
                for i in range(imgs.shape[0]):
                    img = imgs[i]
                    for x in range(0, img.shape[0], self.r*2):
                        for y in range(0, img.shape[1], self.r*2):
                            imgs_transformed[i, :, x:x+self.r*2, y:y+self.r*2] = self.img_transform(Image.fromarray(img[x:x+self.r*2, y:y+self.r*2]))
                imgs = imgs_transformed
                
                mask_tb = self.get_mask_tables(coords, self.wsi)
                
            # imgs = torch.stack([self.img_transform(Image.fromarray(img)) for img in imgs])
            # imgs, mask_tb = self.get_neighbor(coords, self.wsi) 
            
            return {'imgs': imgs, 'barcodes': barcodes, 'coords': coords, 'mask_tb':mask_tb}
        
    def _load_wsi(self, wsi_path):
        if self.use_openslide:
            wsi = OpenSlideWSI(OpenSlide(wsi_path))
        else:
            wsi = wsi_factory(wsi_path)
        
        return wsi
    
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
        mask_tb = torch.ones(self.num_n**2)
        
        def create_mask(ind, mask_tb, window):
            if x-self.r*window < 0:
                mask_tb[self.num_n*ind:self.num_n*ind+self.num_n] = 0 
            if x+self.r*window > img_shape[0]:
                mask_tb[(self.num_n**2-self.num_n*(ind+1)):(self.num_n**2-self.num_n*ind)] = 0 
            if y-self.r*window < 0:
                mask = [i+ind for i in range(self.num_n**2) if i % self.num_n == 0]
                mask_tb[mask] = 0 
            if y+self.r*window > img_shape[1]:
                mask = [i-ind for i in range(self.num_n**2) if i % self.num_n == (self.num_n-1)]
                mask_tb[mask] = 0 
                
            return mask_tb
        
        ind = 0
        window = self.num_n
        while window >= 3: 
            mask_tb = create_mask(ind, mask_tb, window)
            ind += 1 
            window -= 2   

        return mask_tb
    
    def get_mask_tables(self, coords, wsi):
        n_patches = len(coords)
        
        mask_tb = torch.ones((n_patches, self.num_n**2))
        
        for i in range(n_patches):
            x, y = coords[i]

            wsi_shape = wsi.get_dimensions()
            mask = self.make_masking_table(x, y, wsi_shape)
            mask_tb[i] = mask

        return mask_tb
    
    def get_neighbor(self, coords, wsi):
        n_patches = len(coords)
        
        patch_size = self.r * 2 * self.num_n
        neighbor_patches = torch.zeros((n_patches, 3, patch_size, patch_size))
        mask_tb = torch.ones((n_patches, self.num_n**2))
        
        
        k_offsets = torch.arange(self.num_n) * self.r * 2
        m_offsets = torch.arange(self.num_n) * self.r * 2

        
        for i in range(n_patches):
            x, y = coords[i]

            wsi_shape = wsi.get_dimensions()
            mask = self.make_masking_table(x, y, wsi_shape)
            mask_tb_i = mask.clone()

            x_start = x - self.r * self.num_n
            y_start = y - self.r * self.num_n

            # Initialize empty patch
            neighbor_patch = torch.zeros((3, patch_size, patch_size))

            for k in range(self.num_n):
                for m in range(self.num_n):
                    n = k * self.num_n + m
                    if mask_tb_i[n] != 0:
                        current_x = x_start + k_offsets[k]
                        current_y = y_start + m_offsets[m]
                        
                        tmp = wsi.read_region((current_x, current_y), self.level, (self.r * 2, self.r * 2))
                        tmp = self.img_transform(Image.fromarray(tmp))
                        
                        neighbor_patch[:, k * self.r * 2 : (k + 1) * self.r * 2, m * self.r * 2 : (m + 1) * self.r * 2] = tmp
                        
            neighbor_patches[i] = neighbor_patch
            mask_tb[i] = mask

        return neighbor_patches, mask_tb