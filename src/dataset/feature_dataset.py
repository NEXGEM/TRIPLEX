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


class H5TileDataset(Dataset):
    def __init__(self, 
                h5_path, 
                wsi_dir=None, 
                ext='.tif', 
                level=1, 
                img_transform=None, 
                num_n=1, 
                radius=128, 
                chunk_size=1000, 
                num_workers=6,
                use_openslide=False):

        self.h5_path = h5_path
        self.chunk_size = chunk_size
        self.use_openslide = use_openslide
        self.level = level
        self.wsi_loaded = 0
        self.num_workers = num_workers
        
        sample_id = os.path.basename(h5_path).split('.h5')[0]
        
        if wsi_dir is not None:
            if os.path.isfile(f"{wsi_dir}/{sample_id}{ext}"):
                wsi_path = f"{wsi_dir}/{sample_id}{ext}"
            else:
                wsi_path = glob(f"{wsi_dir}/{sample_id}/*{ext}*")[0]
                
            self.wsi = self.load_wsi(wsi_path)
        
        self.img_transform = img_transform
        
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
            
            if self.num_n == 1:
                if 'img' in f.keys():
                    imgs = f['img'][start_idx:end_idx]
                else:
                    if self.use_openslide:
                        imgs = [self.wsi.read_region(coord, self.level, (self.r*2, self.r*2)).convert('RGB') for coord in coords]
                        imgs = [np.array(img) for img in imgs]
                        imgs = np.stack(imgs)
                    else:
                        raise NotImplementedError("Not implemented yet")
                    
                    # wsi = self.load_wsi()
                    # self.wsi_loaded = 1
                    # tmp = wsi.read_region((x_start + k_start, y_start + m_start), 1, (self.r, self.r))
                    
                    # imgs = f['img'][start_idx:end_idx]
        
        if self.num_n == 1:
            if self.img_transform:
                imgs = torch.stack([self.img_transform(Image.fromarray(img)) for img in imgs])
            return {'imgs': imgs, 'barcodes': barcodes, 'coords': coords}
        else:
            imgs, mask_tb = self.get_neighbor(coords, self.wsi) 
            
            return {'imgs': imgs, 'barcodes': barcodes, 'coords': coords, 'mask_tb':mask_tb}
        
    def load_wsi(self, wsi_path):
        if self.use_openslide:
            wsi = OpenSlide(wsi_path)
        else:
            try:
                wsi = tifi.imread(wsi_path)
            except RuntimeError:
                wsi = cv2.imread(wsi_path)
        
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
    
    def get_neighbor(self, coords, wsi):
        n_patches = len(coords)
        
        patch_size = 224 * self.num_n
        neighbor_patches = torch.zeros((n_patches, 3, patch_size, patch_size))
        mask_tb = torch.ones((n_patches, self.num_n**2))

        # Precompute ranges
        k_offsets = torch.arange(self.num_n) * self.r * 2
        m_offsets = torch.arange(self.num_n) * self.r * 2

        # If using OpenSlide, ensure it's loaded once
        if self.use_openslide:
            wsi_level = self.level
            
        for i in range(n_patches):
            
            x, y = coords[i]
            # y, x = coords[i]
            wsi_shape = wsi.shape if hasattr(wsi, 'shape') else wsi.dimensions
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
                        # current_x = y_start + k_offsets[k]
                        # current_y = x_start + m_offsets[m]
                        if self.use_openslide:
                            # Read region using OpenSlide
                            tmp = wsi.read_region((current_x, current_y), wsi_level, (self.r * 2, self.r * 2)).convert('RGB')
                            tmp = self.img_transform(tmp)
                        else:
                            # Efficient slicing with NumPy
                            # tmp = wsi[current_y:current_y + self.r * 2, current_x:current_x + self.r * 2, :]
                            tmp = wsi[current_x:current_x + self.r * 2, current_y:current_y + self.r * 2, :]
                            tmp = self.img_transform(Image.fromarray(tmp))
                        # Assign transformed patch
                        neighbor_patch[:, k * 224:(k + 1) * 224, m * 224:(m + 1) * 224] = tmp
                        
            neighbor_patches[i] = neighbor_patch
            mask_tb[i] = mask

        return neighbor_patches, mask_tb