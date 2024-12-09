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
    def __init__(
        self,
        h5_path,
        wsi_dir="",
        ext='.tif',
        level=1,
        img_transform=None,
        num_n=1,
        radius=128,
        chunk_size=1000,
        use_openslide=False
    ):
        """
        Initializes the H5TileDataset.

        Args:
            h5_path (str): Path to the HDF5 file.
            wsi_dir (str, optional): Directory containing WSI files. Defaults to "".
            ext (str, optional): Extension of WSI files. Defaults to '.tif'.
            level (int, optional): Level of the WSI to read. Defaults to 1.
            img_transform (callable, optional): Transformation to apply to images. Defaults to None.
            num_n (int, optional): Number of neighbors. Must be odd. Defaults to 1.
            radius (int, optional): Radius for neighbor patches. Defaults to 128.
            chunk_size (int, optional): Number of samples per chunk. Defaults to 1000.
            use_openslide (bool, optional): Whether to use OpenSlide for WSI. Defaults to False.
        """
        self.h5_path = h5_path
        self.chunk_size = chunk_size
        self.use_openslide = use_openslide
        self.level = level
        self.img_transform = img_transform
        self.num_n = num_n
        assert num_n % 2 == 1, "num_n must be an odd number"
        self.radius = radius

        sample_id = os.path.basename(h5_path).split('.h5')[0]

        # Determine WSI path
        possible_paths = [
            f"{wsi_dir}/{sample_id}{ext}",
            *glob(f"{wsi_dir}/{sample_id}/*{ext}*")
        ]
        if not possible_paths:
            raise FileNotFoundError(f"WSI file for sample {sample_id} not found in {wsi_dir}")
        self.wsi_path = possible_paths[0]

        # Calculate number of chunks
        with h5py.File(h5_path, 'r') as f:
            self.n_chunks = int(np.ceil(len(f['coords']) / chunk_size))

        # Initialize placeholders for HDF5 and WSI
        self.h5_file = None
        self.wsi = None

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        # Lazy initialization: open HDF5 file and WSI only when accessed
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')

        if self.wsi is None:
            self.wsi = self.load_wsi()

        start_idx = idx * self.chunk_size
        end_idx = (idx + 1) * self.chunk_size

        f = self.h5_file

        # Retrieve barcodes
        if 'barcode' in f.keys():
            barcodes = f['barcode'][start_idx:end_idx].astype(str).tolist()
        else:
            barcodes = ["" for _ in range(min(self.chunk_size, len(f['coords']) - start_idx))]

        # Retrieve coordinates
        coords = f['coords'][start_idx:end_idx]

        if self.num_n == 1:
            if 'img' in f.keys():
                imgs = f['img'][start_idx:end_idx]
                imgs = torch.tensor(imgs).permute(0, 3, 1, 2)  # Assuming img shape is (N, H, W, C)
                if self.img_transform:
                    imgs = self.img_transform(imgs)
            else:
                imgs = [
                    self.extract_patch(coord)
                    for coord in coords
                ]
                imgs = torch.stack(imgs)
                if self.img_transform:
                    imgs = self.img_transform(imgs)
            return {
                'imgs': imgs,
                'barcodes': barcodes,
                'coords': coords
            }
        else:
            imgs, mask_tb = self.get_neighbor(coords)
            return {
                'imgs': imgs,
                'barcodes': barcodes,
                'coords': coords,
                'mask_tb': mask_tb
            }

    def load_wsi(self):
        if self.use_openslide:
            return OpenSlide(self.wsi_path)
        else:
            try:
                wsi = tifi.imread(self.wsi_path)
                return wsi
            except RuntimeError:
                wsi = cv2.imread(self.wsi_path)
                if wsi is None:
                    raise FileNotFoundError(f"Unable to read WSI at {self.wsi_path} using cv2.")
                return wsi

    def extract_patch(self, coord):
        """
        Extracts a single patch from the WSI.

        Args:
            coord (tuple): (y, x) coordinates.

        Returns:
            torch.Tensor: Transformed image patch.
        """
        y, x = coord
        if self.use_openslide:
            patch = self.wsi.read_region((x, y), self.level, (self.radius * 2, self.radius * 2)).convert('RGB')
            if self.img_transform:
                patch = self.img_transform(patch)
            else:
                patch = torch.tensor(np.array(patch)).permute(2, 0, 1)  # Convert to CxHxW
        else:
            # Assuming WSI loaded as NumPy array with shape (H, W, C)
            wsi_shape = self.wsi.shape
            x_start = max(x - self.radius, 0)
            y_start = max(y - self.radius, 0)
            x_end = min(x + self.radius, wsi_shape[1])
            y_end = min(y + self.radius, wsi_shape[0])
            patch = self.wsi[y_start:y_end, x_start:x_end, :]

            # Handle edge cases where patch size is smaller than expected
            if patch.shape[0] != self.radius * 2 or patch.shape[1] != self.radius * 2:
                pad_y = self.radius * 2 - patch.shape[0]
                pad_x = self.radius * 2 - patch.shape[1]
                patch = np.pad(
                    patch,
                    ((0, pad_y), (0, pad_x), (0, 0)),
                    mode='constant',
                    constant_values=0
                )

            patch = Image.fromarray(patch)
            if self.img_transform:
                patch = self.img_transform(patch)
            else:
                patch = torch.tensor(np.array(patch)).permute(2, 0, 1)  # Convert to CxHxW

        return patch

    def make_masking_table(self, x: int, y: int, img_shape: tuple) -> torch.Tensor:
        """
        Generate masking table for neighbor encoder.

        Args:
            x (int): x coordinate of target spot.
            y (int): y coordinate of target spot.
            img_shape (tuple): Shape of whole slide image (H, W, ...).

        Returns:
            torch.Tensor: Masking table.
        """
        mask_tb = torch.ones(self.num_n ** 2, dtype=torch.bool)

        pad_n = self.num_n // 2
        for offset in range(-pad_n, pad_n + 1):
            for od in range(-pad_n, pad_n + 1):
                nx, ny = x + od * self.radius * 2, y + offset * self.radius * 2
                if nx - self.radius < 0 or ny - self.radius < 0:
                    mask_tb[(offset + pad_n) * self.num_n + (od + pad_n)] = False
                if nx + self.radius > img_shape[1] or ny + self.radius > img_shape[0]:
                    mask_tb[(offset + pad_n) * self.num_n + (od + pad_n)] = False

        return mask_tb

    def get_neighbor(self, coords, patch_size=224) -> tuple:
        """
        Retrieves neighbor patches for each coordinate.

        Args:
            coords (np.ndarray): Array of (y, x) coordinates.
            patch_size (int, optional): Size of each patch. Defaults to 224.

        Returns:
            tuple: (neighbor_patches, mask_tb)
        """
        n_patches = len(coords)
        patch_dim = patch_size * self.num_n

        if self.use_openslide:
            img_shape = self.wsi.level_dimensions[self.level]
        else:
            img_shape = self.wsi.shape

        neighbor_patches = torch.zeros((n_patches, 3, patch_dim, patch_dim))
        mask_tb = torch.ones((n_patches, self.num_n ** 2), dtype=torch.bool)

        pad_n = self.num_n // 2

        for i, (y, x) in enumerate(coords):
            # Generate masking table
            mask = self.make_masking_table(x, y, img_shape)
            mask_tb[i] = mask

            # Calculate the top-left corner for the neighbor grid
            x_start = x - self.radius * pad_n
            y_start = y - self.radius * pad_n

            # Initialize an empty tensor for the neighbor grid
            neighbor_grid = torch.zeros((3, patch_dim, patch_dim))

            for n in range(self.num_n ** 2):
                row = n // self.num_n
                col = n % self.num_n

                if not mask[n]:
                    continue

                current_x = x_start + col * self.radius * 2
                current_y = y_start + row * self.radius * 2

                if self.use_openslide:
                    patch = self.wsi.read_region((current_x, current_y), self.level, (self.radius * 2, self.radius * 2)).convert('RGB')
                    if self.img_transform:
                        patch = self.img_transform(patch)
                    else:
                        patch = torch.tensor(np.array(patch)).permute(2, 0, 1)
                else:
                    # Assuming WSI loaded as NumPy array with shape (H, W, C)
                    wsi_shape = self.wsi.shape
                    x_end = current_x + self.radius * 2
                    y_end = current_y + self.radius * 2

                    # Handle edge cases
                    if current_x < 0 or current_y < 0 or x_end > wsi_shape[1] or y_end > wsi_shape[0]:
                        patch = torch.zeros((3, self.radius * 2, self.radius * 2))
                    else:
                        patch = self.wsi[current_y:y_end, current_x:x_end, :]
                        patch = Image.fromarray(patch)
                        if self.img_transform:
                            patch = self.img_transform(patch)
                        else:
                            patch = torch.tensor(np.array(patch)).permute(2, 0, 1)

                if self.img_transform:
                    neighbor_grid[:, row * patch_size:(row + 1) * patch_size, col * patch_size:(col + 1) * patch_size] = patch
                else:
                    neighbor_grid[:, row * patch_size:(row + 1) * patch_size, col * patch_size:(col + 1) * patch_size] = patch

            neighbor_patches[i] = neighbor_grid

        return neighbor_patches, mask_tb