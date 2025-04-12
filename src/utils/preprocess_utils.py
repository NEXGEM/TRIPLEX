import os

import numpy as np
import h5py

from torchvision import transforms
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


from hest import ( STReader, 
                VisiumReader, 
                VisiumHDReader, 
                XeniumReader )
from hest.HESTData import read_HESTData



def load_st(path, platform):
    assert platform in ['st', 'visium', 'visium-hd', 'xenium'], "platform must be one of ['st', 'visium', 'visium-hd', 'xenium']"
    
    if platform == 'st':    
        st = STReader().auto_read(path)
        
    if platform == 'visium':
        st = VisiumReader().auto_read(path)
        
    if platform == 'visium-hd':
        st = VisiumHDReader().auto_read(path)
        
    if platform == 'xenium':
        # st = XeniumReader().auto_read(path)
        st = read_HESTData(
            adata_path = os.path.join(path, 'aligned_adata.h5ad'),
            img = os.path.join(path, 'aligned_fullres_HE.tif'),
            metrics_path = os.path.join(path, 'metrics.json'),
            # cellvit_path,
            # tissue_contours_path,
            xenium_cell_path = os.path.join(path, 'he_cell_seg.parquet'),
            xenium_nucleus_path = os.path.join(path, 'he_nucleus_seg.parquet'),
            transcripts_path = os.path.join(path, 'aligned_transcripts.parquet')
        )
    return st

def map_values(arr, step_size=256):
    """
    Map NumPy array values to integers such that:
    1. The minimum value is mapped to 0
    2. Values within 256 of each other are mapped to the same integer
    
    Args:
    arr (np.ndarray): Input NumPy array of numeric values
    
    Returns:
    tuple: 
        - NumPy array of mapped integer values 
    """
    if arr.size == 0:
        return np.array([]), {}
    
    # Sort the unique values
    unique_values = np.sort(np.unique(arr))
    
    mapping = {}
    current_key = 0
    
    mapping[unique_values[0]] = 0
    current_value = unique_values[0]

    for i in range(1, len(unique_values)):
        if unique_values[i] - current_value > step_size:
            current_key += 1
            current_value = unique_values[i] 
        
        mapping[unique_values[i]] = current_key
    
    mapped_arr = np.vectorize(mapping.get)(arr)
    
    return mapped_arr

def pxl_to_array(pixel_crds, step_size):
    x_crds = map_values(pixel_crds[:,0], step_size)
    y_crds = map_values(pixel_crds[:,1], step_size)
    dst = np.stack((x_crds, y_crds), axis=1)
    return dst


def save_hdf5(output_fpath, 
                      asset_dict, 
                      attr_dict= None, 
                      mode='a', 
                      auto_chunk = True,
                      chunk_size = None):
    with h5py.File(output_fpath, mode) as f:
        for key, val in asset_dict.items():
            data_shape = val.shape
            if len(data_shape) == 1:
                val = np.expand_dims(val, axis=1)
                data_shape = val.shape

            # Determine if the data is of string type
            if np.issubdtype(val.dtype, np.string_) or np.issubdtype(val.dtype, np.unicode_):
                data_type = h5py.string_dtype(encoding='utf-8')
            else:
                data_type = val.dtype

            if key not in f:  # if key does not exist, create dataset
                if auto_chunk:
                    chunks = True  # let h5py decide chunk size
                else:
                    chunks = (chunk_size,) + data_shape[1:]
                dset = f.create_dataset(
                    key,
                    shape=data_shape,
                    chunks=chunks,
                    maxshape=(None,) + data_shape[1:],
                    dtype=data_type
                )
                # Save attribute dictionary
                if attr_dict is not None:
                    if key in attr_dict.keys():
                        for attr_key, attr_val in attr_dict[key].items():
                            dset.attrs[attr_key] = attr_val
                dset[:] = val
            else:
                dset = f[key]
                dset.resize(len(dset) + data_shape[0], axis=0)
                if dset.dtype != data_type:
                    raise TypeError(f"Data type mismatch for key '{key}'. Dataset dtype: {dset.dtype}, value dtype: {data_type}")
                dset[-data_shape[0]:] = val


def get_transforms(mean, std, target_img_size = -1, center_crop = False, transform_type = 'eval'):
    trsforms = []
    
    # Apply specific transformation based on transform_type
    if transform_type == 'hori':
        # Horizontal flip
        trsforms.append(transforms.RandomHorizontalFlip(p=1.0))
    elif transform_type == 'vert':
        # Vertical flip
        trsforms.append(transforms.RandomVerticalFlip(p=1.0))
    elif transform_type == 'rot_90':
        # 90-degree rotation
        trsforms.append(transforms.RandomRotation((90, 90)))
    elif transform_type == 'rot_180':
        # 180-degree rotation
        trsforms.append(transforms.RandomRotation((180, 180)))
    elif transform_type == 'rot_270':
        # 270-degree rotation
        trsforms.append(transforms.RandomRotation((270, 270)))
    elif transform_type == 'tp':
        # Transpose (90-degree rotation + horizontal flip)
        class Transpose(object):
            def __call__(self, img):
                return transforms.functional.hflip(transforms.functional.rotate(img, 90))
        trsforms.append(Transpose())
    elif transform_type == 'tv':
        # Transverse (90-degree rotation + vertical flip)
        class Transverse(object):
            def __call__(self, img):
                return transforms.functional.vflip(transforms.functional.rotate(img, 90))
        trsforms.append(Transverse())
        
    elif transform_type == 'eval':
        # Default 'eval' mode has no augmentations
        pass
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")
    
    if target_img_size > 0:
        trsforms.append(transforms.Resize(target_img_size))
    if center_crop:
        assert target_img_size > 0, "target_img_size must be set if center_crop is True"
        trsforms.append(transforms.CenterCrop(target_img_size))
        
    trsforms.append(transforms.ToTensor())
    if mean is not None and std is not None:
        trsforms.append(transforms.Normalize(mean, std))
    trsforms = transforms.Compose(trsforms)

    return trsforms



def add_augmentation_to_transform(existing_transform, transform_type='eval'):
    """
    Adds the specified augmentation to an existing transform pipeline.
    
    Args:
        existing_transform (transforms.Compose): Existing transformation pipeline
        transform_type (str): Type of augmentation to add ('hori', 'vert', 'rot_90', 
                            'rot_180', 'rot_270', 'tp', 'tv', or 'eval')
    
    Returns:
        transforms.Compose: New transformation pipeline with added augmentation
    """
    from torchvision import transforms
    
    # Create augmentation based on transform_type
    augmentation = None
    if transform_type == 'hori':
        # Horizontal flip
        augmentation = transforms.RandomHorizontalFlip(p=1.0)
    elif transform_type == 'vert':
        # Vertical flip
        augmentation = transforms.RandomVerticalFlip(p=1.0)
    elif transform_type == 'rot_90':
        # 90-degree rotation
        augmentation = transforms.RandomRotation((90, 90))
    elif transform_type == 'rot_180':
        # 180-degree rotation
        augmentation = transforms.RandomRotation((180, 180))
    elif transform_type == 'rot_270':
        # 270-degree rotation
        augmentation = transforms.RandomRotation((270, 270))
    elif transform_type == 'tp':
        # Transpose (90-degree rotation + horizontal flip)
        class Transpose(object):
            def __call__(self, img):
                return transforms.functional.hflip(transforms.functional.rotate(img, 90))
        augmentation = Transpose()
    elif transform_type == 'tv':
        # Transverse (90-degree rotation + vertical flip)
        class Transverse(object):
            def __call__(self, img):
                return transforms.functional.vflip(transforms.functional.rotate(img, 90))
        augmentation = Transverse()
    elif transform_type == 'eval':
        # No augmentation in eval mode
        return existing_transform
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")
    
    # Extract transforms from the existing pipeline
    transform_list = list(existing_transform.transforms)
    
    # Insert the augmentation at the beginning of the pipeline
    transform_list.insert(0, augmentation)
    
    # Return new transform pipeline
    return transforms.Compose(transform_list)