
import os
import sys
from glob import glob
from tqdm import tqdm

import argparse
import numpy as np
import pandas as pd
import h5py
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from openslide import OpenSlide
import multiprocessing as mp
import scanpy as sc

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_st, pxl_to_array, normalize_adata, save_hdf5


def preprocess_hest(input_path, output_dir):
    fname =  os.path.splitext(os.path.basename(input_path))[0]
    
    # if os.path.exists(name_st):
    #     print("ST data already exists. Skipping...")
    # else:
    #     print("Dumping matched ST data...")
    with h5py.File(f"{output_dir}/patches/{fname}.h5", "r") as f:
        barcode = f['barcode'][:].astype('str').squeeze()
    
    adata = sc.read_h5ad(input_path)
    
    # if 'log1p' in adata.uns:
    #     print("ST data already processed. Skipping...")
    #     return None
    
    barcode = pd.DataFrame(index=barcode)
    barcode_merged = pd.merge(adata.obs, barcode, left_index=True, right_index=True).index
    adata = adata[barcode_merged]
    
    print("Normalizing ST data...")
    # adata = normalize_adata(adata)
    adata.write(f"{output_dir}/adata/{fname}.h5ad")
        
    return fname

def preprocess_train_new(input_path, output_dir, platform='visium'):
    fname = os.path.basename(input_path)
    
    print("Loading ST data...")
    try:
        st = load_st(input_path, platform=platform)
    except:
        print("Failed to load ST data. Move to next sample.")
        return None
    
    if os.path.exists(f"{output_dir}/patches/{fname}.h5"):
        print("Patches already exists. Skipping...")
    else:
        # TODO: Add code to save result of segmentation & Resume
        print("Segmenting tissue...")
        st.segment_tissue(method='deep')
    
        print("Dumping patches...")
        st.dump_patches(
            f"{output_dir}/patches",
            name=fname,
            target_patch_size=224, # target patch size in 224
            target_pixel_size=0.5 # pixel size of the patches in um/px after rescaling
        )
    
    name_st = f"{output_dir}/adata/{fname}.h5ad"
    
    if os.path.exists(name_st):
        print("ST data already exists. Skipping...")
    else:
        print("Dumping matched ST data...")
        with h5py.File(f"{output_dir}/patches/{fname}.h5", "r") as f:
            barcode = f['barcode'][:].astype('str').squeeze()
            
        barcode = pd.DataFrame(index=barcode)
        barcode_merged = pd.merge(st.adata.obs, barcode, left_index=True, right_index=True).index
        adata = st.adata[barcode_merged]
        
        print("Normalizing ST data...")
        adata = normalize_adata(adata, cpm=True, smooth=True)
        adata.write(name_st)
        
    return fname
        
def get_pos(input_path, output_dir, step_size=160):
    fname = os.path.splitext(os.path.basename(input_path))[0]
    
    with h5py.File(input_path, 'r') as f:
        crds = f['coords'][:]
        
    array_crds = pxl_to_array(crds, step_size)
    
    
    df_crds = pd.DataFrame(array_crds) 
    check_dup = df_crds.apply(tuple, axis=1).duplicated().sum()
    
    while check_dup > 0:
        print(f"Adjusting step_size for {fname}...")
        print(f"Stepsize change from {step_size} to {step_size-10}")
        step_size -= 10
        array_crds = pxl_to_array(crds, step_size)
        df_crds = pd.DataFrame(array_crds) 
        check_dup = df_crds.apply(tuple, axis=1).duplicated().sum()
    
    np.save(f"{output_dir}/{fname}", array_crds)
    
# def preprocess_image(input_path, processed_path, slide_level=0, patch_size=256):
#     wsi = OpenSlide(input_path)
    
#     with h5py.File(processed_path, 'r') as f:
#         coords = f['coords'][:]
        
#     # imgs = [wsi.read_region(coord, slide_level, (patch_size, patch_size)).convert('RGB') for coord in coords]
#     # imgs = [np.array(img) for img in imgs]
#     # imgs = np.stack(imgs)
#     imgs = np.empty((len(coords), patch_size, patch_size, 3), dtype=np.uint8)
#     for i, coord in enumerate(coords):
#         img = wsi.read_region(coord, slide_level, (patch_size, patch_size)).convert('RGB')
#         imgs[i] = np.array(img)

#     asset_dict = {'img': imgs}
#     save_hdf5(processed_path, asset_dict=asset_dict, mode='a')

# Global variable to hold the OpenSlide object in each process
# _GLOBAL_WSI = None

# def init_worker(input_path):
#     # This function will run once in each worker process when the pool is initialized.
#     global _GLOBAL_WSI
#     _GLOBAL_WSI = OpenSlide(input_path)

# def process_patch(coord, slide_level, patch_size):
#     # Uses the global _GLOBAL_WSI opened once per process
#     img = _GLOBAL_WSI.read_region(coord, slide_level, (patch_size, patch_size)).convert('RGB')
#     # Convert directly to a NumPy array
#     return np.array(img)

# def process_patch(coord, input_path, slide_level, patch_size):
#     # Instantiate OpenSlide within each process to ensure safety
#     wsi = OpenSlide(input_path)
#     img = wsi.read_region(coord, slide_level, (patch_size, patch_size)).convert('RGB')
#     wsi.close()
#     return np.array(img, dtype=np.uint8)
    
def save_image(input_path, processed_path, slide_level=0, patch_size=256):
    # Open and read all coordinates
    with h5py.File(processed_path, 'r') as f:
        coords = f['coords'][:]
    
    # num_workers = mp.cpu_count()
    # if num_workers > 4:
    #     num_workers = 4
    wsi = OpenSlide(input_path)
    imgs = [np.array(wsi.read_region(coord, slide_level, (patch_size, patch_size)).convert('RGB')) for coord in coords]
    # with ProcessPoolExecutor(max_workers=num_workers,
    #                         initializer=init_worker,
    #                         initargs=(input_path,)) as executor:
    #     # Increase chunksize for fewer task dispatches
    #     imgs = list(executor.map(
    #         partial(process_patch, slide_level=slide_level, patch_size=patch_size),
    #         coords,
    #         chunksize=64  # Try a larger chunksize
    #     ))

    # Stack images into a single NumPy array
    imgs = np.stack(imgs)
    
    # Prepare the dictionary for HDF5
    asset_dict = {'img': imgs}
    
    # Save to HDF5
    save_hdf5(processed_path, asset_dict=asset_dict, mode='a')
    
# def preprocess_image(input_path, processed_path, slide_level=0, patch_size=256):
#     wsi = OpenSlide(input_path)

#     with h5py.File(processed_path, 'r') as f:
#         coords = f['coords'][:]

#     def process_patch(coord):
#         img = wsi.read_region(coord, slide_level, (patch_size, patch_size)).convert('RGB')
#         return np.array(img, dtype=np.uint8)

#     # Use parallel processing
#     with ThreadPoolExecutor(max_workers=8) as executor:
#         imgs = list(executor.map(process_patch, coords))

#     imgs = np.stack(imgs)
#     asset_dict = {'img': imgs}
#     save_hdf5(processed_path, asset_dict=asset_dict, mode='a')

    
        
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input_dir", type=str, default=None)
    argparser.add_argument("--output_dir", type=str, required=True)
    argparser.add_argument("--platform", type=str, default='visium')
    argparser.add_argument("--prefix", type=str, default='')
    argparser.add_argument("--mode", type=str, default='cv')
    argparser.add_argument("--step_size", type=int, default=160)
    
    argparser.add_argument("--slide_level", type=int, default=0)
    argparser.add_argument("--slide_ext", type=str, default='.svs')
    argparser.add_argument("--patch_size", type=int, default=256)
    
    args = argparser.parse_args()
    
    mode = args.mode
    input_dir = args.input_dir
    output_dir = args.output_dir
    platform = args.platform
    prefix = args.prefix
    step_size = args.step_size
    
    assert mode in ['train', 'train_hest', 'inference'], "mode must be either 'train' or 'train_hest' or 'inference'"
    
    if mode == 'train_new':
        os.makedirs(f"{output_dir}/patches", exist_ok=True)
        os.makedirs(f"{output_dir}/adata", exist_ok=True)
        
        ids = glob(f"{input_dir}/{prefix}*")
        
        sample_ids = []
        for input_path in tqdm(ids):
            sample_id = preprocess_train_new(input_path, output_dir, platform=platform)
            if sample_id is not None:
                sample_ids.append(sample_id)
        
        pd.DataFrame(sample_ids, columns=['sample_id']).to_csv(f"{output_dir}/ids.csv", index=False)
        
    elif mode == 'train_hest':
        patch_dir = f"{output_dir}/patches"
        
        ids = glob(f"{patch_dir}/*.h5")
    
        for input_path in tqdm(ids):
            name = os.path.splitext(os.path.basename(input_path))[0]
            raw_path = f"{input_dir}/{name}.h5ad"
            sample_id = preprocess_hest(raw_path, output_dir)
            
    elif mode == 'inference':
        slide_level = args.slide_level
        patch_size = args.patch_size
        slide_ext = args.slide_ext
        
        crd_dir = f"{output_dir}/patches"
        output_dir = f"{output_dir}/pos"
        os.makedirs(output_dir, exist_ok=True)
        
        ids = glob(f"{crd_dir}/*.h5")
        for input_path in tqdm(ids):
            # with h5py.File(input_path, 'r') as f:
            #     if 'img' in f.keys():
            #         print("Image already exists. Skipping...")
            #         continue
            # get_pos(input_path, output_dir, step_size)
            
            name = os.path.splitext(os.path.basename(input_path))[0]
            raw_path = f"{input_dir}/{name}{slide_ext}"
            save_image(raw_path, input_path, slide_level, patch_size)