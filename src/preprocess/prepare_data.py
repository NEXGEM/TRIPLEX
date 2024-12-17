
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


def preprocess_st(name, adata, output_dir, normalize=False):
    print("Dumping matched ST data...")
    with h5py.File(f"{output_dir}/patches/{name}.h5", "r") as f:
        barcode = f['barcode'][:].astype('str').squeeze()
        
    barcode = pd.DataFrame(index=barcode)
    barcode_merged = pd.merge(adata.obs, barcode, left_index=True, right_index=True).index
    adata = adata[barcode_merged]
    
    if normalize:
        print("Normalizing ST data...")
        adata = normalize_adata(adata, cpm=True, smooth=True)
    adata.write(f"{output_dir}/adata/{name}.h5ad")
    
    return adata

def save_patches(name, input_dir, output_dir, platform='visium'):

    print("Loading ST data...")
    try:
        st = load_st(f"{input_dir}/{name}", platform=platform)
    except:
        print("Failed to load ST data. Move to next sample.")
        return None
    
    if os.path.exists(f"{output_dir}/patches/{name}.h5"):
        print("Patches already exists. Skipping...")
    else:
        # TODO: Add code to save result of segmentation & Resume
        print("Segmenting tissue...")
        st.segment_tissue(method='deep')
    
        print("Dumping patches...")
        st.dump_patches(
            f"{output_dir}/patches",
            name=name,
            target_patch_size=224, # target patch size in 224
            target_pixel_size=0.5 # pixel size of the patches in um/px after rescaling
        )
    
    return st
        
def get_pos(name, input_dir, output_dir, step_size=160):
    
    with h5py.File(f"{input_dir}/{name}.h5", 'r') as f:
        crds = f['coords'][:]
        
    array_crds = pxl_to_array(crds, step_size)
    
    df_crds = pd.DataFrame(array_crds) 
    check_dup = df_crds.apply(tuple, axis=1).duplicated().sum()
    
    while check_dup > 0:
        print(f"Adjusting step_size for {name}...")
        print(f"Stepsize change from {step_size} to {step_size-10}")
        step_size -= 10
        array_crds = pxl_to_array(crds, step_size)
        df_crds = pd.DataFrame(array_crds) 
        check_dup = df_crds.apply(tuple, axis=1).duplicated().sum()
    
    np.save(f"{output_dir}/{name}", array_crds)
        
def save_image(slide_path, patch_path, slide_level=0, patch_size=256):
    # Open and read all coordinates
    with h5py.File(patch_path, 'r') as f:
        coords = f['coords'][:]
    
    wsi = OpenSlide(slide_path)
    imgs = [np.array(wsi.read_region(coord, slide_level, (patch_size, patch_size)).convert('RGB')) \
        for coord in coords]

    imgs = np.stack(imgs)
    
    asset_dict = {'img': imgs}
    save_hdf5(patch_path, asset_dict=asset_dict, mode='a')


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input_dir", type=str, default=None)
    argparser.add_argument("--output_dir", type=str, required=True)
    argparser.add_argument("--platform", type=str, default='visium')
    argparser.add_argument("--prefix", type=str, default='')
    argparser.add_argument("--mode", type=str, default='train')
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
    
    assert mode in ['train', 'hest', 'inference'], "mode must be either 'train' or 'hest' or 'inference'"
    
    if mode == 'train':
        os.makedirs(f"{output_dir}/patches", exist_ok=True)
        os.makedirs(f"{output_dir}/adata", exist_ok=True)
        
        ids = glob(f"{input_dir}/{prefix}*")
        
        sample_ids = []
        for input_path in tqdm(ids):
            name = os.path.basename(input_path)
            
            st = save_patches(name, input_dir, output_dir, platform=platform)
            if st is not None:
                sample_ids.append(name)
                preprocess_st(name, st.adata, output_dir, normalize=True)
            
        pd.DataFrame(sample_ids, columns=['sample_id']).to_csv(f"{output_dir}/ids.csv", index=False)
        
    elif mode == 'hest':
        
        ids = glob(f"{output_dir}/patches/*.h5")
    
        for input_path in tqdm(ids):
            name = os.path.splitext(os.path.basename(input_path))[0]
            st_path = f"{input_dir}/{name}.h5ad"
            adata = sc.read_h5ad(st_path)
            sample_id = preprocess_st(name, adata, output_dir, normalize=False)
            
    elif mode == 'inference':
        slide_level = args.slide_level
        patch_size = args.patch_size
        slide_ext = args.slide_ext
        
        patch_dir = f"{output_dir}/patches"
        output_dir = f"{output_dir}/pos"
        os.makedirs(output_dir, exist_ok=True)
        
        ids = glob(f"{patch_dir}/*.h5")
        
        for input_path in tqdm(ids):
            name = os.path.splitext(os.path.basename(input_path))[0]
            
            if os.path.exists(f"{output_dir}/{name}.npy"):
                print("Position already exists. Skipping...")
            else:
                get_pos(name, patch_dir, output_dir, step_size)
            
            with h5py.File(input_path, 'r') as f:
                keys = f.keys()
            
            if 'img' in keys:
                print("Image already exists. Skipping...")
            
            else:
                slide_path = f"{input_dir}/{name}{slide_ext}"
                save_image(slide_path, input_path, slide_level, patch_size)