
import os
import sys
from glob import glob
from tqdm import tqdm

import argparse
import numpy as np
import pandas as pd
import h5py

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_st, pxl_to_array, normalize_adata


def preprocess_st(input_path, output_dir, platform='visium'):
    fname = os.path.basename(input_path)
    
    os.makedirs(f"{output_dir}/patches", exist_ok=True)
    os.makedirs(f"{output_dir}/st", exist_ok=True)
    
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
    
    name_st = f"{output_dir}/st/{fname}.h5ad"
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
        adata = normalize_adata(adata, smooth=True)
        adata.write(name_st)
        
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
    
        
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input_dir", type=str, required=True)
    argparser.add_argument("--output_dir", type=str, required=True)
    argparser.add_argument("--platform", type=str, default='visium')
    argparser.add_argument("--prefix", type=str, default='')
    argparser.add_argument("--mode", type=str, default='cv')
    argparser.add_argument("--step_size", type=int, default=160)
    
    args = argparser.parse_args()
    
    mode = args.mode
    input_dir = args.input_dir
    output_dir = args.output_dir
    platform = args.platform
    prefix = args.prefix
    step_size = args.step_size
    
    assert mode in ['pair', 'image'], "mode must be either 'pair' or 'image'"
    
    if mode == 'pair':
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"{input_dir}/{prefix}*")
        ids = glob(f"{input_dir}/{prefix}*")
        pd.DataFrame(ids, columns=['sample_id']).to_csv(f"{output_dir}/ids.csv", index=False)
        
        for input_path in tqdm(ids):
            preprocess_st(input_path, output_dir, platform=platform)
            
    elif mode == 'image':
        output_dir = f"{output_dir}/pos"
        os.makedirs(output_dir, exist_ok=True)
        
        for input_path in glob(f"{input_dir}/*.h5"):
            get_pos(input_path, output_dir, step_size)