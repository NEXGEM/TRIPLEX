
import os
from glob import glob
from tqdm import tqdm

import argparse
import pandas as pd
import h5py

from utils import load_st


def preprocess_st(input_dir, output_dir, name, platform='visium'):
    os.makedirs(f"{output_dir}/patches", exist_ok=True)
    os.makedirs(f"{output_dir}/st", exist_ok=True)
    
    print("Loading ST data...")
    st = load_st(input_dir, platform=platform)
    
    print("Segmenting tissue...")
    st.segment_tissue(method='deep')
    
    if os.path.exists(f"{output_dir}/patches/{name}.h5"):
        print("Patches already exists. Skipping...")
    else:
        print("Dumping patches...")
        st.dump_patches(
            f"{output_dir}/patches",
            name=name,
            target_patch_size=224, # target patch size in 224
            target_pixel_size=0.5 # pixel size of the patches in um/px after rescaling
        )
    
    name_st = f"{output_dir}/st/{name}.h5ad"
    if os.path.exists(name_st):
        print("ST data already exists. Skipping...")
    else:
        print("Dumping matched ST data...")
        with h5py.File(f"{output_dir}/patches/{name}.h5", "r") as f:
            barcode = f['barcode'][:].astype('str').squeeze()
            
        barcode = pd.DataFrame(index=barcode)
        barcode_merged = pd.merge(st.adata.obs, barcode, left_index=True, right_index=True).index
        adata = st.adata[barcode_merged]
        adata.write(name_st)
        
        
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input_dir", type=str, required=True)
    argparser.add_argument("--output_dir", type=str, required=True)
    argparser.add_argument("--platform", type=str, default='visium')
    argparser.add_argument("--prefix", type=str, default='')
    
    args = argparser.parse_args()
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    platform = args.platform
    prefix = args.prefix
    
    os.makedirs(output_dir, exist_ok=True)
    
    ids = glob(input_dir + "/{prefix}*")
    pd.DataFrame(ids, columns=['sample_id']).to_csv(f"{output_dir}/ids.csv", index=False)
    
    for path in tqdm(ids):
        preprocess_st(path, output_dir, name=path.split('/')[-1], platform=platform)