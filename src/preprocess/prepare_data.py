import os
import sys
from glob import glob
from tqdm import tqdm
from pathlib import Path
import shutil

import argparse
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

import h5py
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from openslide import OpenSlide
import multiprocessing as mp
import scanpy as sc

import datasets
from hest import iter_hest


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocess_utils import load_st, pxl_to_array, save_hdf5


def find_matches(array_i, array_j):
    """
    For each point in array_i, find an exact match in array_j if it exists,
    otherwise find the nearest point in array_j.
    
    Args:
        array_i: First array of points, shape (n, d)
        array_j: Second array of points, shape (m, d)
        
    Returns:
        matches: Array of indices in array_j that match or are closest to points in array_i
    """
    n_points = len(array_i)
    matches = np.zeros(n_points, dtype=int)
    
    # Calculate pairwise distances between all points
    distances = cdist(array_i, array_j)
    
    # For each point in array_i
    for i in range(n_points):
        # First check for exact matches (distance = 0)
        exact_matches = np.where(distances[i] == 0)[0]
        
        if len(exact_matches) > 0:
            # If exact matches exist, use the first one
            matches[i] = exact_matches[0]
        else:
            # Otherwise, find the nearest point
            matches[i] = np.argmin(distances[i])
    
    return matches

# def match_to_target(coords_target, coords_neighbor, dst_pixel_size, src_pixel_size, num_n):
    
#     patch_size_target = 224 * (dst_pixel_size / src_pixel_size)
#     patch_size_neighbor = 224 * num_n * (dst_pixel_size / src_pixel_size)
#     coords_target = coords_target + int(patch_size_target // 2)
#     coords_neighbor = coords_neighbor + int(patch_size_neighbor // 2)
    
#     matches = find_matches(coords_target, coords_neighbor)
    
#     return matches

def match_to_target(target_path, neighbor_path):
    with h5py.File(target_path, 'r') as f:
        barcode_target = f['barcode'][:].astype('str').squeeze()
            
    with h5py.File(neighbor_path, 'r+') as f:
        neighbor_img = f['img'][:]
        crd_neighbor = f['coords'][:]
        barcode_neighbor = f['barcode'][:]
        
        if len(barcode_target) != len(barcode_neighbor):
            print("Mismatch between target and neighbor barcodes")
            
            idx_matched = np.intersect1d(barcode_target, 
                                        barcode_neighbor.astype('str').squeeze(), 
                                        return_indices=True)[2]

            del f['coords']
            f.create_dataset('coords', data=crd_neighbor[idx_matched])
            del f['img']
            f.create_dataset('img', data=neighbor_img[idx_matched])
            del f['barcode']
            f.create_dataset('barcode', data=barcode_neighbor[idx_matched])
            f.attrs['matched_to_target'] = True


def preprocess_st(name, adata, output_dir):
    os.makedirs(f"{output_dir}/adata", exist_ok=True)
    save_dir = f"{output_dir}/adata/{name}.h5ad"
    
    if os.path.exists(save_dir):
        print(f"ST data already exists for {name}. Skipping...")
        return None
    print(f"Saving ST data for {name}...")
    
    print("Dumping matched ST data...")
    with h5py.File(f"{output_dir}/patches/{name}.h5", "r") as f:
        barcode = f['barcode'][:].astype('str').squeeze()
        
    barcode = pd.DataFrame(index=barcode)
    barcode_merged = pd.merge(adata.obs, barcode, left_index=True, right_index=True).index
    adata = adata[barcode_merged]
    
    adata.write(save_dir)
    
    return adata

def save_patches(name, input_dir, output_dir, platform='visium', save_targets=True, save_neighbors=False, num_n=5):

    print("Loading ST data...")
    if platform == 'hest':
        st = [st for st in iter_hest(input_dir, id_list=[name])][0]
    
    else:
        try:
            st = load_st(f"{input_dir}/{name}", platform=platform)
        except:
            print("Failed to load ST data. Move to next sample.")
            return None
    
    dst_pixel_size = 0.5
    
    if save_targets:
        if os.path.exists(f"{output_dir}/patches/{name}.h5"):
            print("Target patches already exists. Skipping...")
        else:
            if st._tissue_contours is None:
                print("Segmenting tissue...")
                st.segment_tissue(method='deep')

            print("Dumping target patches...")
            st.dump_patches(
                f"{output_dir}/patches",
                name=name,
                target_patch_size=224, # target patch size in 224
                target_pixel_size=dst_pixel_size, # pixel size of the patches in um/px after rescaling
                dump_visualization=False
            )
        
    if save_neighbors:
        if os.path.exists(f"{output_dir}/patches/neighbor/{name}.h5"):
            print("Neighbor patches already exists. Skipping...")
        else:  
            if st._tissue_contours is None:
                print("Segmenting tissue...")
                st.segment_tissue(method='deep')
                
            print("Dumping neighbor patches...")
            st.dump_patches(
                f"{output_dir}/patches/neighbor",
                name=name,
                target_patch_size=224*num_n, # neighbor patch size in 1120
                target_pixel_size=dst_pixel_size, # pixel size of the patches in um/px after rescaling,
                threshold=int(0.15 // num_n),
                dump_visualization=False
            )
        
            print("Matching neighbor patches to target patches...")
            target_path = f"{output_dir}/patches/{name}.h5"
            neighbor_path = f"{output_dir}/patches/neighbor/{name}.h5"
                        
            match_to_target(target_path, neighbor_path)

    return st
        
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
    argparser.add_argument("--hest_dir", type=str, default=None)
    argparser.add_argument("--platform", type=str, default='visium')
    argparser.add_argument("--prefix", type=str, default='')
    argparser.add_argument("--mode", type=str, default='train')
    argparser.add_argument("--step_size", type=int, default=160)
    
    argparser.add_argument("--slide_level", type=int, default=0)
    argparser.add_argument("--slide_ext", type=str, default='.svs')
    argparser.add_argument("--patch_size", type=int, default=256)
    argparser.add_argument("--num_n", type=int, default=5)
    argparser.add_argument("--save_neighbors", action='store_true', default=False)
    
    args = argparser.parse_args()
    
    mode = args.mode
    input_dir = args.input_dir
    output_dir = args.output_dir
    hest_dir = args.hest_dir
    platform = args.platform
    prefix = args.prefix
    step_size = args.step_size
    
    assert mode in ['train', 'hest', 'inference'], "mode must be either 'train' or 'hest' or 'inference'"
    
    if mode == 'train':
        os.makedirs(f"{output_dir}/patches", exist_ok=True)
        if args.save_neighbors:
            os.makedirs(f"{output_dir}/patches/neighbor", exist_ok=True)
        os.makedirs(f"{output_dir}/adata", exist_ok=True)
        
        if not os.path.exists(f"{output_dir}/ids.csv"):
            ids = glob(f"{input_dir}/{prefix}*")
            sample_ids = [os.path.basename(id) for id in ids]
            pd.DataFrame(sample_ids, columns=['sample_id']).to_csv(f"{output_dir}/ids.csv", index=False)
            sample_ids = [os.path.basename(id) for id in ids]
            pd.DataFrame(sample_ids, columns=['sample_id']).to_csv(f"{output_dir}/ids.csv", index=False)
        else:
            ids = pd.read_csv(f"{output_dir}/ids.csv")['sample_id'].tolist()
            
        sample_ids = []
        for input_path in tqdm(ids):
            name = os.path.basename(input_path)
            st = save_patches(name, 
                            input_dir, 
                            output_dir, 
                            platform=platform, 
                            save_neighbors=args.save_neighbors)
            
            if st is not None:
                sample_ids.append(name)
                preprocess_st(name, st.adata, output_dir)
            
    elif mode == 'hest':
            
        if os.path.exists(f"{output_dir}/ids.csv"):
            ids = pd.read_csv(f"{output_dir}/ids.csv")['sample_id'].tolist()

            if hest_dir is not None:
                # Create output patches directory if it doesn't exist
                output_patches_dir = f"{output_dir}/patches"
                os.makedirs(output_patches_dir, exist_ok=True)
                
                # Copy only the specific h5 files for the ids in the list
                hest_patches_dir = f"{hest_dir}/patches"
                if os.path.exists(hest_patches_dir):
                    print(f"Copying specific patch files from {hest_patches_dir} to {output_patches_dir}...")
                    for id_name in tqdm(ids):
                        src_file = f"{hest_patches_dir}/{id_name}.h5"
                        dst_file = f"{output_patches_dir}/{id_name}.h5"
                        if os.path.exists(src_file):
                            if not os.path.exists(dst_file):
                                shutil.copy2(src_file, dst_file)
                            else:
                                print(f"File {dst_file} already exists. Skipping.")
                        else:
                            print(f"Source file {src_file} does not exist. Skipping.")
                else:
                    print(f"Source directory {hest_patches_dir} does not exist. Nothing to copy.")
            
            elif not os.path.exists(f"{output_dir}/patches"):
                list_patterns = [f"*{id}[_.]**" for id in ids]
                datasets.load_dataset(
                    'MahmoodLab/hest', 
                    cache_dir=output_dir,
                    patterns=list_patterns
                )   
                
        else:
            ids = glob(f"{output_dir}/patches/*.h5")
            sample_ids = [os.path.splitext(os.path.basename(id))[0] for id in ids]
            pd.DataFrame(sample_ids, columns=['sample_id']).to_csv(f"{output_dir}/ids.csv", index=False)
            sample_ids = [os.path.splitext(os.path.basename(id))[0] for id in ids]
            pd.DataFrame(sample_ids, columns=['sample_id']).to_csv(f"{output_dir}/ids.csv", index=False)
            
        input_dir = input_dir if hest_dir is None else hest_dir
        for input_path in tqdm(ids):
            name = os.path.splitext(os.path.basename(input_path))[0]
            
            if args.save_neighbors:
                os.makedirs(f"{output_dir}/patches/neighbor", exist_ok=True)
                st = save_patches(name, 
                                input_dir,
                                output_dir, 
                                platform='hest', 
                                save_targets=False,
                                save_neighbors=args.save_neighbors,
                                num_n=args.num_n)
            
            st_path = f"{input_dir}/st/{name}.h5ad"
            adata = sc.read_h5ad(st_path)
            preprocess_st(name, adata, output_dir)
            
    elif mode == 'inference':
        slide_level = args.slide_level
        patch_size = args.patch_size
        slide_ext = args.slide_ext
        
        patch_dir = f"{output_dir}/patches"
        output_dir = f"{output_dir}/pos"
        os.makedirs(output_dir, exist_ok=True)
        
        ids = glob(f"{patch_dir}/*.h5")
        sample_ids = [os.path.splitext(os.path.basename(id))[0] for id in ids]
        pd.DataFrame(sample_ids, columns=['sample_id']).to_csv(f"{output_dir}/ids.csv", index=False)
        sample_ids = [os.path.splitext(os.path.basename(id))[0] for id in ids]
        pd.DataFrame(sample_ids, columns=['sample_id']).to_csv(f"{output_dir}/ids.csv", index=False)
        
        for input_path in tqdm(ids):
            name = os.path.splitext(os.path.basename(input_path))[0]
            
            with h5py.File(input_path, 'r') as f:
                keys = f.keys()
            
            if 'img' in keys:
                print("Image already exists. Skipping...")
            
            else:
                slide_path = f"{input_dir}/{name}{slide_ext}"
                save_image(slide_path, input_path, slide_level, patch_size)