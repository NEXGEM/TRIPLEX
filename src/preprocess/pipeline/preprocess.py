import os
import sys
import subprocess
from glob import glob
from typing import List, Dict, Union, Optional

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

# Ensure module can import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.utils import run_command


def preprocess_data(
    input_dir: str,
    output_dir: str,
    hest_dir: str = None,
    mode: str = 'train',
    platform: str = 'visium',
    slide_ext: str = '.svs',
    patch_size: int = 224,
    slide_level: int = 0,
    step_size: int = 160,
    save_neighbors: bool = False,
    num_n: int = 5
) -> None:
    """
    Run data preprocessing
    
    Args:
        input_dir: Input directory containing data
        output_dir: Output directory for processed data
        mode: Processing mode ('train', 'hest', or 'inference')
        platform: ST platform type
        slide_ext: Slide file extension
        patch_size: Size of extracted patches
        slide_level: Slide pyramid level for extraction
        step_size: Step size for patch sampling
        save_neighbors: Whether to save neighbor patches
        num_n: Number of neighbors to extract
    """
    cmd = [
        "python", "src/preprocess/prepare_data.py",
        "--input_dir", input_dir,
        "--output_dir", output_dir,
        "--mode", mode,
        "--platform", platform,
        "--slide_ext", slide_ext,
        "--patch_size", str(patch_size),
        "--slide_level", str(slide_level),
        "--step_size", str(step_size),
        "--num_n", str(num_n)
    ]
    
    if hest_dir is not None:
        cmd.extend(["--hest_dir", hest_dir])
    
    if save_neighbors:
        cmd.append("--save_neighbors")
    
    return_code, output = run_command(cmd)
    
    if return_code != 0:
        print(f"Error occurred during preprocessing: {output}")
        return False
    
    return True


def extract_features_single(
    wsi_dataroot: str,
    patch_dataroot: str,
    embed_dataroot: str,
    slide_ext: str = '.svs',
    model_name: str = 'cigar',
    num_n: int = 1,
    batch_size: int = 1024,
    num_workers: int = 4,
    overwrite: bool = False,
    id_path: Optional[str] = None
) -> None:
    """
    Extract image features
    
    Args:
        wsi_dataroot: Directory containing WSI images
        patch_dataroot: Directory containing patches
        embed_dataroot: Output directory for embeddings
        slide_ext: Slide file extension
        model_name: Model name for feature extraction
        num_n: Number of neighbors (1 for global, >1 for neighbor features)
        batch_size: Batch size for processing
        num_workers: Number of workers for data loading
        overwrite: Whether to overwrite existing embeddings
        id_path: Optional path to CSV file with sample IDs to process
    """

    cmd = [
        "python", "src/preprocess/extract_img_features.py",
        "--wsi_dataroot", wsi_dataroot,
        "--patch_dataroot", patch_dataroot,
        "--embed_dataroot", embed_dataroot,
        "--slide_ext", slide_ext,
        "--model_name", model_name,
        "--num_n", str(num_n),
        "--batch_size", str(batch_size),
        "--num_workers", str(num_workers)
    ]
    
    if overwrite:
        cmd.append("--overwrite")
    
    if id_path:
        cmd.extend(["--id_path", id_path])
    
    return_code, output = run_command(cmd)
    
    if return_code != 0:
        print(f"Error occurred during feature extraction: {output}")
        return False
    
    return True


def extract_features_parallel(
    wsi_dataroot: str,
    patch_dataroot: str,
    embed_dataroot: str,
    slide_ext: str = '.svs',
    model_name: str = 'cigar',
    num_n: int = 1,
    batch_size: int = 1024,
    num_workers: int = 4,
    overwrite: bool = False,
    gpus: List[int] = [0,1],
    sample_ids: List[str] = []
) -> None:
    """
    Extract image features
    
    Args:
        wsi_dataroot: Directory containing WSI images
        patch_dataroot: Directory containing patches
        embed_dataroot: Output directory for embeddings
        slide_ext: Slide file extension
        model_name: Model name for feature extraction
        num_n: Number of neighbors (1 for global, >1 for neighbor features)
        batch_size: Batch size for processing
        num_workers: Number of workers for data loading
        total_gpus: Total number of GPUs to use
        overwrite: Whether to overwrite existing embeddings
        id_path: Optional path to CSV file with sample IDs to process
    """
    num_gpus = len(gpus)
    
    # Split samples across GPUs
    gpu_samples = {i: [] for i in range(num_gpus)}
    for i, sample_id in enumerate(sample_ids):
        gpu_idx = i % num_gpus
        gpu_samples[gpu_idx].append(sample_id)
        
    processes = []
    for gpu_idx, samples in gpu_samples.items():
        if not samples:
            continue
            
        # Create ID file for this GPU
        id_file = f"{embed_dataroot}/gpu_{gpu_idx}_ids.csv"
        pd.DataFrame({'sample_id': samples}).to_csv(id_file, index=False)
        
        # Prepare command
        cmd = [
            "CUDA_VISIBLE_DEVICES=" + str(gpus[gpu_idx]),
            "python", "src/preprocess/extract_img_features.py",
            "--id_path", id_file,
            "--wsi_dataroot", wsi_dataroot,
            "--patch_dataroot", patch_dataroot,
            "--embed_dataroot", embed_dataroot,
            "--slide_ext", slide_ext,
            "--model_name", model_name,
            "--num_n", str(num_n),
            "--batch_size", str(batch_size),
            "--num_workers", str(num_workers),
            "--total_gpus", str(total_gpus)
        ]
        
        if overwrite:
            cmd.append("--overwrite")
        
        # Start process
        process = subprocess.Popen(" ".join(cmd), shell=True)
        processes.append(process)
    
    # Wait for all processes to complete
    for process in processes:
        process.wait()
        
    # Clean up ID files
    for gpu_idx in gpu_samples:
        id_file = f"{embed_dataroot}/gpu_{gpu_idx}_ids.csv"
        if os.path.exists(id_file):
            os.remove(id_file)
    
    return True