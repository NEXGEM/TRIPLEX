import os
import sys
import subprocess
import importlib
from typing import List, Dict, Tuple, Optional

import torch


def setup_paths(output_dir: str) -> Dict[str, str]:
    """
    Create necessary directories for the pipeline
    
    Args:
        output_dir: Base output directory
        
    Returns:
        Dictionary of created paths
    """
    paths = {
        'patches': os.path.join(output_dir, 'patches'),
        'patches_neighbor': os.path.join(output_dir, 'patches', 'neighbor'),
        'adata': os.path.join(output_dir, 'adata'),
        'emb': os.path.join(output_dir, 'emb'),
        'emb_global': os.path.join(output_dir, 'emb', 'global'),
        'emb_neighbor': os.path.join(output_dir, 'emb', 'neighbor'),
        'pos': os.path.join(output_dir, 'pos'),
    }
    
    # Create directories
    for name, path in paths.items():
        if name in ['emb_global', 'emb_neighbor']:
            os.makedirs(path, exist_ok=True)
    
    return paths


def check_dependencies() -> bool:
    """
    Check if all required dependencies are installed
    
    Returns:
        True if all dependencies are met, False otherwise
    """
    required_packages = [
        'torch', 'numpy', 'pandas', 'h5py', 'scanpy', 
        'anndata', 'openslide', 'tqdm', 'Pillow', 'timm'
    ]
    
    missing = []
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Missing required packages: {', '.join(missing)}")
        print("Please install them with: pip install " + " ".join(missing))
        return False
    
    return True


def get_available_gpus() -> List[int]:
    """
    Get list of available GPUs
    
    Returns:
        List of available GPU indices
    """
    if not torch.cuda.is_available():
        return []
    
    # gpu_count = torch.cuda.device_count()
    gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    return [int(gpu) for gpu in gpus]


def split_list_for_gpus(items: List, num_gpus: int) -> List[List]:
    """
    Split a list of items among multiple GPUs
    
    Args:
        items: List of items to split
        num_gpus: Number of GPUs to split items among
        
    Returns:
        List of lists, where each inner list contains items for one GPU
    """
    result = [[] for _ in range(num_gpus)]
    for i, item in enumerate(items):
        gpu_idx = i % num_gpus
        result[gpu_idx].append(item)
    return result


def run_command(cmd: List[str], verbose: bool = True) -> Tuple[int, str]:
    """
    Run a shell command and return the output
    
    Args:
        cmd: Command to run as a list of strings
        verbose: Whether to print command and output
        
    Returns:
        Tuple of (return_code, output)
    """
    # if verbose:
    #     print(f"Running command: {' '.join(cmd)}")
    
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    output = ""
    for line in process.stdout:
        output += line
        if verbose:
            print(line, end="")
    
    return_code = process.wait()
    return return_code, output