import os
import sys
import argparse
import subprocess
from glob import glob
import multiprocessing
from typing import List, Dict, Union, Optional, Tuple

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

# Ensure pipeline module can import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.utils import setup_paths, get_available_gpus
from pipeline.preprocess import (
    preprocess_data, extract_features_single, extract_features_parallel
    )


class TriplexPipeline:
    """
    Unified data processing pipeline for TRIPLEX
    
    The pipeline integrates:
    - Data preprocessing for ST data and WSI images
    - Feature extraction for image data
    - Model training and inference
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the TRIPLEX pipeline
        
        Args:
            config: Configuration dictionary with processing parameters
        """
        self.config = config or {}
        self._setup_defaults()
        self._setup_dirs()
        
        self.gpus = get_available_gpus()
        
    def _setup_defaults(self):
        """Set default configuration values if not provided"""
        defaults = {
            'mode': 'train',            # 'train', 'inference', or 'hest'
            'platform': 'visium',       # 'visium', 'xenium', 'cosmx', etc.
            'slide_ext': '.svs',        # Slide file extension
            'patch_size': 224,          # Patch size for extracting features
            'slide_level': 0,           # Slide pyramid level for preprocessing
            'step_size': 160,           # Step size for patch extraction
            'save_neighbors': False,    # Whether to save neighbor patches
            'n_splits': 4,              # Number of data splits for cross-validation
            'n_top_hvg': 50,            # Number of top highly variable genes
            'n_top_heg': 1000,          # Number of top highly expressed genes
            'model_name': 'cigar',      # Model name for feature extraction
            'num_n': 5,                 # Number of neighbors for feature extraction
            'batch_size': 1024,         # Batch size for feature extraction
            'num_workers': 4,           # Number of workers for data loading
            'total_gpus': len(self.gpus) if hasattr(self, 'gpus') else 1,
            'overwrite': False,         # Whether to overwrite existing results
        }
        
        # Update defaults with provided config
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    def _setup_dirs(self):
        """Setup directory structure"""
        # Extract directory settings from config
        self.input_dir = self.config.get('input_dir')
        self.output_dir = self.config.get('output_dir')
        
        mode = self.config['mode']
        hest_dir = self.config.get('hest_dir')
        if mode == 'hest':
            if hest_dir is None:
                self.wsi_dataroot = f"{self.input_dir}/wsis"
            else:
                self.wsi_dataroot = f"{hest_dir}/wsis"
        else:
            self.wsi_dataroot = self.input_dir
        
        # self.wsi_dataroot = f"{self.input_dir}/wsis" if self.config['mode'] == 'hest' else self.input_dir
        
        if not self.output_dir:
            raise ValueError("Output directory must be specified")
        
        # Create required directories
        self.dirs = setup_paths(self.output_dir)
    
    def preprocess(self):
        """Run preprocessing steps based on mode"""
        mode = self.config['mode']
        print(f"Running preprocessing for mode: {mode}")
        
        # Run preprocessing based on mode
        preprocess_data(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            hest_dir=self.config.get('hest_dir'),
            mode=mode,
            platform=self.config['platform'],
            slide_ext=self.config['slide_ext'],
            patch_size=self.config['patch_size'],
            slide_level=self.config['slide_level'],
            step_size=self.config['step_size'],
            save_neighbors=self.config['save_neighbors']
        )
        
        # Prepare gene sets if in training mode
        if mode in ['train', 'hest']:
            self._prepare_genesets()
            
        # Split data if in training mode
        if mode == 'train':
            self._split_data()
    
    def _prepare_genesets(self):
        """Prepare gene sets for training"""
        print("Preparing gene sets...")
        cmd = [
            "python", "src/preprocess/get_geneset.py",
            "--st_dir", f"{self.output_dir}/adata",
            "--output_dir", self.output_dir,
            "--n_top_hvg", str(self.config['n_top_hvg']),
            "--n_top_heg", str(self.config['n_top_heg'])
        ]
        subprocess.run(cmd)
    
    def _split_data(self):
        """Split data for cross-validation"""
        print("Splitting data for cross-validation...")
        cmd = [
            "python", "src/preprocess/split_data.py",
            "--input_dir", self.output_dir,
            "--n_splits", str(self.config['n_splits'])
        ]
        subprocess.run(cmd)
    
    def run_extraction(self, feature_type: str = 'both'):
        """
        Extract image features
        
        Args:
            feature_type: Type of features to extract ('global', 'neighbor', or 'both')
        """
        assert feature_type in ['global', 'neighbor', 'both'], \
            "feature_type must be 'global', 'neighbor', or 'both'"
        
        
        # Extract global features
        if feature_type in ['global', 'both']:
            print("Extracting global features...")
            extract_features_single(
                wsi_dataroot=self.wsi_dataroot,
                patch_dataroot=f"{self.output_dir}/patches",
                embed_dataroot=f"{self.output_dir}/emb/global",
                slide_ext=self.config['slide_ext'],
                model_name=self.config['model_name'],
                num_n=1,
                batch_size=self.config['batch_size'],
                num_workers=self.config['num_workers'],
                overwrite=self.config['overwrite']
            )
        
        # Extract neighbor features
        if feature_type in ['neighbor', 'both']:
            print("Extracting neighbor features...")
            extract_features_single(
                wsi_dataroot=self.wsi_dataroot,
                patch_dataroot=f"{self.output_dir}/patches/neighbor",
                embed_dataroot=f"{self.output_dir}/emb/neighbor",
                slide_ext=self.config['slide_ext'],
                model_name=self.config['model_name'],
                num_n=self.config['num_n'],
                batch_size=self.config['batch_size'],
                num_workers=self.config['num_workers'],
                overwrite=self.config['overwrite']
            )
    
    def run_parallel_extraction(self, feature_type: str = 'both'):
        """Run feature extraction in parallel using multiple GPUs"""
        
        assert feature_type in ['global', 'neighbor', 'both'], \
            "feature_type must be 'global', 'neighbor', or 'both'"
            
        gpus = self.gpus
        if not gpus:
            print("No GPUs available. Running on CPU.")
            self.extract_features()
            return
        
        # Get list of samples
        sample_ids = self._get_sample_ids()
        
        # Extract global features
        if feature_type in ['global', 'both']:
            print("Extracting global features...")
            extract_features_parallel(
                wsi_dataroot=self.wsi_dataroot,
                patch_dataroot=f"{self.output_dir}/patches",
                embed_dataroot=f"{self.output_dir}/emb/global",
                slide_ext=self.config['slide_ext'],
                model_name=self.config['model_name'],
                num_n=1,
                batch_size=self.config['batch_size'],
                num_workers=self.config['num_workers'],
                total_gpus=self.config['total_gpus'],
                overwrite=self.config['overwrite'],
                gpus=gpus,
                sample_ids=sample_ids
            )
        
        # Extract neighbor features
        if feature_type in ['neighbor', 'both']:
            print("Extracting neighbor features...")
            extract_features_parallel(
                wsi_dataroot=self.wsi_dataroot,
                patch_dataroot=f"{self.output_dir}/patches/neighbor",
                embed_dataroot=f"{self.output_dir}/emb/neighbor",
                slide_ext=self.config['slide_ext'],
                model_name=self.config['model_name'],
                num_n=self.config['num_n'],
                batch_size=self.config['batch_size'],
                num_workers=self.config['num_workers'],
                total_gpus=self.config['total_gpus'],
                overwrite=self.config['overwrite'],
                gpus=gpus,
                sample_ids=sample_ids
            )
        
    def _get_sample_ids(self):
        """Get list of sample IDs from patches directory"""
        id_file = f"{self.output_dir}/ids.csv"
        if os.path.exists(id_file):
            return pd.read_csv(id_file)['sample_id'].tolist()
        else:
            patch_files = glob(f"{self.output_dir}/patches/*.h5")
            return [os.path.splitext(os.path.basename(f))[0] for f in patch_files]
    
    def run_pipeline(self):
        """Run the complete pipeline"""
        print("Starting TRIPLEX pipeline...")
        
        # Step 1: Preprocessing
        self.preprocess()
        
        # Step 2: Feature extraction
        if self.config['total_gpus'] > 1:
            self.run_parallel_extraction('both')
        else:
            self.run_extraction('both')
            
        print("Pipeline complete!")


def main():
    """Command line interface for TRIPLEX pipeline"""
    parser = argparse.ArgumentParser(description="TRIPLEX Pipeline")
    
    # Basic configuration
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "inference", "hest"], 
                        help="Pipeline mode")
    parser.add_argument("--platform", type=str, default="visium", 
                        help="ST platform (visium, xenium, etc.)")
    
    # Preprocessing parameters
    parser.add_argument("--slide_ext", type=str, default=".svs", help="Slide file extension")
    parser.add_argument("--patch_size", type=int, default=224, help="Patch size for extraction")
    parser.add_argument("--slide_level", type=int, default=0, help="Slide pyramid level")
    parser.add_argument("--step_size", type=int, default=160, help="Step size for patch extraction")
    parser.add_argument("--save_neighbors", action="store_true", help="Save neighbor patches")
    
    # Feature extraction parameters
    parser.add_argument("--model_name", type=str, default="cigar", help="Model name for feature extraction")
    parser.add_argument("--num_n", type=int, default=5, help="Number of neighbors for feature extraction")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for feature extraction")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    
    # Additional parameters
    parser.add_argument("--n_splits", type=int, default=4, help="Number of data splits")
    parser.add_argument("--n_top_hvg", type=int, default=50, help="Number of top highly variable genes")
    parser.add_argument("--n_top_heg", type=int, default=1000, help="Number of top highly expressed genes")
    parser.add_argument("--total_gpus", type=int, default=1, help="Total GPUs to use")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results")
    
    args = parser.parse_args()
    config = vars(args)
    
    # Initialize and run pipeline
    pipeline = TriplexPipeline(config)
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()
