import argparse
import json
import os
import sys
from tqdm import tqdm
import time

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
from loguru import logger

torch.multiprocessing.set_sharing_strategy('file_system')

from hest.bench.cpath_model_zoo.inference_models import ( InferenceEncoder, 
                                                        inf_encoder_factory )
from hest.bench.utils.file_utils import save_hdf5
from hestcore.segmentation import get_path_relative

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import H5TileDataset


# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for linear probing')

### data settings ###
parser.add_argument('--overwrite', action='store_true', default=False, help='overwrite existing results')
parser.add_argument('--patch_dataroot', type=str, default='./res_tcga/patches')
parser.add_argument('--embed_dataroot', type=str, default='./res_tcga/features/neighbor')
parser.add_argument('--wsi_dataroot', type=str, default=None)
parser.add_argument('--id_path', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--level', type=int, default=1)
parser.add_argument('--use_openslide', action='store_true', default=False)
parser.add_argument('--weights_root', type=str, default='fm_v1')

### GPU settings ###
parser.add_argument('--total_gpus', type=int, default=1)
parser.add_argument('--min_gpu_id', type=int, default=0)

### specify encoder settings ###
parser.add_argument('--img_resize', type=int, default=224, help='Image resizing (-1 to not resize)')
parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for dataloader')

### specify dataset settings ###
parser.add_argument('--model_name', nargs='+', help='', default='uni_v1')
parser.add_argument('--num_n', type=int, default=5)


def post_collate_fn(batch):
    """
    Post collate function to clean up batch
    """
    if batch["imgs"].dim() == 5:
        assert batch["imgs"].size(0) == 1
        batch["imgs"] = batch["imgs"].squeeze(0)
    if batch["coords"].dim() == 3:
        assert batch["coords"].size(0) == 1
        batch["coords"] = batch["coords"].squeeze(0)
    if "mask_tb" in batch:
        if batch["mask_tb"].dim() == 3:
            assert batch["mask_tb"].size(0) == 1
            batch["mask_tb"] = batch["mask_tb"].squeeze(0)
        
    return batch

def embed_tiles(dataloader,
                model,
                embedding_save_path,
                device,
                precision=torch.float32):
    """
    Extract embeddings from tiles using encoder and save to h5 file
    """
    model.eval()
    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch = post_collate_fn(batch)
        imgs = batch['imgs']
        if imgs.shape[2] == 224:
            # status = 'global'
            with torch.inference_mode(), torch.cuda.amp.autocast(dtype=precision):
                imgs = imgs.to(device)
                embeddings = model(imgs)
        else:
            # status = 'neighbor'
            k_ranges = [(112 * 2 * k, 112 * 2 * (k + 1)) for k in range(5)]
            m_ranges = [(112 * 2 * m, 112 * 2 * (m + 1)) for m in range(5)]
            embeddings = []
            for k, (k_start, k_end) in enumerate(k_ranges):
                for m, (m_start, m_end) in enumerate(m_ranges):
                    
                    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=precision):
                        tmp = imgs[:, :, k_start:k_end, m_start:m_end].to(device)
                        embedding = model(tmp)
                        embeddings.append(embedding.cpu())
                
            embeddings = torch.stack(embeddings, dim=1)
            
        if batch_idx == 0:
            mode = 'w'
        else:
            mode = 'a'
        asset_dict = {'embeddings': embeddings.cpu().numpy()}
        asset_dict.update({key: np.array(val) for key, val in batch.items() if key != 'imgs'})
        
        save_hdf5(embedding_save_path,
                asset_dict=asset_dict,
                mode=mode)
    return embedding_save_path  

def get_bench_weights(weights_root, name):
    local_ckpt_registry = get_path_relative(__file__, 'local_ckpts.json')
    with open(local_ckpt_registry, 'r') as f:
        ckpt_registry = json.load(f)
    if name in ckpt_registry:
        path = ckpt_registry[name]
        if os.path.isabs(path):
            return path
        else:
            return os.path.join(weights_root, path)
    else:
        raise ValueError(f"Please specify the weights path to {name} in {local_ckpt_registry}")

def main(args, device):

    embedding_dir = os.path.join(args.embed_dataroot, args.model_name)
    os.makedirs(embedding_dir, exist_ok=True)
    
    if args.id_path is not None:
        if not os.path.isfile(args.id_path):
            raise ValueError(f"{args.id_path} doesn't exist")
        
        if 'sample_id' not in pd.read_csv(args.id_path).columns:
            raise ValueError("Column 'sample_id' not found in id file")
        ids = pd.read_csv(args.id_path)['sample_id'].tolist()
    
    else:
        ids = [os.path.splitext(f)[0] for f in os.listdir(args.patch_dataroot) if f.endswith('.h5')]
    
    # Embed patches
    logger.info(f"Embedding tiles using {args.model_name} encoder")
    weights_path = get_bench_weights(args.weights_root, args.model_name)    
    encoder: InferenceEncoder = inf_encoder_factory(args.model_name)(weights_path)
    precision = encoder.precision
    
    total_gpus = args.total_gpus
    min_gpu_id = args.min_gpu_id
    
    for i, sample_id in tqdm(enumerate(ids)):
        start = time.time()

        if total_gpus > 1:
            gpu_id = int(os.environ.get['CUDA_VISIBLE_DEVICES']) - min_gpu_id
            if i % total_gpus != gpu_id:  # Only process items assigned to this GPU
                continue
        
        print(f"Extracting features for {sample_id}...")
        # if sample_id == 'IOSMC_896_E_0':
        #     continue

        tile_h5_path =  f"{args.patch_dataroot}/{sample_id}.h5"
        
        if not os.path.isfile(tile_h5_path):
            print(f"{tile_h5_path} doesn't exist")
            continue
        
        embed_path = os.path.join(embedding_dir, f'{sample_id}.h5')
        if not os.path.isfile(embed_path) or args.overwrite:
            
            _ = encoder.eval()
            encoder.to(device)
                
            tile_dataset = H5TileDataset(tile_h5_path, 
                                        wsi_dir=args.wsi_dataroot, 
                                        ext=args.slide_ext, 
                                        level=args.level, 
                                        img_transform=encoder.eval_transforms, 
                                        num_n=args.num_n, 
                                        chunk_size=args.batch_size, 
                                        # num_workers=args.num_workers,
                                        use_openslide=args.use_openslide)
            
            tile_dataloader = torch.utils.data.DataLoader(tile_dataset,
                                                        batch_size=1,
                                                        shuffle=False,
                                                        num_workers=args.num_workers)
            _ = embed_tiles(tile_dataloader, encoder, embed_path, device, precision=precision)
        else:
            logger.info(f"Skipping {sample_id} as it already exists")
            
        end = time.time()
        print(f"Time taken for {sample_id}: {end - start}")
    
    
if __name__ == '__main__':
    args = parser.parse_args()    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    main(args, device)
    