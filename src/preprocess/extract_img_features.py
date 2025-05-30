import argparse
import json
import os
import sys
from tqdm import tqdm
import time
import wget

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
import torch.nn as nn
import torchvision
from loguru import logger

torch.multiprocessing.set_sharing_strategy('file_system')

from hest.bench.cpath_model_zoo.inference_models import ( InferenceEncoder, 
                                                        inf_encoder_factory )
from hest.bench.utils.file_utils import save_hdf5
from hest.bench.cpath_model_zoo.utils.transform_utils import \
    get_eval_transforms
from hestcore.segmentation import get_path_relative

# from utils.transform_utils import get_transforms, add_augmentation_to_transform

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import H5TileDataset
from utils.preprocess_utils import add_augmentation_to_transform, save_hdf5
# from preprocess.utils import get_transforms

# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for linear probing')

### data settings ###
parser.add_argument('--overwrite', action='store_true', default=False, help='overwrite existing results')
parser.add_argument('--patch_dataroot', type=str, default='input/ST/andrew/patches')
parser.add_argument('--embed_dataroot', type=str, default='input/ST/andrew/emb/global')
parser.add_argument('--wsi_dataroot', type=str, default='input/ST/andrew/wsis')
parser.add_argument('--id_path', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default= '.tif')
parser.add_argument('--level', type=int, default=1)
parser.add_argument('--weights_root', type=str, default='fm_v1')
parser.add_argument('--transform_type', type=str, default='eval')

### specify encoder settings ###
parser.add_argument('--img_resize', type=int, default=224, help='Image resizing (-1 to not resize)')
parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for dataloader')

### specify dataset settings ###
parser.add_argument('--model_name', type=str, help='', default='uni_v1')
parser.add_argument('--num_n', type=int, default=1)



class CigarInferenceEncoder(InferenceEncoder):       
    def __init__(self):
        super().__init__()    
        
    def _build(
        self, _
    ):
        import timm

        model = torchvision.models.__dict__['resnet18'](weights=None)
        
        ckpt_dir = './weights/cigar'
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = f'{ckpt_dir}/tenpercent_resnet18.ckpt'
        
        # prepare the checkpoint
        if not os.path.exists(ckpt_path):
            ckpt_url='https://github.com/ozanciga/self-supervised-histopathology/releases/download/tenpercent/tenpercent_resnet18.ckpt'
            wget.download(ckpt_url, out=ckpt_dir)
            
        state = torch.load(ckpt_path)
        state_dict = state['state_dict']
        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)
        
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        if state_dict == {}:
            print('No weight could be loaded..')
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
        model.fc = nn.Identity()
        
        mean=(0.485, 0.456, 0.406)
        std=(0.229, 0.224, 0.225)
        
        eval_transform = get_eval_transforms(mean, std, target_img_size=224)
        precision = torch.float32
        
        return model, eval_transform, precision
    
    def forward(self, x):
        out = self.model(x)
        
        return out


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
                precision=torch.float32,
                transform_type=None):
    """
    Extract embeddings from tiles using encoder and save to h5 file
    """
    model.eval()
    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch = post_collate_fn(batch)
        imgs = batch['imgs']
        if 'global' in embedding_save_path:
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
            
        if (batch_idx == 0):
            mode = 'w'
        else:
            mode = 'a'
            
        asset_dict = {'embeddings': embeddings.cpu().numpy()}
        if transform_type == 'eval':
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
    if args.transform_type != 'eval':
        embedding_dir = os.path.join(embedding_dir, args.transform_type)
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
    if args.model_name == 'cigar':
        encoder = CigarInferenceEncoder()
    else:
        weights_path = get_bench_weights(args.weights_root, args.model_name)    
        encoder: InferenceEncoder = inf_encoder_factory(args.model_name)(weights_path)
        
    precision = encoder.precision
    
    for i, sample_id in tqdm(enumerate(ids)):
        start = time.time()

        print(f"Extracting features for {sample_id}...")

        tile_h5_path =  f"{args.patch_dataroot}/{sample_id}.h5"
        
        if not os.path.isfile(tile_h5_path):
            print(f"{tile_h5_path} doesn't exist")
            continue
        
        embed_path = os.path.join(embedding_dir, f'{sample_id}.h5')
        if not os.path.isfile(embed_path) or args.overwrite:
            
            _ = encoder.eval()
            encoder.to(device)
            
            transforms = add_augmentation_to_transform(
                encoder.eval_transforms, transform_type=args.transform_type)
            
            tile_dataset = H5TileDataset(tile_h5_path, 
                                        wsi_dir=args.wsi_dataroot, 
                                        ext=args.slide_ext, 
                                        level=args.level, 
                                        img_transform=transforms, 
                                        num_n=args.num_n, 
                                        chunk_size=args.batch_size, 
                                        # num_workers=args.num_workers,
                                        # use_openslide=args.use_openslide
                                        )
            
            tile_dataloader = torch.utils.data.DataLoader(tile_dataset,
                                                        batch_size=1,
                                                        shuffle=False,
                                                        num_workers=args.num_workers)
            _ = embed_tiles(tile_dataloader, encoder, embed_path, device, precision=precision, transform_type=args.transform_type)
        else:
            logger.info(f"Skipping {sample_id} as it already exists")
            
        end = time.time()
        print(f"Time taken for {sample_id}: {end - start}")
    
    
if __name__ == '__main__':
    args = parser.parse_args()    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(args.model_name)
    
    main(args, device)
    