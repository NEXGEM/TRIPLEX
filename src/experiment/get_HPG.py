
import os
from glob import glob
import argparse

import numpy as np

def main(args):
    for dataset in glob(f"{args.output_dir}/{args.dataset}/*"):
        dataset_name = dataset.split('/')[-1]
        path = f"{dataset}/{args.model}"
        
        avg_pcc_rank = []
        for fold_path in glob(f"{path}/fold*"):
            if not os.path.exists(f"{fold_path}/pcc_rank.npy"):
                fold = fold_path.split('/')[-1]
                print(f"No pcc_rank.npy found in {fold} of {dataset_name}")
                continue
            
            pcc_rank = np.load(f"{fold_path}/pcc_rank.npy")
            avg_pcc_rank.append(pcc_rank)
        
        avg_pcc_rank = np.mean(avg_pcc_rank, axis=0)
        idx_top = np.argsort(avg_pcc_rank)[::-1][:50]

        np.save(f"{path}/idx_top.npy", idx_top)
        
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', type=str, default='TRIPLEX')
    argparser.add_argument('--dataset', type=str, default='hest')
    argparser.add_argument('--output_dir', type=str, default='./output/pred')
    args = argparser.parse_args()
    
    main(args)
    