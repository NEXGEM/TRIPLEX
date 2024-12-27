import os
from tqdm import tqdm 

import argparse
import h5py
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import torch


def main(data_dir):
    split_dir = f"{data_dir}/splits"
    emb_dir = f"{data_dir}/emb/global/uni_v1"
    save_dir = f"{data_dir}/exemplar"
    num_fold = len(os.listdir(split_dir)) // 2

    output = []
    for fold in range(num_fold):
        print(f"Processing fold {fold}...")
        
        train_split = os.path.join(split_dir, f"train_{fold}.csv")
        test_split = os.path.join(split_dir, f"test_{fold}.csv")
        
        train_dataset = pd.read_csv(train_split)
        test_dataset = pd.read_csv(test_split)
        
        # Load training `st_embeddings` and collect their IDs
        train_names = train_dataset["sample_id"].tolist()
        train_embs = []
        name_list = []
        flag_list = []
        sid_list = []
        for name in train_names:
            h5_path = os.path.join(emb_dir, f"{name}.h5")
            with h5py.File(h5_path, "r") as f:
                img_emb = f['embeddings'][:]
                
            train_embs.append(img_emb)
            
            name_ = np.repeat(name, img_emb.shape[0])
            flag_ = np.repeat('train', img_emb.shape[0])
            sid = np.arange(img_emb.shape[0])
            
            name_list.append(name_)
            flag_list.append(flag_)
            sid_list.append(sid)
                
        train_embs = np.concatenate(train_embs, axis=0)
        print(f"Loaded {train_embs.shape[0]} `image embeddings` for training.")
        
        # Load testing `img_embeddings` and collect their IDs
        test_names = test_dataset["sample_id"].tolist()
        test_embs = []
        for name in test_names:
            h5_path = os.path.join(emb_dir, f"{name}.h5")
            with h5py.File(h5_path, "r") as f:
                img_emb = f['embeddings'][:]
            
            test_embs.append(img_emb)
            
            name_ = np.repeat(name, img_emb.shape[0])
            flag_ = np.repeat('test', img_emb.shape[0])
            sid = np.arange(img_emb.shape[0])

            name_list.append(name_)
            flag_list.append(flag_)
            sid_list.append(sid)
            
        test_embs = np.concatenate(test_embs, axis=0)
        print(f"Loaded {test_embs.shape[0]} `image embeddings` for testing.")
        
        embs = np.concatenate((train_embs, test_embs), axis=0)
        names = np.concatenate(name_list, axis=0)
        flags = np.concatenate(flag_list, axis=0)
        sids = np.concatenate(sid_list, axis=0)
        
        unique_name = np.unique(names)
        result = {}
        for n in tqdm(unique_name):
            idx_query = names == n
            
            idx_key = (~idx_query) & (flags == 'train')
            if idx_key.sum() == 0:
                idx_key = (flags == 'train')
                
            emb_query = torch.Tensor(embs[idx_query]).cuda()
            emb_key = torch.Tensor(embs[idx_key]).cuda()
            
            dist = torch.cdist(emb_query, emb_key, p = 2).squeeze(0) 
            topk = min(len(dist), 100)
                
            knn = dist.topk(topk, dim = 1, largest=False)
            
            knn_indices = knn.indices.cpu().numpy()
            knn_values = knn.values.cpu().numpy()
            
            ex_sid = []
            ex_name = []
            for i in range(knn_indices.shape[0]):
                indices = knn_indices[i]
                name_ = names[idx_key][indices]
                sid_ = sids[idx_key][indices]
                
                ex_sid.append(sid_)
                ex_name.append(name_)

            ex_sid = np.stack(ex_sid, axis=0)
            ex_name = np.stack(ex_name, axis=0)
            
            flag_ = flags[idx_query][0]
            
            save_path = f"{save_dir}/fold{fold}/{flag_}"
            os.makedirs(save_path, exist_ok=True)
            
            with h5py.File(f"{save_path}/{n}.h5", "w") as f:
                f.create_dataset("sid", data=ex_sid)
                f.create_dataset("pid", data=ex_name.astype('S13'))
            
            result[n] = (ex_sid, ex_name)    
            
        output.append(result)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build exemplar dataset")
    parser.add_argument("--data_dir", type=str, default="./input/smc/lung", help="Path to the data directory")
    
    args = parser.parse_args()
    data_dir = args.data_dir
    
    main(data_dir)