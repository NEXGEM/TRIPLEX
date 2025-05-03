import os
from glob import glob
import argparse
import json

import pandas as pd
import numpy as np
import anndata as ad
import scanpy as sc 


def load_data(st_dir):

    file_list = glob(f"{st_dir}/*.h5ad")
    data_list = [sc.read_h5ad(file) for file in file_list]
        
    return data_list

def find_geneset(data_list, n_top_hvg=50, n_top_heg=1000, n_top_hmhvg=200, min_spot_percentage=0.1, method='ALL'):
    """
    Get genes based on variability or total counts.
    Parameters:
    data_list (list): List of AnnData objects.
    n_top_hvg (int, optional): Number of top highly variable genes to select. Default is 50.
    n_top_heg (int, optional): Number of top highly expressed genes to select. Default is 1000.
    min_spot_percentage (float, optional): Minimum percentage of spots a gene must be expressed in to be considered. Default is 0.1.
    method (str, optional): Method to use for gene selection. Options are 'HVG' (highly variable genes), 'HEG' (highly expressed genes), or 'ALL'. Default is 'ALL'.
    Returns:
    dict: Dictionary containing selected genes. Keys are 'var' for highly variable genes and 'mean' for highly expressed genes.
    """
    """Get genes based on variability or total counts"""
    
    common_genes = list(set.intersection(*[set(adata.var_names) for adata in data_list]))
    total_spot_number = sum(adata.shape[0] for adata in data_list)
    
    spot_counts = sum((adata[:,common_genes].X > 0).sum(axis=0) for adata in data_list)
    spot_counts = np.array(spot_counts).squeeze()
    expressed_genes = np.array(common_genes)[spot_counts/total_spot_number >= min_spot_percentage]
    
    output = {}
    
    if method not in ['HMHVG', 'HVG', 'HEG', 'ALL']:
        raise ValueError("method must be either 'HVG' or 'HEG' or 'ALL'")
    
    if method in ['HMHVG', 'ALL']:
        data_lst = []
        first = True
        for adata in data_list:
            data = adata.copy()
            data_lst.append(data)
            if first:
                common_genes = data.var_names 
                first = False
                print(data.shape)
                continue
            common_genes = set(common_genes).intersection(set(data.var_names))
            print(data.shape, end="\t")

        # keep common genes
        print("Length of common genes: ", len(common_genes))
        common_genes = sorted(list(common_genes))
        for i in range(len(data_list)):
            data = data_lst[i].copy()
            data_lst[i] = data[:, common_genes].copy()
            print(data_lst[i].shape)
    
 
        union_hvg = set()

        for fn_idx in range(len(data_list)):
            adata = data_lst[fn_idx].copy()

            sc.pp.filter_cells(adata, min_genes=1)
            sc.pp.filter_genes(adata, min_cells=1)
            sc.pp.normalize_total(adata, inplace=True)
            sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(adata, n_top_genes=2000)

            union_hvg = union_hvg.union(set(adata.var_names[adata.var["highly_variable"]]))
            print(len(union_hvg))

        union_hvg = sorted([gene for gene in union_hvg if not gene.startswith(("MT", "mt", "RPS", "RPL"))]) # [optional] remove mitochondrial genes and ribosomal genes
        print(len(union_hvg))

        # select union_hvg and concat all slides
        all_count_df = pd.DataFrame(
            data_lst[0][:, union_hvg].X.toarray(),
            columns=union_hvg,
            index=[f"sample0_{i}" for i in range(data_lst[0].shape[0])]
        ).T

        for idx, adata in enumerate(data_lst[1:], start=1):
            df = pd.DataFrame(
                adata[:, union_hvg].X.toarray(),
                columns=union_hvg,
                index=[f"sample{idx}_{i}" for i in range(adata.shape[0])]
            )
            all_count_df = pd.concat([all_count_df, df.T], axis=1)


        all_count_df.fillna(0, inplace=True)
        all_count_df = all_count_df.T

        # 1. 유전자 평균 및 표준편차 계산
        gene_means = all_count_df.mean(axis=0)
        gene_stds  = all_count_df.std(axis=0)

        # 2. 각각 순위 매기기 (작을수록 상위)
        mean_ranks = gene_means.rank(ascending=False, method='min')  # 1이 가장 큼
        std_ranks  = gene_stds.rank(ascending=False, method='min')

        # 3. 점수 합산 (작을수록 mean & std 모두 상위)
        combined_score = mean_ranks + std_ranks

        # top-k 선택 및 저장
        top_genes_hmhvg = combined_score.sort_values().head(n_top_hmhvg).index
        top_genes_hmhvg = sorted(top_genes_hmhvg)
        output['hmhvg'] = top_genes_hmhvg
        print(f"Selected HMHVG genes: {output['hmhvg']}")
    
    if method in ['HVG', 'ALL']:
        data_combined = []
        for adata in data_list:
            data = adata.copy()
            sc.pp.normalize_total(data, target_sum=1e4)
            sc.pp.log1p(data)
            data_combined.append(data[:, expressed_genes])
            
        data_combined = ad.concat(data_combined, label="batch")    
        sc.pp.highly_variable_genes(data_combined, n_top_genes=n_top_hvg, batch_key="batch")
        top_genes_var = expressed_genes[data_combined.var['highly_variable']]
        output['var'] = top_genes_var.tolist()
        print(f"Selected highly variable genes: {output['var']}")

    if method in ['HEG', 'ALL']:
        gene_counts_list = []
        for adata in data_list:
            gene_counts = adata[:, expressed_genes].X.sum(axis=0)
            if len(gene_counts.shape) == 1:
                gene_counts = gene_counts.reshape(1, -1)
            gene_counts = pd.DataFrame(gene_counts)
            gene_counts_list.append(gene_counts)
        gene_counts = pd.concat(gene_counts_list, axis=0)
        total_counts = gene_counts.sum(axis=0)
        top_genes_mean = expressed_genes[total_counts.argsort()[::-1][:n_top_heg]]
        output['mean'] = top_genes_mean.tolist()
        print(f"Selected highly expressed genes: {output['mean']}")
        # top_genes = expressed_genes[total_counts.nlargest(n_top).index]
    
    return output


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--st_dir", type=str, default='input/ST/bryan/adata', help="Path to the directory containing ST data")
    argparser.add_argument("--n_top_hvg", type=int, default=50, help="Number of top HVGs to select")
    argparser.add_argument("--n_top_heg", type=int, default=1000, help="Number of top HEGs to select")
    argparser.add_argument("--n_top_hmhvg", type=int, default=200, help="Number of top HMHVGs to select")
    argparser.add_argument("--output_dir", type=str, default='input/ST/bryan')
    
    args = argparser.parse_args()
    st_dir = args.st_dir
    n_top_hvg = args.n_top_hvg
    n_top_heg = args.n_top_heg
    n_top_hmhvg = args.n_top_hmhvg
    
    method = 'ALL'
    # exist = 0 
    
    # if os.path.exists(f"{args.output_dir}/var_{n_top_hmhvg}genes.json"):
    #     method = 'HMHVG'
    
    # else:
    #     if os.path.exists(f"{args.output_dir}/var_{n_top_hvg}genes.json"):
    #         print(f"Geneset already exists in {args.output_dir}/var_{n_top_hvg}genes.json. Exiting.")
    #         method = 'HEG'
    #         exist += 1

    #     if os.path.exists(f"{args.output_dir}/mean_{n_top_heg}genes.json"):
    #         print(f"Geneset already exists in {args.output_dir}/mean_{n_top_heg}genes.json. Exiting.")
    #         exist += 1
    #         if exist == 2:
    #             print("Both genesets exist. Exiting.")
    #             exit()
    #         if exist == 1:
    #             method = 'HVG'
            
   
    data_list = load_data(st_dir)
    geneset = find_geneset(data_list, method=method, n_top_hvg=n_top_hvg, n_top_heg=n_top_heg, n_top_hmhvg=n_top_hmhvg)
    
    for prefix, genes in geneset.items():
        if prefix == 'var':
            n_top = n_top_hvg
        elif prefix == 'mean':
            n_top = n_top_heg
        elif prefix == 'hmhvg':
            n_top = n_top_hmhvg
        else:
            raise ValueError(f"Unknown gene selection method: {prefix}")
        
        with open(f"{args.output_dir}/{prefix}_{n_top}genes.json", "w") as f:
            json.dump({"genes": genes}, f)
