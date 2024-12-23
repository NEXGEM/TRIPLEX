
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

def find_geneset(data_list, n_top_hvg=50, n_top_heg=1000, min_spot_percentage=0.1, method='ALL'):
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
    
    else:
        raise ValueError("method must be either 'HVG' or 'HEG' or 'ALL")
    
    return output


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--st_dir", type=str, default='input/ST/bryan/adata', help="Path to the directory containing ST data")
    argparser.add_argument("--n_top_hvg", type=int, default=50, help="Number of top HVGs to select")
    argparser.add_argument("--n_top_heg", type=int, default=1000, help="Number of top HEGs to select")
    argparser.add_argument("--output_dir", type=str, default='input/ST/bryan')
    
    args = argparser.parse_args()
    st_dir = args.st_dir
    n_top_hvg = args.n_top_hvg
    n_top_heg = args.n_top_heg
    
    data_list = load_data(st_dir)
    geneset = find_geneset(data_list, method='ALL', n_top_hvg=n_top_hvg, n_top_heg=n_top_heg)
    # geneset_exp = find_geneset(data_list, method='HEG', n_top=n_top_heg)
    # geneset = {'hvg': geneset_var.tolist(), 'heg': geneset_exp.tolist()}
    
    for prefix in ['var', 'mean']:
        n_top = n_top_hvg if prefix == 'var' else n_top_heg
        with open(f"{args.output_dir}/{prefix}_{n_top}genes.json", "w") as f:
            json.dump({"genes": geneset[prefix]}, f)
