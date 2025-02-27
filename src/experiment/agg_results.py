
import os
import argparse
from glob import glob
import pandas as pd
import numpy as np


def main(args):    
    
    results = []
    for dataset in glob(f"{args.log_dir}/{args.dataset}/*"):
        
        if not os.path.isdir(f"{dataset}/{args.model}"):
            if args.model in dataset:
                dataset_name = args.dataset
                # path = glob(f"{args.log_dir}/{args.dataset}/{args.model}/*")[-1]
                path = f"{args.log_dir}/{args.dataset}/{args.model}/{args.log_name}"
            else:
                continue
        
        else:
            dataset_name = dataset.split('/')[-1]
            # path = glob(f"{dataset}/{args.model}/*")[-1]
            path = f"{dataset}/{args.model}/{args.log_name}"
            
        
        for fold_path in glob(f"{path}/*"):
            if not os.path.exists(f"{fold_path}/eval/metrics.csv"):
                fold = fold_path.split('/')[-1]
                print(f"No metrics.csv found in {fold} of {dataset_name}")
                continue
            metrics = pd.read_csv(f"{fold_path}/eval/metrics.csv")
            idx = metrics.test_PearsonCorrCoef.argmax()
            
            # Get test metric columns dynamically
            test_cols = [col for col in metrics.columns if col.startswith('test_')]
            
            idx = metrics.test_PearsonCorrCoef.argmax()
            best_metrics = metrics.iloc[idx,:]
            
            if not 'metric_lists' in locals():
                metric_lists = {col: [] for col in test_cols}
            
            for col in test_cols:
                metric_lists[col].append(best_metrics[col])
                    
        results.append({
            'Dataset': dataset_name,
            **{f"{col.replace('test_', '')}_mean": np.mean(metric_lists[col]) for col in test_cols},
            **{f"{col.replace('test_', '')}_std": np.std(metric_lists[col]) for col in test_cols}
        })
        
    results_df = pd.DataFrame(results)
    means = results_df.iloc[:, 1:].mean()
    results_df.loc[len(results_df)] = ['Average'] + list(means)
    print(results_df.to_string(index=False))
    
    output_path = f"{args.output_dir}/{args.dataset}/{args.model}/results.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)
    
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', type=str, default='TRIPLEX')
    argparser.add_argument('--log_dir', type=str, default='./logs')
    argparser.add_argument('--log_name', type=str, default='2025-02-24-14-35')
    argparser.add_argument('--dataset', type=str, default='GSE240429')
    argparser.add_argument('--output_dir', type=str, default='./output/res')
    args = argparser.parse_args()
    
    main(args)
    