
import os
import argparse
from glob import glob
import pandas as pd
import numpy as np


def main(args):    
    
    results = []
    for dataset in glob(f"{args.log_dir}/{args.dataset}/*"):
        dataset_name = dataset.split('/')[-1]
        path = glob(f"{dataset}/{args.model}/*")[-1]
        pccs, mses, cccs, evs, maes = [], [], [], [], []
        
        for fold_path in glob(f"{path}/*"):
            if not os.path.exists(f"{fold_path}/eval/metrics.csv"):
                fold = fold_path.split('/')[-1]
                print(f"No metrics.csv found in {fold} of {dataset_name}")
                continue
            metrics = pd.read_csv(f"{fold_path}/eval/metrics.csv")
            idx = metrics.test_PearsonCorrCoef.argmax()
            best_metrics = metrics.iloc[idx,:]
            best_pcc = best_metrics.test_PearsonCorrCoef
            best_ccc = best_metrics.test_ConcordanceCorrCoef
            best_mse = best_metrics.test_MeanSquaredError
            best_mae = best_metrics.test_MeanAbsoluteError
            best_ev = best_metrics.test_ExplainedVariance
            
            pccs.append(best_pcc)
            mses.append(best_mse)
            cccs.append(best_ccc)
            evs.append(best_ev)
            maes.append(best_mae)
            
        results.append({
            'Dataset': dataset_name,
            'PCC_mean': np.mean(pccs),
            'PCC_std': np.std(pccs),
            'MSE_mean': np.mean(mses),
            'MSE_std': np.std(mses),
            'CCC_mean': np.mean(cccs),
            'CCC_std': np.std(cccs),
            'EV_mean': np.mean(evs),
            'EV_std': np.std(evs),
            'MAE_mean': np.mean(maes),
            'MAE_std': np.std(maes)
        })

    results_df = pd.DataFrame(results)
    means = results_df.iloc[:, 1:].mean()
    results_df.loc[len(results_df)] = ['Average'] + list(means)
    print(results_df.to_string(index=False))
    
    output_path = f"{args.output_dir}/{args.dataset}/res/{args.model}/results.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)
    
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', type=str, default='TRIPLEX')
    argparser.add_argument('--log_dir', type=str, default='./logs')
    argparser.add_argument('--dataset', type=str, default='hest')
    argparser.add_argument('--output_dir', type=str, default='./output')
    args = argparser.parse_args()
    
    main(args)
    