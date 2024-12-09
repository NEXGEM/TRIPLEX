import argparse
import os

import pandas as pd
from sklearn.model_selection import KFold


def split_data_cv(data, n_splits=5, shuffle=True, random_state=42):
    os.makedirs(f'{input_dir}/splits', exist_ok=True)
    
    # Initialize KFold
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    # Create splits and save files
    for fold, (train_idx, test_idx) in enumerate(kf.split(ids_df)):
        train_data = ids_df.iloc[train_idx]
        test_data = ids_df.iloc[test_idx]
        
        train_data.to_csv(f'{input_dir}/splits/train_{fold}.csv', index=False)
        test_data.to_csv(f'{input_dir}/splits/test_{fold}.csv', index=False)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input_dir", type=str, help="Path to the directory containing ST data")
    argparser.add_argument("--n_splits", type=str, help="Number of splits for cross-validation")
    
    args = argparser.parse_args()
    input_dir = args.input_dir
    n_splits = args.n_splits

    ids_df = pd.read_csv(f'{input_dir}/ids.csv')    
    split_data_cv(ids_df, n_splits=n_splits)

