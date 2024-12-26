import argparse
import os

import pandas as pd
from sklearn.model_selection import KFold, GroupKFold


def split_data_cv(df, n_splits=5, shuffle=True, random_state=42):
    os.makedirs(f'{input_dir}/splits', exist_ok=True)
    
    if 'patient' in df.columns:
        # Use GroupKFold for patient-based splitting
        gkf = GroupKFold(n_splits=n_splits)
        split_generator = gkf.split(df, groups=df['patient'])
    else:
        # Use KFold for sample-based splitting
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        split_generator = kf.split(df)

    for fold, (train_idx, test_idx) in enumerate(split_generator):
        train_data = df.iloc[train_idx]
        test_data = df.iloc[test_idx]
        
        train_data.to_csv(f'{input_dir}/splits/train_{fold}.csv', index=False)
        test_data.to_csv(f'{input_dir}/splits/test_{fold}.csv', index=False)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input_dir", type=str, help="Path to the directory containing ST data")
    argparser.add_argument("--n_splits", type=int, default=5, help="Number of splits for cross-validation")
    argparser.add_argument("--random_state", type=int, default=42, help="Random seed for splitting")
    
    args = argparser.parse_args()
    input_dir = args.input_dir
    n_splits = args.n_splits
    random_state = args.random_state

    ids_df = pd.read_csv(f'{input_dir}/ids.csv')    
    split_data_cv(ids_df, n_splits=n_splits, random_state=random_state)

