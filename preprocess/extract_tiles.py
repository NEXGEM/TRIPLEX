
import os
import time
from pathlib import Path
import argparse
import pandas as pd
from multiprocessing import Pool, cpu_count

from pathml.slide import Slide


def process_slide(case):
    analysis_dir_path = '.'
    pathml_slide_dir_path = os.path.join(analysis_dir_path, 'pathml_slides')
    
    tl_thres = 0.9
    fgl_thre = 88
    
    pathml_slide = Slide(os.path.join(pathml_slide_dir_path, case+'.pml'))
    counts_all = len(pathml_slide.suitableTileAddresses(tissueLevelThreshold=tl_thres, 
                                                    foregroundLevelThreshold=fgl_thre))
    pathml_slide.extractRandomUnannotatedTiles(
        analysis_dir_path, 
        numTilesToExtract=counts_all, 
        extractSegmentationMasks=True, 
        tissueLevelThreshold=tl_thres, 
        foregroundLevelThreshold=fgl_thre)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--wsi_path', type=str, default='/data/temp/spatial/TRIPLEX/data/test/DRP2/ST-imgs', help='')
    parser.add_argument('--meta_path', type=str, default='/data/temp/spatial/TRIPLEX/data/test/DRP2/Yale_trastuzumab_response_cohort_metadata_clean.xlsx', help='')
    parser.add_argument('--meta_colname', type=str, default='Patient', help='')
    
    
    args = parser.parse_args()
    # meta = pd.read_excel("/home/chungym/project/TRIPLEX-DRP/data/slide_metadata.xlsx")
    # cases = meta["Slide.ID"].astype('str').to_list()
    meta = pd.read_excel(args.meta_path)
    cases = meta[args.meta_colname].astype('str').to_list()
    
    print("Start tile extraction...")
    
    start_time = time.time()
    
    # Determine the number of processes
    num_processes = min(len(cases), cpu_count())
    
    # Create a pool of processes
    with Pool(processes=num_processes) as pool:
        # Map process_slide function to each case
        pool.map(process_slide, cases)
    
    time_elapsed = time.time() - start_time
    print('Complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
