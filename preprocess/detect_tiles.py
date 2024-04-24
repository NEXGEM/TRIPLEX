
import os
import time
from pathlib import Path
import argparse

import pandas as pd
from multiprocessing import Pool, cpu_count

from pathml.slide import Slide


def main(wsi_paths,
         pathml_slide_dir_path,
         cases,
         batch_size,
         model_path='./models/deep-tissue-detector_densenet_state-dict.pt'):
    print("Start detection...")
    start_time = time.time()

    tile_size = 224 # pixels

    for i, wsi_path in enumerate(wsi_paths):
        file_path = os.path.join(pathml_slide_dir_path, cases[i]+'.pml')
        
        if not os.path.exists(file_path):    
            case = Path(wsi_path).stem
            pathml_slide = Slide(wsi_path, level=0).setTileProperties(tileSize=tile_size)
            pathml_slide.detectTissue(tissueDetectionTileSize=tile_size, 
                                      batchSize=batch_size, 
                                      numWorkers=0,
                                      modelStateDictPath=model_path) 
            pathml_slide.detectForeground(level=2)    
            pathml_slide.save(folder=pathml_slide_dir_path)
            
    time_elapsed = time.time() - start_time
    print('Complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--analysis_dir_path', type=str, default='.', help='dataset name:{"10x_breast_ff1","10x_breast_ff2", "10x_breast_ff3"}.')
    parser.add_argument('--wsi_path', type=str, default='/data/temp/spatial/TRIPLEX/data/test/DRP2/ST-imgs', help='')
    parser.add_argument('--meta_path', type=str, default='/data/temp/spatial/TRIPLEX/data/test/DRP2/Yale_trastuzumab_response_cohort_metadata_clean.xlsx', help='')
    parser.add_argument('--meta_colname', type=str, default='Patient', help='')
    parser.add_argument('--extension', type=str, default='svs', help='')
    parser.add_argument('--batch_size', type=int, default=32, help='')

    args = parser.parse_args()
    
    analysis_dir_path = args.analysis_dir_path
    wsi_path = args.wsi_path

    pathml_slide_dir_path = os.path.join(analysis_dir_path, 'pathml_slides')
    os.makedirs(pathml_slide_dir_path, exist_ok=True)

    # meta = pd.read_excel("/home/chungym/project/TRIPLEX-DRP/data/slide_metadata.xlsx")
    # cases = meta["Slide.ID"].astype('str').to_list()
    meta = pd.read_excel(args.meta_path)
    cases = meta[args.meta_colname].astype('str').to_list()
    wsi_paths = [os.path.join(wsi_path, str(case)+f'.{args.extension}') for case in cases]
    
    main(wsi_paths,
         pathml_slide_dir_path,
         cases,
         args.batch_size)