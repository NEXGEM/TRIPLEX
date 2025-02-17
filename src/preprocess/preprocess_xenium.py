from hest import XeniumReader
from glob import glob
import os 
from tqdm import tqdm


xenium_input_dir = '/home/shared/spRNAseq/takano/Xenium'
file_lists = glob(xenium_input_dir + '/*')
print("file_lists: ", file_lists)

for path in tqdm(file_lists):
    file_name = os.path.basename(path)
    if file_name in ['Xenium_TSU-20']:
        continue
    
    input_dir = f'/home/shared/spRNAseq/takano/Xenium/{file_name}'
    output_dir = f'processed_sample/{file_name}'
    
    st = XeniumReader().auto_read(input_dir)
    
    # Path to the H&E image we just saved
    he_path = f'{output_dir}/aligned_fullres_HE.tif'
    # he_path = f'{input_dir}/XeniumHE_LUAD_No14.tif'
    # he_path = f'{input_dir}/{file_name}.ome.tif'
    dapi_path = f'{input_dir}/morphology_focus.ome.tif'
    
    if not os.path.exists(he_path):
        print("Pyramidal tiff not found. Saving...")
        # Save image to Generic pyramidal tiff
        st.save(output_dir, save_img=True)
    
    # save_dir = f'valis_results/{file_name}'
    st.align_with_valis(output_dir, he_path, dapi_path, align_transcripts=True, align_cells=True, align_nuclei=True)