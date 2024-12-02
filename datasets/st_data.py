
import os
from glob import glob
import warnings
warnings.filterwarnings('ignore')

import pickle 
import numpy as np
import pandas as pd 
from sklearn.model_selection import KFold
from PIL import ImageFile, Image
import torch
import torchvision
import torchvision.transforms as transforms
import scprep as scp

from utils import smooth_exp
from openslide import OpenSlide

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class BaselineDataset(torch.utils.data.Dataset):
    """Some Information about baselines"""
    def __init__(self):
        super(BaselineDataset, self).__init__()
        
        self.train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomApply([torchvision.transforms.RandomRotation((90, 90))]),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        
        self.test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def get_img(self, name: str):
        """Load whole slide image of a sample.

        Args:
            name (str): name of a sample

        Returns:
            PIL.Image: return whole slide image.
        """
        
        img_dir = self.data_dir+'/ST-imgs'
        if self.data == 'her2st':
            pre = img_dir+'/'+name[0]+'/'+name
            fig_name = os.listdir(pre)[0]
            path = pre+'/'+fig_name
        elif self.data == 'stnet' or '10x_breast' in self.data or 'GBM_data' in self.data:
            path = glob(img_dir+'/*'+name+'.tif')[0]
        #elif 'DRP' in self.data:
        elif 'DRP' in self.data or 'TCGA-GBM' in self.data:
            path = glob(img_dir+'/*'+name+'.svs')[0]
        else:
            path = glob(img_dir+'/*'+name+'.jpg')[0]
    
        if self.use_pyvips:    
            import pyvips as pv
            im = pv.Image.new_from_file(path, level=0)
        else:
            #im = Image.open(path)
            im = OpenSlide(path)
    
        
        return im
    
    def get_cnt(self, name: str):
        """Load gene expression data of a sample.

        Args:
            name (str): name of a sample

        Returns:
            pandas.DataFrame: return gene expression. 
        """
        path = self.data_dir+'/ST-cnts/'+name+'_sub.parquet'
        df = pd.read_parquet(path)

        return df

    def get_pos(self, name: str):
        """Load position information of a sample.
        The 'id' column is for matching against the gene expression table.

        Args:
            name (str): name of a sample

        Returns:
            pandas.DataFrame: return DataFrame with position information.
        """
        path = self.data_dir+'/ST-spotfiles/'+name+'_selection.tsv'
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id

        return df

    def get_meta(self, name: str):
        """Load both gene expression and postion data and merge them.

        Args:
            name (str): name of a sample

        Returns:
            pandas.DataFrame: return merged table (gene exp + position)
        """
        
        pos = self.get_pos(name)
        
        #if 'DRP' not in self.data:
        if self.mode != 'inference' and 'DRP' not in self.data:
            cnt = self.get_cnt(name)
            meta = cnt.join(pos.set_index('id'),how='inner')
        else:
            meta = pos
        
        if (self.mode == "external_test") or (self.data == "GBM_data"):
            meta = meta.sort_values(['x', 'y'])
        else:
            meta = meta.sort_values(['y', 'x'])

        return meta


class STDataset(BaselineDataset):
    """Dataset to load ST data for TRIPLEX
    """
    def __init__(self, 
                 mode: str, 
                 fold: int=0, 
                 extract_mode: str=None, 
                 test_data=None, 
                 **kwargs):
        """
        Args:
            mode (str): 'train', 'test', 'external_test', 'extraction', 'inference'.
            fold (int): Number of fold for cross validation.
            test_data (str, optional): Test data name. Defaults to None.
        """
        super().__init__()
        
        # Set primary attribute
        self.gt_dir = kwargs['t_global_dir']
        self.num_neighbors = kwargs['num_neighbors']
        self.neighbor_dir = f"{kwargs['neighbor_dir']}_{self.num_neighbors}_224"
        
        self.use_pyvips = kwargs['use_pyvips']
        
        self.r = kwargs['radius']//2
        self.extract_mode = extract_mode
        
        self.mode = mode
        if test_data:
            self.data = test_data
            self.data_dir = f"{kwargs['data_dir']}/test/{self.data}"    
        else:
            self.data = kwargs['type']
            self.data_dir = f"{kwargs['data_dir']}/{self.data}"
    
        names = os.listdir(self.data_dir+'/ST-spotfiles')
        names.sort()
        names = [i.split('_selection.tsv')[0] for i in names]
        
        if mode in ["external_test", "inference"]:
            self.names = names
            
        elif mode == "extraction":
            # self.names = np.array_split(names, 2)[node_id]
            self.names = names
            if extract_mode == "neighbor":
                self.names = [name for name in self.names if not os.path.exists(os.path.join(self.neighbor_dir, f"{name}.pt"))]
            elif extract_mode == "target":
                self.names = [name for name in self.names if not os.path.exists(os.path.join(self.gt_dir, f"{name}.pt"))]
            
        else:
            self.fold_name = None
            if self.data == 'stnet':
                kf = KFold(8, shuffle=True, random_state=2021)
                patients = ['BC23209','BC23270','BC23803','BC24105',
                            'BC24220','BC23268','BC23269','BC23272',
                            'BC23277','BC23287','BC23288','BC23377',
                            'BC23450','BC23506','BC23508','BC23567',
                            'BC23810','BC23895','BC23901','BC23903',
                            'BC23944','BC24044','BC24223']
                patients = np.array(patients)
                _, ind_val = [i for i in kf.split(patients)][fold]
                paients_val = patients[ind_val]
                
                te_names = []
                for pp in paients_val:
                    te_names += [i for i in names if pp in i]
                    
                self.fold_name = f"BC{fold+1}"
                
            elif self.data == 'her2st':
                patients = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
                te_names = [i for i in names if patients[fold] in i]
            elif self.data == 'skin':
                patients = ['P2', 'P5', 'P9', 'P10']
                te_names = [i for i in names if patients[fold] in i]
                
            elif self.data == 'GBM_data':
                kf = KFold(5, shuffle=True, random_state=2021)
                patients = ['SNU16','SNU17','SNU18','SNU19',
                            'SNU21','SNU22','SNU23','SNU24',
                            'SNU25','SNU27','SNU33','SNU34',
                            'SNU38','SNU40','SNU43','SNU46','SNU51']
                patients = np.array(patients)
                _, ind_val = [i for i in kf.split(patients)][fold]
                paients_val = patients[ind_val]
                
                te_names = []
                for pp in paients_val:
                    te_names += [i for i in names if pp in i]
                
                self.fold_name = f"GBM{fold+1}"
                
            tr_names = list(set(names)-set(te_names))

            if self.mode == 'train':
                self.names = tr_names
            else:
                self.names = te_names
        
        if self.use_pyvips:
            self.img_dict = {i:self.get_img(i) for i in self.names}
            
            with open(f"{self.data_dir}/slide_shape.pickle", "rb") as f:
                self.img_shape_dict = pickle.load(f)
        else:
            self.img_dict = {i:np.array(self.get_img(i)) for i in self.names}
            
        self.meta_dict = {i:self.get_meta(i) for i in self.names}
        
        if mode not in  ["extraction", "inference"]:
            gene_list = list(np.load(self.data_dir + f'/genes_{self.data}.npy', allow_pickle=True))    
            self.exp_dict = {i:scp.transform.log(scp.normalize.library_size_normalize(m[gene_list])) for i,m in self.meta_dict.items()}
        
            # Smoothing data 
            self.exp_dict = {i:smooth_exp(m).values for i,m in self.exp_dict.items()}
        
        if (mode == "external_test") or (self.data == 'GBM_data'):
            self.center_dict = {i:np.floor(m[['pixel_y','pixel_x']].values).astype(int) for i,m in self.meta_dict.items()}
            self.loc_dict = {i:m[['x','y']].values for i,m in self.meta_dict.items()}
        else:
            self.center_dict = {i:np.floor(m[['pixel_x','pixel_y']].values).astype(int) for i,m in self.meta_dict.items()}
            self.loc_dict = {i:m[['y','x']].values for i,m in self.meta_dict.items()}

        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(self.names))

    def __getitem__(self, index):
        """Return one piece of data for training, and all data within a patient for testing.

        Returns:
            tuple: 
                patches (torch.Tensor): Target spot images
                exps (torch.Tensor): Gene expression of the target spot.
                pid (torch.LongTensor): patient index
                sid (torch.LongTensor): spot index
                wsi (torch.Tensor): Features extracted from all spots for the patient
                position (torch.LongTensor): Relative position of spots 
                neighbors (torch.Tensor): Features extracted from neighbor regions of the target spot.
                maks_tb (torch.Tensor): Masking table for neighbor features
        """
        if self.mode == 'train':
            i = 0
            while index>=self.cumlen[i]:
                i += 1
            idx = index
            if i > 0:
                idx = index - self.cumlen[i-1]
                
            name = self.id2name[i]
            
            im = self.img_dict[name]
            if self.use_pyvips:
                img_shape = self.img_shape_dict[name]
            else:
                img_shape = im.shape
                
            center = self.center_dict[name][idx]
            x, y = center
            
            mask_tb =  self.make_masking_table(x, y, img_shape)
                
            if self.use_pyvips:
                patches = im.extract_area(x,y,self.r*2,self.r*2).numpy()[:,:,:3]
            else:
                patches = im[y-self.r:y+self.r, x-self.r:x+self.r, :] 
                            
            if self.mode == "external_test":
                patches = self.test_transforms(patches)
            else:
                patches = self.train_transforms(patches)
            
            exps = self.exp_dict[name][idx]
            exps = torch.Tensor(exps)
            
            sid = torch.LongTensor([idx])
            
            #neighbors = torch.load(self.data_dir + f"/{self.neighbor_dir}/{name}.pt")[idx]
            neighbor_path = os.path.join(self.data_dir, self.neighbor_dir, f"{name}.h5")
            with h5py.File(neighbor_path, 'r') as f:
                neighbors = f['embeddings'][idx]
            neighbors = torch.Tensor(neighbors)

        else:
            i = index
            name = self.id2name[i]
            
            im = self.img_dict[name]    
            if self.use_pyvips:
                img_shape = self.img_shape_dict[name]
            elif isinstance(im, OpenSlide):
                width, height = im.level_dimensions[0]
                img_shape = (height, width, 3)
            else:
                img_shape = im.shape
                
            centers = self.center_dict[name]
            
            n_patches = len(centers)    
            if self.extract_mode == 'neighbor':
                patches = torch.zeros((n_patches,3,2*self.r*self.num_neighbors,2*self.r*self.num_neighbors))
            else:
                patches = torch.zeros((n_patches,3,2*self.r,2*self.r))
            mask_tb = torch.ones((n_patches, self.num_neighbors**2))
            for j in range(n_patches):
                center = centers[j]
                x, y = center
                
                mask_tb[j] = self.make_masking_table(x, y, img_shape)
                
                if self.extract_mode == 'neighbor':
                    k_ranges = [(self.r * 2 * k, self.r * 2 * (k + 1)) for k in range(self.num_neighbors)]
                    m_ranges = [(self.r * 2 * m, self.r * 2 * (m + 1)) for m in range(self.num_neighbors)]
                    patch = torch.zeros((3, 2 * self.r * self.num_neighbors, 2 * self.r * self.num_neighbors))
                    
                    if self.use_pyvips:
                        patch_unnorm = self.extract_patches_pyvips(im, x, y, img_shape)
                        
                        for k, (k_start, k_end) in enumerate(k_ranges):
                            for m, (m_start, m_end) in enumerate(m_ranges):
                                n = k * self.num_neighbors + m
                                if mask_tb[j, n] != 0:
                                    # Since patch_unnorm is not a tensor, assume it needs iterative access.
                                    patch_data = patch_unnorm[k_start:k_end, m_start:m_end, :]
                                    transformed_patch = self.test_transforms(patch_data)
                                    patch[:, k_start:k_end, m_start:m_end] = transformed_patch
                        
                    else:
                        y_start = y-self.r*self.num_neighbors
                        x_start = x-self.r*self.num_neighbors                        

                        # Loop over pre-calculated ranges
                        for k, (k_start, k_end) in enumerate(k_ranges):
                            for m, (m_start, m_end) in enumerate(m_ranges):
                                n = k * self.num_neighbors + m  # Calculate n based on loop indices
                                if mask_tb[j, n] != 0:
                                    # Extract and transform patch if mask is non-zero
                                    if isinstance(im, OpenSlide):
                                        patch_size = self.r*2
                                        x0 = int(x_start + m * patch_size)
                                        y0 = int(y_start + k * patch_size)
                                        tmp = im.read_region((x0, y0), 0, (patch_size, patch_size)).convert('RGB')
                                        tmp = np.array(tmp)
                                    else:
                                        tmp = im[y_start + k_start:y_start + k_end, x_start + m_start:x_start + m_end, :]
                                    patch[:, k_start:k_end, m_start:m_end] = self.test_transforms(tmp)
                        
                else:
                    if self.use_pyvips:
                        patch = im.extract_area(x,y,self.r*2,self.r*2).numpy()[:,:,:3]
                    elif isinstance(im, OpenSlide):
                        x0 = int(x - self.r)
                        y0 = int(y - self.r)
                        patch = im.read_region((x0, y0), 0, (self.r*2, self.r*2)).convert('RGB')
                        patch = np.array(patch)
                    else:
                        patch = im[y-self.r:y+self.r,x-self.r:x+self.r,:]
                        
                    patch = self.test_transforms(patch)
                
                patches[j] = patch

            if self.mode == "extraction":
                return patches
            
            if self.mode != "inference":
                exps = self.exp_dict[name]
                exps = torch.Tensor(exps)
            
            sid = torch.arange(n_patches)            
            #neighbors = torch.load(self.data_dir +  f"/{self.neighbor_dir}/{name}.pt")
            neighbor_path = os.path.join(self.data_dir, self.neighbor_dir, f"{name}.h5")
            with h5py.File(neighbor_path, 'r') as f:
                neighbors = f['embeddings'][idx]
            neighbors = torch.Tensor(neighbors)
        
        wsi = torch.load(self.data_dir +  f"/{self.gt_dir}/{name}.pt")
        
        pid = torch.LongTensor([i])
        pos = self.loc_dict[name]
        position = torch.LongTensor(pos)
        
        if self.mode not in ["external_test", "inference"]:
            name += f"+{self.data}"
            if self.fold_name:
                name += f"+{self.fold_name}"
        
        if self.mode == 'train':
            return patches, exps, pid, sid, wsi, position, neighbors, mask_tb
        elif self.mode == 'inference':
            return patches, sid, wsi, position, neighbors, mask_tb
        else:
            return patches, exps, sid, wsi, position, name, neighbors, mask_tb
        
    def __len__(self):
        if self.mode == 'train':
            return self.cumlen[-1]
        else:
            return len(self.meta_dict)
        
    def make_masking_table(self, x: int, y: int, img_shape: tuple):
        """Generate masking table for neighbor encoder.

        Args:
            x (int): x coordinate of target spot
            y (int): y coordinate of target spot
            img_shape (tuple): Shape of whole slide image

        Raises:
            Exception: if self.num_neighbors is bigger than 5, raise error.

        Returns:
            torch.Tensor: masking table
        """
        
        # Make masking table for neighbor encoding module
        mask_tb = torch.ones(self.num_neighbors**2)
        
        def create_mask(ind, mask_tb, window):
            if y-self.r*window < 0:
                mask_tb[self.num_neighbors*ind:self.num_neighbors*ind+self.num_neighbors] = 0 
            if y+self.r*window > img_shape[0]:
                mask_tb[(self.num_neighbors**2-self.num_neighbors*(ind+1)):(self.num_neighbors**2-self.num_neighbors*ind)] = 0 
            if x-self.r*window < 0:
                mask = [i+ind for i in range(self.num_neighbors**2) if i % self.num_neighbors == 0]
                mask_tb[mask] = 0 
            if x+self.r*window > img_shape[1]:
                mask = [i-ind for i in range(self.num_neighbors**2) if i % self.num_neighbors == (self.num_neighbors-1)]
                mask_tb[mask] = 0 
                
            return mask_tb
        
        ind = 0
        window = self.num_neighbors
        while window >= 3: 
            mask_tb = create_mask(ind, mask_tb, window)
            ind += 1 
            window -= 2   
         
        return mask_tb
    
    def extract_patches_pyvips(self, slide, x: int, y: int, img_shape: tuple):
        tile_size = self.r*2
        expansion_size = tile_size * self.num_neighbors
        padding_color = 255

        x_lt = x - tile_size * 2
        y_lt = y - tile_size * 2
        x_rd = x + tile_size * 3
        y_rd = y + tile_size * 3

        # Determine if padding is needed and calculate padding amounts
        x_left_pad = max(0, -x_lt)
        x_right_pad = max(0, x_rd - img_shape[1])
        y_up_pad = max(0, -y_lt)
        y_down_pad = max(0, y_rd - img_shape[0])

        # Adjust coordinates and dimensions
        x_lt = max(x_lt, 0)
        y_lt = max(y_lt, 0)
        width = min(x_rd, img_shape[1]) - x_lt
        height = min(y_rd, img_shape[0]) - y_lt

        # Extract and convert image
        im = slide.extract_area(x_lt, y_lt, width, height)
        im = np.array(im)[:, :, :3]

        # Check if any padding is necessary
        if x_left_pad or x_right_pad or y_up_pad or y_down_pad:
            # Create a full image with padding where necessary
            padded_image = np.full((expansion_size, expansion_size, 3), padding_color, dtype='uint8')

            # Calculate the placement indices for the image within the padded array
            start_x = x_left_pad
            end_x = x_left_pad + width
            start_y = y_up_pad
            end_y = y_up_pad + height

            # Place the image within the padded area
            padded_image[start_y:end_y, start_x:end_x] = im
            image = padded_image
        else:
            image = im

        return image
