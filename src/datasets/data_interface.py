import inspect 
import importlib 
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms


class DataInterface(pl.LightningDataModule):

    def __init__(self, 
                dataset_name=None, 
                **kwargs):
        """[summary]

        Args:
            batch_size (int, optional): [description]. Defaults to 64.
            num_workers (int, optional): [description]. Defaults to 8.
            dataset_name (str, optional): [description]. Defaults to ''.
        """        
        super().__init__()

        self.dataset_name = dataset_name
        self.kwargs = kwargs
        self.train_config = kwargs['train_dataloader']
        self.test_config = kwargs['test_dataloader']
        self.load_data_module()

    def prepare_data(self):
        # 1. how to download
        # MNIST(self.data_dir, train=True, download=True)
        # MNIST(self.data_dir, train=False, download=True)
        ...

    def setup(self, stage=None):
        # 2. how to split, argument
        """  
        - count number of classes

        - build vocabulary

        - perform train/val/test splits

        - apply transforms (defined explicitly in your datamodule or assigned in init)
        """
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_dataset = self.instancialize(phase='train')
            self.val_dataset = self.instancialize(phase='val')

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            # self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
            self.test_dataset = self.instancialize(phase='test')


    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                        batch_size=self.train_config['batch_size'], 
                        num_workers=self.train_config['num_workers'], 
                        pin_memory=self.train_config['pin_memory'],
                        shuffle=self.train_config['shuffle'])

    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                        batch_size=self.test_config['batch_size'], 
                        num_workers=self.test_config['num_workers'], 
                        pin_memory=self.test_config['pin_memory'],
                        shuffle=self.test_config['shuffle'])

    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                        batch_size=self.test_config['batch_size'], 
                        num_workers=self.test_config['num_workers'], 
                        pin_memory=self.test_config['pin_memory'],
                        shuffle=self.test_config['shuffle'])


    def load_data_module(self):
        
        camel_name =  ''.join([i.capitalize() for i in (self.dataset_name).split('_')])
        try:
            self.data_module = getattr(importlib.import_module(
                f'datasets.{self.dataset_name}'), camel_name)
        except:
            raise ValueError(
                'Invalid Dataset File Name or Invalid Class Name!')
    
    def instancialize(self, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.kwargs.
        """
        class_args = inspect.getargspec(self.data_module.__init__).args[1:]
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(other_args)
        return self.data_module(**args1)