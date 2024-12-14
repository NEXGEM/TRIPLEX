
from base_dataset import STDataset

class StnetDataset(STDataset):
    def __init__(self, **kwargs):
        super(StnetDataset, self).__init__(**kwargs)