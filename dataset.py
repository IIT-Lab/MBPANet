# padataset.py :power allocatio dataset

import os
import scipy.io as sio
from torch.utils.data import Dataset,DataLoader

__all__ = ["PADataset"]

class PADataset(Dataset):

    def __init__(self,root,mode,scenario):
        super(PADataset, self).__init__()
        assert os.path.isdir(root)
        assert mode in {"train","vaild","test"}
        assert scenario in {5, 10, 15, 20, 25}
        dir_data = os.path.join(root, f"{mode}_{scenario}.mat")
        
        data = sio.loadmat(dir_data) 
        self.K = scenario
        self.csis = data['csi']
        self.powers = data['power']
    
    def __len__(self):
        
        return len(self.csis)
    
    def __getitem__(self,idx):
        csi = self.csis[idx]
        power = self.powers[idx]

        return (csi,power)