from torch.utils.data import Dataset
import torch
import glob
import os
import numpy as np
def normalization(data,clip_s=3.2):
    data = data.float()
    data = torch.clip(data,min=-clip_s,max=clip_s)
    data = (data-torch.min(data))/(torch.max(data)-torch.min(data))
    return data

class MyDataset(Dataset):
    def __init__(self,label_path):
        super(MyDataset, self).__init__()
        self.label_paths = glob.glob(os.path.join(label_path,'*.npy'))

    def __len__(self):
        return len(self.label_paths)

    def __getitem__(self, index):
        data = np.load(self.label_paths[index])
        data = torch.from_numpy(data)
        data = normalization(data)
        data = torch.unsqueeze(data,0)
        return data,self.label_paths[index].split('/')[-1]

