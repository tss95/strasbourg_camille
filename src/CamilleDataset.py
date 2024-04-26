import torch
from torch.utils.data import Dataset
import os
import h5py
from sklearn.model_selection import train_test_split
import numpy as np
from time import time


class CamillePretrainDataset(Dataset):
    def __init__(self, data_path, dataset_type, eb_transform=None, psd_transform=None, spec_transform=None, stalta_transform=None, image_size=(256, 256), seed=42):
        self.data_path = data_path
        self.eb_transform = eb_transform
        self.psd_transform = psd_transform
        self.spec_transform = spec_transform
        self.stalta_transform = stalta_transform
        self.dataset_type = dataset_type
        self.image_size = image_size
        self.seed = seed
        self.paths = os.listdir(data_path)
        self.paths = self.__split_paths()
        self.index_list, self.path_list = self.__create_index_dict()

    def __split_paths(self):
        train_paths, test_paths = train_test_split(self.paths, test_size=0.2, random_state=self.seed)
        train_paths, val_paths = train_test_split(train_paths, test_size=0.25, random_state=self.seed)  # 0.25 x 0.8 = 0.2
        if self.dataset_type == 'train':
            return train_paths
        elif self.dataset_type == 'val':
            return val_paths
        elif self.dataset_type == 'test':
            return test_paths

    def __create_index_dict(self):
        index_array = [0]  # Start with 0 for the very beginning
        path_list = []
        total_entries = 0
        for path in self.paths:
            f = h5py.File(os.path.join(self.data_path, path), 'r')
            entries_in_file = f['eb_data'].shape[0]
            total_entries += entries_in_file
            index_array.append(total_entries)  # End index for current file
            path_list.append(path)
            f.close()
        return np.array(index_array), path_list

    def get_file(self, index):
        if index < 0 or index >= self.index_list[-1]:
            raise ValueError(f"Index {index} out of bounds")

        file_index = np.searchsorted(self.index_list, index, side='right') - 1
        file_path = self.path_list[file_index]
        start_index = self.index_list[file_index]

        return file_path, start_index
    
    def __len__(self):
        return int(self.index_list[-1])

    
    def __getitem__(self, idx):
        path, start_index = self.get_file(idx)
        converted_index = idx - start_index

        # Debugging output
        #print(f"Global index (idx): {idx}")
        #print(f"Path: {path}")
        #print(f"Start index of file: {start_index}")
        #print(f"Local index in file: {converted_index}")
        f = h5py.File(os.path.join(self.data_path, path), 'r')
        eb = f['eb_data'][converted_index]
        psd = f['psd_data'][converted_index]
        spec = f['spec_data'][converted_index]
        stalta = f['stalta_data'][converted_index]
        
        if self.eb_transform:
            eb1 = torch.tensor(self.eb_transform(eb), dtype=torch.float32)
            eb2 = torch.tensor(self.eb_transform(eb), dtype=torch.float32)
        if self.psd_transform:
            psd1 = torch.tensor(self.psd_transform(psd), dtype=torch.float32)
            psd2 = torch.tensor(self.psd_transform(psd), dtype=torch.float32)
        if self.spec_transform:
            spec1 = torch.tensor(self.spec_transform(spec), dtype=torch.float32)
            spec2 = torch.tensor(self.spec_transform(spec), dtype=torch.float32)
        if self.stalta_transform:
            stalta1 = torch.tensor(self.stalta_transform(stalta), dtype=torch.float32)
            stalta2 = torch.tensor(self.stalta_transform(stalta), dtype=torch.float32)
    
        # Should do:
        # Noise Aug, Normalization per sub sample
            
        f.close()
        return {"eb1": eb1, "eb2": eb2, "psd1": psd1, "psd2": psd2, "spec1": spec1, "spec2": spec2, "stalta1": stalta1, "stalta2": stalta2}
        
        
        
        
   
    
                    