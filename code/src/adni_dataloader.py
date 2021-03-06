import os
import numpy as np
import pandas as pd
import torch
import pickle

from pathlib import Path
from random import seed, shuffle
from torch.utils.data import Dataset
import pdb

### DIMENSION OF 3D VOLUME
dX = 91
dY = 109
dZ = 91

local_datapath = './data/adni_data/'

class ADNIGattrDataset(Dataset):
    def __init__(self, datapath=local_datapath, ad_cn=False, gattr_name = 'age', debug=False,  split_filename=None,
                 return_ids=False, train=True):
        super(ADNIGattrDataset, self).__init__()
        assert(split_filename is not None)

        self.debug = debug
        self.return_ids = return_ids
        self.datapath = datapath
        self.metapath = 'data/ADNI_data_all_info.csv'
        if not Path(self.datapath).exists() or not Path(self.metapath).exists() or not Path(split_filename).exists():
            print('Paths specified do not exist.')
            return

        metadata = pd.read_csv(self.metapath, sep=',')
        split_ids = pickle.load(open(split_filename, 'rb'))

        self.split_ids = split_ids
        metadata = metadata[metadata['Subject ID'].isin(split_ids)]

        # loads all image data into memory!!!
        print('Loading all ADNI images into RAM...', len(metadata))

        X = self.load_X_data(metadata["Subject ID"].values + '.npy').astype(np.float32)
        X_mean = np.mean(X)
        X_std = np.std(X)
        X = (X - X_mean) / X_std
        print('Done.\n')

        x_control = metadata["Imaging Protocol"].values
        protocols = ['Manufacturer=GE MEDICAL SYSTEMS', 'Manufacturer=Philips Medical Systems', 
                     'Manufacturer=SIEMENS']
        for i in range(len(protocols)):
            x_control[x_control == protocols[i]] = i
        x_control = x_control.astype(np.long)
        
        self.gattr_name = gattr_name
        if self.gattr_name == 'age':
            gattrs = metadata['Age'].values
            print('GATTRS MAX VALUE', gattrs.max())
            gattrs = gattrs / gattrs.max()
            gattrs = gattrs.astype(np.float)
        elif self.gattr_name == 'gender':
            gattrs = metadata['Sex'].values
            gattrs[gattrs == 'F'] = 0.
            gattrs[gattrs == 'M'] = 1.
            gattrs = gattrs.astype(np.long)
        else:
            raise NotImplementedError
        
        self.ad_cn = ad_cn
        if self.ad_cn:
            y = metadata['Research Group'].values
            index = (y == 'AD') | (y == 'CN')
            
            y = y[index]
            y[y == 'AD'] = 1.
            y[y == 'CN'] = 0.
            y = y.astype(np.float32)
            X = X[index]
            x_control = x_control[index]
            gattrs = gattrs[index]
        else:
            y = metadata["MMSE Total Score"].values.astype(np.float32)


        print('Numer of samples', len(X))
        self.x = X
        self.y = y
        self.control = x_control
        self.gattrs = gattrs
        
    def load_X_data(self, fnames):
        dat = np.empty((len(fnames), dX, dY, dZ), dtype=np.float32)
        for f, i in zip(fnames, range(0,len(fnames))):
            if self.debug and i > 10:
                print('Using the Debug Option!')
                break
            tmp = np.load(self.datapath+f)
            dat[i,:,:,:] = tmp
        return dat
    
    def __getitem__(self, idx):
        x_val = self.x[idx]
        y_val = self.y[idx]
        c_val = self.control[idx]
        g_val = self.gattrs[idx]

        x_shape = x_val.shape
        if not self.return_ids:
            return torch.Tensor(x_val).view(1, *x_shape), torch.LongTensor([y_val]).squeeze(), \
                   torch.LongTensor([c_val]).squeeze(), torch.FloatTensor([g_val]).squeeze()
        else:
            return self.split_ids[idx], torch.Tensor(x_val).view(1, *x_shape), torch.LongTensor([y_val]).squeeze(), \
                   torch.LongTensor([c_val]).squeeze(), torch.FloatTensor([g_val]).squeeze()
    
    def __len__(self):
        return len(self.x)
    

def create_train_valid_splits():
    # We creat some random train and validation splits
    for idx in range(0, 5):
        metapath = 'ADNI_data_all_info.csv'
        trainsplit = 0.8
        metadata = pd.read_csv(metapath, sep=',')
        ids = metadata['Subject ID'].values
        perm = list(range(len(ids)))
        shuffle(perm)
        ids = ids[perm]
        split_point = int(round(len(ids) * trainsplit))
        train_ids = ids[:split_point]
        val_ids = ids[split_point:]
        train_name = 'train_{}.data'.format(idx)
        train_file = open(os.path.join('./data/splits', train_name), 'wb')
        pickle.dump(train_ids, train_file)
        test_name = 'val_{}.data'.format(idx)
        test_file = open(os.path.join('./data/splits', test_name), 'wb')
        pickle.dump(val_ids, test_file)
        print(len(train_ids), len(val_ids))
