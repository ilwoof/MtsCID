import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler


class SWaTSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/train.csv', header=0, decimal=',')
        data = data.values[:, 1:-1]

        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)

        test_data = pd.read_csv(data_path + '/test.csv', delimiter=';', decimal=',')

        y = test_data['Normal/Attack'].to_numpy()
        labels = []
        for i in y:
            if i == 'Attack':
                labels.append(1)
            else:
                labels.append(0)
        labels = np.array(labels)

        test_data = test_data.values[:, 1:-1]
        test_data = np.nan_to_num(test_data)

        self.test = self.scaler.transform(test_data)
        self.train = data
        self.test_labels = labels.reshape(-1, 1)

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        mode : "train" or "test"
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.train.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(np.zeros(self.win_size))
        elif self.mode == 'test':
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])


class PSMSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/train.csv')
        data = data.values[:, 1:]

        data = np.nan_to_num(data)

        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/test.csv')

        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)

        self.test = self.scaler.transform(test_data)

        self.train = data

        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        mode : "train" or "test"
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.train.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(np.zeros(self.win_size))
        elif self.mode == 'test':
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])

class MSLSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/MSL_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/MSL_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.test_labels = np.load(data_path + "/MSL_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.train.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(np.zeros(self.win_size))
        elif self.mode == 'test':
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])

class SMAPSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMAP_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMAP_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.test_labels = np.load(data_path + "/SMAP_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.train.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(np.zeros(self.win_size))
        elif self.mode == 'test':
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])

class SMDSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMD_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMD_test.npy")
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.test_labels = np.load(data_path + "/SMD_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)
        
    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.train.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(np.zeros(self.win_size))
        elif self.mode == 'test':
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])

class WADISegLoader(Dataset):
    def __init__(self, root_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        # train data does not have label column
        train_data = pd.read_csv(os.path.join(root_path, 'WADI_14days_new.csv'))
        test_data = pd.read_csv(os.path.join(root_path, 'WADI_attackdataLABLE.csv'), header=1)

        # Remove the blank space in the columns
        train_data.columns = [col.strip(' ') for col in train_data.columns]
        test_data.columns = [col.strip(' ') for col in test_data.columns]

        labels = test_data["Attack LABLE (1:No Attack, -1:Attack)"].apply(lambda x: 0 if x == 1 else 1).values
        test_data = test_data.drop(['Attack LABLE (1:No Attack, -1:Attack)'], axis=1)

        # Removing columns that only contain time and nans (data missing from the csv file)
        train_nan_columns = {col for col in train_data.columns if train_data[col].isna().all()}
        test_nan_columns = {col for col in test_data.columns if test_data[col].isna().all()}
        common_nan_columns = list(train_nan_columns.intersection(test_nan_columns))

        train_data = train_data.drop(['Row', 'Date', 'Time'] + common_nan_columns, axis=1)
        test_data = test_data.drop(['Row', 'Date', 'Time'] + common_nan_columns, axis=1)

        # fill the missing values
        train_data = train_data.interpolate().bfill()
        test_data = test_data.interpolate().bfill()

        train_data = train_data.values
        test_data = test_data.values

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)

        data_len = len(train_data)
        self.train = train_data[:int(data_len * 0.8)]
        self.val = train_data[int(data_len * 0.8):]
        self.test = test_data
        self.test_labels = labels
        print("train:", self.train.shape)
        print("test:", self.test.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'val':
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(np.zeros(self.win_size))
        elif self.mode == 'val':
            return np.float32(self.val[index:index + self.win_size]), np.float32(np.zeros(self.win_size))
        elif self.mode == 'test':
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class NIPS_TS_WaterSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/NIPS_TS_Water_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/NIPS_TS_Water_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/NIPS_TS_Water_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'val':
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(np.zeros(self.win_size))
        elif self.mode == 'val':
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'test':
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class NIPS_TS_SwanSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/NIPS_TS_Swan_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/NIPS_TS_Swan_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/NIPS_TS_Swan_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'val':
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(np.zeros(self.win_size))
        elif self.mode == 'val':
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'test':
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class NIPS_TS_CCardSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/NIPS_TS_Creditcard_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/NIPS_TS_Creditcard_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/NIPS_TS_Creditcard_test_label.npy")

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'val':
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(np.zeros(self.win_size))
        elif self.mode == 'val':
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'test':
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode='train', dataset='KDD', val_ratio=0.2):
    """
    model : 'train' or 'test'
    """

    if dataset == 'SMD':
        dataset = SMDSegLoader(data_path, win_size, step, mode)
    elif dataset == 'MSL':
        dataset = MSLSegLoader(data_path, win_size, step, mode)
    elif dataset == 'SMAP':
        dataset = SMAPSegLoader(data_path, win_size, step, mode)
    elif dataset == 'PSM':
        dataset = PSMSegLoader(data_path, win_size, step, mode)
    elif dataset == 'SWaT':
        dataset = SWaTSegLoader(data_path, win_size, step, mode)
    elif dataset == 'WADI':
        dataset = WADISegLoader(data_path, win_size, step, mode)
    elif dataset == 'NIPS_TS_Water':
        dataset = NIPS_TS_WaterSegLoader(data_path, win_size, step, mode)
    elif dataset == 'NIPS_TS_Swan':
        dataset = NIPS_TS_SwanSegLoader(data_path, win_size, step, mode)
    elif dataset == 'NIPS_TS_Creditcard':
        dataset = NIPS_TS_CCardSegLoader(data_path, win_size, step, mode)

    shuffle = False

    if mode == 'train':
        shuffle = True

        dataset_len = int(len(dataset))
        train_use_len = int(dataset_len * (1 - val_ratio))

        val_use_len = int(dataset_len * val_ratio)
        val_start_index = random.randrange(train_use_len)

        indices = torch.arange(dataset_len)
        
        train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index+val_use_len:]])
        train_subset = Subset(dataset, train_sub_indices)

        val_sub_indices = indices[val_start_index:val_start_index+val_use_len]
        val_subset = Subset(dataset, val_sub_indices)
        
        train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
        val_loader = DataLoader(dataset=val_subset, batch_size=batch_size, shuffle=shuffle)

        # k_use_len = int(train_use_len*0.1)
        # k_sub_indices = indices[:k_use_len]
        # k_subset = Subset(dataset, k_sub_indices)
        # k_loader = DataLoader(dataset=k_subset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
        #
        # return train_loader, val_loader, k_loader

        return train_loader, val_loader

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader
