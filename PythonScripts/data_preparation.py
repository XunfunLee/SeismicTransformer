"""
Contains functionality for process the ground motion data.
1. Load data from matlab(.mat format)
2. Process data from .mat file to numpy array
3. Put data into dataloader, also from numpy array to tensor
4. Create a custom dataset which can merge the training step and 

Author: Jason Jiang (Xunfun Lee)
Date: 2023.11.30
"""

import os
import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
from torch.utils.data.dataset import ConcatDataset
from typing import List, Tuple

# 1. Load data from matlab(.mat format)
def LoadData(
    time_series: str,
    file_name: str) -> str:
    """Load data from \data folder. Load data from .mat file from matlab format.

    Args:
    time_series: sample rate & times (e.g. 50Hz_60s, 50Hz_90s)
    file_name: xxx.mat

    Returns:
    Three directory for train, validation, test data.

    """
    # train, validation, test data folder (os.path.join can fit to both windows and macos)
    # using relative directory
    traindata_path = os.path.join("Data", time_series, "train", file_name)
    valdata_path = os.path.join("Data", time_series, "validation", file_name)
    testdata_path = os.path.join("Data", time_series, "test", file_name)

    # Check if the path is exist
    assert os.path.isdir(os.path.dirname(traindata_path)), "Train data directory doesn't exist. Please Check."
    assert os.path.isdir(os.path.dirname(valdata_path)), "Validation data directory doesn't exist. Please Check."
    assert os.path.isdir(os.path.dirname(testdata_path)), "Test data directory doesn't exist. Please Check."

    # Check if the file is exist
    if os.path.exists(traindata_path) == False:
        print("Train data file doesn't exist. Please Check.")
    else:
        print(f"traindata_path:", traindata_path)   

    if os.path.exists(traindata_path) == False:
        print("Validation data file doesn't exist. Please Check.")
    else:
        print(f"valdata_path:", valdata_path)   

    if os.path.exists(traindata_path) == False:
        print("Test data file doesn't exist. Please Check.")
    else:
        print(f"testdata_path:", testdata_path)

    return traindata_path, valdata_path, testdata_path

# 2. Process data from .mat file to numpy array
    # using numpy, h5py to process the data from matlab.
        # numpy: transpose the array.
        # h5py: to read matlab file.
        # matlab data comes from @Jie Zheng's job, thanks for your job :)
def H5pyToTensor(
    traindata_path: str,
    valdata_path: str,
    testdata_path: str,
    transpose:bool=True) -> np.ndarray:
    """Process .mat data. Read file(h5py) and transpose(numpy)

    Args:
    traindata_path: train data directory
    valdata_path: validation data directory
    testdata_path: test data directory
    transpose: the data framework in matlab, @Jie Zheng's job need to transpose

    Returns:
    Six array.
    Three for time-series data (gm_recs)
    Three for lable data (labels)
    """
    # train data: will influence the weights of the model
    train_gm_recs = np.array(h5py.File(traindata_path, 'r')['gms'])
    train_labels = np.array(h5py.File(traindata_path, 'r')['labels'])

    # validation data: just calculate the lost function, won't influence the backpropogation process
    val_gm_recs = np.array(h5py.File(valdata_path, 'r')['gms'])
    val_labels = np.array(h5py.File(valdata_path, 'r')['labels'])

    # test data: using to test the model after the model is trained completely
    test_gm_recs = np.array(h5py.File(testdata_path, 'r')['gms'])
    test_labels = np.array(h5py.File(testdata_path, 'r')['labels'])

    if transpose == True:
        train_gm_recs = np.transpose(train_gm_recs)
        train_labels = np.transpose(train_labels)

        val_gm_recs = np.transpose(val_gm_recs)
        val_labels = np.transpose(val_labels)

        test_gm_recs = np.transpose(test_gm_recs)
        test_labels = np.transpose(test_labels)

    train_gm_recs = torch.from_numpy(train_gm_recs)
    train_labels = torch.from_numpy(train_labels)
    val_gm_recs = torch.from_numpy(val_gm_recs)
    val_labels = torch.from_numpy(val_labels)
    test_gm_recs = torch.from_numpy(test_gm_recs)
    test_labels = torch.from_numpy(test_labels)

    # print(f"--------------------------------")
    # print(f"Length of train dataset({len(train_gm_recs)}) --> label({len(train_labels)})")
    # print(f"Length of validation dataset({len(val_gm_recs)}) --> label({len(val_labels)})")
    # print(f"Length of test dataset({len(test_gm_recs)}) --> label({len(test_labels)})")
    # print(f"--------------------------------")
    # print(f"Shape of train dataset({train_gm_recs.shape}) = {len(train_gm_recs)} GMs --> label({train_labels.shape})")
    # print(f"Shape of validation dataset({val_gm_recs.shape}) = {len(val_gm_recs)} GMs --> label({val_labels.shape})")
    # print(f"Shape of test dataset({test_gm_recs.shape}) = {len(test_gm_recs)} GMs --> label({test_labels.shape})")
    # print(f"--------------------------------")

    return train_gm_recs, train_labels, val_gm_recs, val_labels, test_gm_recs, test_labels

# 3. Put data into dataloader, also from numpy array to tensor
def CreateDataLoader(input_data:torch.Tensor, 
                     labels:torch.Tensor, 
                     batch_size:int) -> DataLoader:
    """Create a dataloader to manage the data

    Args:
    input_data: data in torch.Tensor type, usually a ground motion time seires curve in Seismic Transformer
    labels: lable of the damage state of the structure under input_data ground motion
    batch_size: the number of samples load to model one time

    Returns:
    torch.utils.data.DataLoader(class): can shuffle the data, load data in loop
    """
    dataset = TensorDataset(input_data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# 4. Create a custom dataset to split the training and validation dataset
class CustomTensorDataset(Dataset):
    """Custom dataset"""
    def __init__(self, tensors):
        self.tensors = tensors

    def __len__(self):
        return len(self.tensors[0])

    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]
        return x, y

def CreateDataLoadersWithMultiDataset(data_list:List[torch.Tensor],
                                    label_list:List[torch.Tensor], 
                                    train_ratio:float=0.8, 
                                    batch_size:int=64) -> Tuple[DataLoader, DataLoader]:
    """ Merge multi dataset and label and split them into train and validation set

    Args:
        data_list: [n_samples, length_of_gm, 1]   e.g. [60940, 3000, 1]
        label_list: [n_samples]
        train_ratio: ratio of training data of the whole dataset
        batch_size: batch size

    return:
        train_dataloader, validation_dataloader
    """
    # merge all dataset in dataset list
    all_data = torch.cat(data_list, dim=0)
    all_labels = torch.cat(label_list, dim=0)

    # create CustomTensorDataset instance
    combined_dataset = CustomTensorDataset((all_data, all_labels))

    # calculate the size of training dataset
    train_size = int(train_ratio * len(combined_dataset))
    validation_size = len(combined_dataset) - train_size

    # random split the data
    train_dataset, validation_dataset = random_split(combined_dataset, [train_size, validation_size])
    print(f"--------------------------------")
    print(f"Length of train dataset({len(train_dataset)})")
    print(f"Length of validation dataset({len(validation_dataset)})")
    print(f"--------------------------------")

    # create dataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)



    return train_dataloader, validation_dataloader






