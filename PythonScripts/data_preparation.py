"""
Contains functionality for process the ground motion data.
1. LoadData(): data from matlab(.mat format)
2. H5pyToTensor(): Process data from .mat file to numpy array
3. MaskingDataFake(): Generate fake mask data which is not masked
4. TransformMaskData(): Generta mask data from the ground motion data
5. CreateDataLoader(): Put data into dataloader, also from numpy array to tensor
6. Class DataLabelMaskDataset(): Create a custom dataset: store the ground motion data, label, and mask
7. CreateDataLoadersWithMultiDataset(): Merge multi dataset and label and split them into train and validation set
8. FastFourierTransform(): FFT the ground motion data
9. CreateKeyPaddingMask_AllFalse(): Key padding mask for SeT-3 (all False mask which stands for no mask at all)
10. CreateLookAheadMask(): Create look ahead mask for the decoder in SeT-3

Author: Jason Jiang (Xunfun Lee)
Date: 2023.11.30
"""

import os
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
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
    transpose:bool=True) -> torch.Tensor:
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

# 3
def MaskingDataFake(input_array: np.ndarray) -> np.ndarray:
    """ Masking Fake data for comparation, this function is not used in the model
        just want to compare the performance of the model with and without masking

    Args:
    input_array: numpy array
    
    Returns:
    masked numpy array, e.g. [-1e9, -1e9, -1e9, -1e9, 0, 0, 0, 0, 0, 0, 0, 0, 0] size=(12,1)
    """
    mask = np.zeros_like(input_array)
    return mask

def MaskingData(input_array: np.ndarray,
                min_length:int=250,
                factor:int=50) -> np.ndarray:
    """Mask the data which is padding by 0 to match the exact size of the input of the model.

    We are setting a series of rule to decide whether a point is padding points, not for 0 padding only.
    If there are more than 250 constant points which values are less than 1/50 of the peak acceleration of the curve, we assume it is padding.
    Further more, padding won't exist in the middle of the ground motion, it must to be in the start or the end of the ground motion record.

    Args:
    input_array: numpy array
    min_length: the minimum length of the sequence
    factor: the factor of the threshold value
    
    Returns:
    masked numpy array, e.g. [-1e9, -1e9, -1e9, -1e9, 0, 0, 0, 0, 0, 0, 0, 0, 0] size=(12,1)
    """
    mask = np.zeros_like(input_array)
    _, sample_length = input_array.shape

    for i, row in enumerate(input_array):
        # Caculate the max value of the current sequence as the threshold value
        threshold_value = np.max(np.abs(row)) / factor

        # Check the start of the sequence
        start_count = 0
        while start_count < sample_length and abs(row[start_count]) < threshold_value:
            start_count += 1

        # if the start of the sequence is long enough, mask it
        if start_count >= min_length:
            mask[i, :start_count] = -1e9

        # Check the end of the sequence
        end_count = 0
        while end_count < sample_length and abs(row[sample_length - end_count - 1]) < threshold_value:
            end_count += 1

        # if the end of the sequence is long enough, mask it
        if end_count >= min_length:
            mask[i, sample_length - end_count:] = -1e9

    return mask

# 4
def TransformMaskDataV1(mask_data: np.ndarray,
                      num_of_patch:int=12,
                      patch_size:int=250) -> torch.Tensor:

    """ Split the mask like the gm to patches and add class token           [num_of_mask, 3000] --> [num_of_mask, 13]
        
        e.g. 3000 // 12 = 250, length of each mask is 250, but not all patches is padding.
        So we are using some math to count the padding point inside each patch,
        if the number of padding point is greater than useful points, then this patch is considered to be padding.

        Turning padding patch --> True, useful patch --> False, for the purpose of the `attn_weights` in Multi-head Attention Block

    Args:
        origin_mask: the mask data list             e.g. [60940, 3000]
        num_of_patch: the number of patch
        patch_size: the size of the patch

    Returns:

    """
    # Ensure mask_data length is 3000
    if mask_data.shape[1] != num_of_patch * patch_size:
        raise ValueError(f"Expected each mask data row to be of length {num_of_patch * patch_size},"
                         f" but got {mask_data.shape[1]}")

    # Define the chunk size as the total length divided by 12
    segment_size = patch_size

    # Initialize the results list
    results = []

    # Process each mask_data entry (each row in mask_data)
    for mask in mask_data:
        # 初始化一个布尔数组，用于存放每个分段的结果
        segment_results = np.zeros(num_of_patch, dtype=bool)

        # Split mask data into num_of_patch chunks and process each chunk
        for i in range(num_of_patch):
            start_idx = i * segment_size
            end_idx = start_idx + segment_size
            segment = mask[start_idx:end_idx]

            # 计算0和-1e9的数量
            num_zeros = np.sum(segment == 0)
            num_neg_inf = np.sum(segment == -1e9)

            # 根据数量多少设置结果
            segment_results[i] = num_zeros < num_neg_inf

        # Convert the row_result list to a PyTorch tensor and append to results
        attn_weights_mask_1D = torch.tensor(segment_results, dtype=torch.bool)
        # Add class token mask and attn_weigths_mask to match the input length of sequence
        # If we adding Frecrency and Structural propetries in front of the class token, we need to add more dimension here for padding.
        class_token_mask = torch.zeros(1, dtype=torch.bool)

        attn_weights_mask_with_classtoken_1D = torch.cat((class_token_mask, attn_weights_mask_1D), dim=0)
        
        results.append(attn_weights_mask_with_classtoken_1D)

    # here our output is 1D mask while the model need 2D input, we scale the dimension inside the multi-head attention forward function to save the memory
    return torch.stack(results)

# 4 adding frequency mask
def TransformMaskDataV2(mask_data: np.ndarray,
                      num_of_patch:int=12,
                      patch_size:int=250) -> torch.Tensor:

    """ Split the mask like the gm to patches and add class token           [num_of_mask, 3000] --> [num_of_mask, 13]
        
        e.g. 3000 // 12 = 250, length of each mask is 250, but not all patches is padding.
        So we are using some math to count the padding point inside each patch,
        if the number of padding point is greater than useful points, then this patch is considered to be padding.

        Turning padding patch --> True, useful patch --> False, for the purpose of the `attn_weights` in Multi-head Attention Block

    Args:
        origin_mask: the mask data list             e.g. [60940, 3000]
        num_of_patch: the number of patch
        patch_size: the size of the patch

    Returns:

    """
    # Ensure mask_data length is 3000
    if mask_data.shape[1] != num_of_patch * patch_size:
        raise ValueError(f"Expected each mask data row to be of length {num_of_patch * patch_size},"
                         f" but got {mask_data.shape[1]}")

    # Define the chunk size as the total length divided by 12
    segment_size = patch_size

    # Initialize the results list
    results = []

    # Process each mask_data entry (each row in mask_data)
    for mask in mask_data:
        # 初始化一个布尔数组，用于存放每个分段的结果
        segment_results = np.zeros(num_of_patch, dtype=bool)

        # Split mask data into num_of_patch chunks and process each chunk
        for i in range(num_of_patch):
            start_idx = i * segment_size
            end_idx = start_idx + segment_size
            segment = mask[start_idx:end_idx]

            # 计算0和-1e9的数量
            num_zeros = np.sum(segment == 0)
            num_neg_inf = np.sum(segment == -1e9)

            # 根据数量多少设置结果
            segment_results[i] = num_zeros < num_neg_inf

        # Convert the row_result list to a PyTorch tensor and append to results
        attn_weights_mask_1D = torch.tensor(segment_results, dtype=torch.bool)
        # Add class token mask and attn_weigths_mask to match the input length of sequence
        # If we adding Frecrency and Structural propetries in front of the class token, we need to add more dimension here for padding.
        class_token_mask = torch.zeros(1, dtype=torch.bool)

        # add frequency with token mask
        frequency_token_mask = torch.zeros(1, dtype=torch.bool)
        attn_weights_mask_with_classtoken_1D = torch.cat((class_token_mask, attn_weights_mask_1D, frequency_token_mask), dim=0)
        
        results.append(attn_weights_mask_with_classtoken_1D)

    # here our output is 1D mask while the model need 2D input, we scale the dimension inside the multi-head attention forward function to save the memory
    return torch.stack(results)


# 5 Put data into dataloader, also from numpy array to tensor
def CreateDataLoaderV1(input_data:torch.Tensor, 
                     labels:torch.Tensor,
                     mask:torch.Tensor,
                     batch_size:int) -> DataLoader:
    """Create a dataloader to manage the data

    Args:
    input_data: data in torch.Tensor type, usually a ground motion time seires curve in Seismic Transformer
    labels: lable of the damage state of the structure under input_data ground motion
    batch_size: the number of samples load to model one time

    Returns:
    torch.utils.data.DataLoader(class): can shuffle the data, load data in loop
    """
    dataset = DataLabelMaskDataset(input_data, labels, mask)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# 5.1 data, label, mask, frequency
def CreateDataLoaderV2(input_data:torch.Tensor, 
                     labels:torch.Tensor,
                     mask:torch.Tensor,
                     frequency:torch.Tensor,
                     batch_size:int) -> DataLoader:
    """Create a dataloader to manage the data

    Args:
    input_data: data in torch.Tensor type, usually a ground motion time seires curve in Seismic Transformer
    labels: lable of the damage state of the structure under input_data ground motion
    batch_size: the number of samples load to model one time

    Returns:
    torch.utils.data.DataLoader(class): can shuffle the data, load data in loop
    """
    dataset = DataLabelMaskFreqDataset(input_data, labels, mask, frequency)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# 6 Create a custom dataset: store the ground motion data, label, and mask
class DataLabelMaskDataset(Dataset):
    """Custom dataset
    data: [n_samples, length_of_gm, 1]   e.g. [60940, 3000, 1]
    labels: [n_samples]                  e.g. [60940]
    masks: [n_samples, length_of_gm]     e.g. [60940, 3000]

    In fact, what this custom dataset do is just unsqueeze the mask data, comparing to th TensorDataset class
    """
    def __init__(self, data, labels, masks):
        self.data = data
        self.labels = labels
        self.masks = masks

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        mask = self.masks[idx]
        # mask need to add one dimension to adjust the multihead attention
        # len(attn_mask) should equal to 3000
        # mask [3000] -> [1, 3000]
        mask = mask.unsqueeze(0)
        return data, label, mask

# 6.1 data, label, mask, frequency
class DataLabelMaskFreqDataset(Dataset):
    """Custom dataset
    data: [n_samples, length_of_gm, 1]          e.g. [60940, 3000, 1]
    labels: [n_samples]                         e.g. [60940]
    masks: [n_samples, length_of_gm]            e.g. [60940, 3000]
    frequency: [n_samples, length_of_gm/2]      e.g. [60940, 1500]

    In fact, what this custom dataset do is just unsqueeze the mask data, comparing to th TensorDataset class
    """
    def __init__(self, data, labels, masks, frequency):
        self.data = data
        self.labels = labels
        self.masks = masks
        self.frequency = frequency

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        frequency = self.frequency[idx]
        mask = self.masks[idx]
        # mask need to add one dimension to adjust the multihead attention
        # len(attn_mask) should equal to 3000
        # mask [3000] -> [1, 3000]
        mask = mask.unsqueeze(0)
        return data, label, mask, frequency

# 7 data, label, mask
def CreateDataLoadersWithMultiDatasetV1(data_list:List[torch.Tensor],
                                    label_list:List[torch.Tensor], 
                                    mask_list:List[torch.Tensor],
                                    train_ratio:float=0.8, 
                                    batch_size:int=64) -> Tuple[DataLoader, DataLoader]:
    """ Merge multi dataset and label and split them into train and validation set

    Args:
        data_list: [n_samples, length_of_gm, 1]   e.g. [60940, 3000, 1]
        label_list: [n_samples]
        mask_list: [n_samples, length_of_gm]     e.g. [60940, 3000]
        train_ratio: ratio of training data of the whole dataset
        batch_size: batch size

    return:
        train_dataloader, validation_dataloader
    """
    # merge all dataset in dataset list
    all_data = torch.cat(data_list, dim=0)
    all_labels = torch.cat(label_list, dim=0)
    all_masks = torch.cat(mask_list, dim=0)

    # create CustomTensorDataset instance
    combined_dataset = DataLabelMaskDataset(all_data, all_labels, all_masks)

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
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, validation_dataloader

# 7.1 data, label, mask, frequency
def CreateDataLoadersWithMultiDatasetV2(data_list:List[torch.Tensor],
                                    label_list:List[torch.Tensor], 
                                    mask_list:List[torch.Tensor],
                                    frequency_list:List[torch.Tensor],
                                    train_ratio:float=0.8, 
                                    batch_size:int=64) -> Tuple[DataLoader, DataLoader]:
    """ Merge multi dataset and label and split them into train and validation set

    Args:
        data_list: [n_samples, length_of_gm, 1]   e.g. [60940, 3000, 1]
        label_list: [n_samples]
        mask_list: [n_samples, length_of_gm]     e.g. [60940, 3000]
        frequency_listL [n_samples, length_of_gm / 2]   e.g. [60940, 1500]
        train_ratio: ratio of training data of the whole dataset
        batch_size: batch size

    return:
        train_dataloader, validation_dataloader
    """
    # merge all dataset in dataset list
    all_data = torch.cat(data_list, dim=0)
    all_labels = torch.cat(label_list, dim=0)
    all_masks = torch.cat(mask_list, dim=0)
    all_frequency = torch.cat(frequency_list, dim=0)

    # create CustomTensorDataset instance
    combined_dataset = DataLabelMaskFreqDataset(all_data, all_labels, all_masks, all_frequency)

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
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, validation_dataloader

# 8
def FastFourierTransform(input_data:torch.Tensor) -> torch.Tensor:
    """FFT the ground motion data

    [60940, 3000]  -->  [60940, 1500]

    Args:
    input_data: data in torch.Tensor type, usually a ground motion time seires curve in Seismic Transformer

    Returns:
    FFT data in torch.Tensor type
    """
    # excute fft to each row
    freq_domain_data = np.fft.rfft(input_data)

    # fft data is complex number, we only need the real part
    # just need 1500 points, because the rest of the points are the mirror of the first 1500 points
    output_data = np.abs(freq_domain_data[:, :1500])
    output_data = torch.from_numpy(output_data)
    print(output_data.shape)

    return output_data

# 9
def CreateKeyPaddingMask_AllFalse(batch_size, seq_len) -> torch.Tensor:
    """Create key padding mask for encoder in SeT-3
    
    Args:
        size: size of the mask, usually the length of the sequence, 12 in SeT-3(encoder)
    """
    key_padding_mask = torch.zeros(batch_size, seq_len)
    return key_padding_mask == 1  # convert to bool

# 10
def CreateLookAheadMask(seq_len) -> torch.Tensor:
    """Create look ahead mask for decoder in SeT-3

    Args:
        size: size of the mask, usually the length of the sequence, 12 in SeT-3(decoder)
    """
    look_ahead_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return look_ahead_mask == 1  # convert to bool