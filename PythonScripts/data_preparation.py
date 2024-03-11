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
11. ReadH5FileV1(): Read h5 file from .h file (for SeT-3)
12. ReadH5FileV2(): Read h5 file from .h file (for SeT-4)
13. Class DynamicDatasetV1(): for SeT-4 to load huge data in dynamic way
14. custom_collate(): for SeT-4 to load huge data in dynamic way

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

# 11. ReadH5FileV1(): Read h5 file from .h file (for SeT-3 and above version)
def ReadH5FileV1(path:str,
               dataset_name:str) -> np.ndarray:
    
    # Check if the file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"The file at {path} does not exist.")

    with h5py.File(path, 'r') as f:
        # Check if the dataset exists
        if dataset_name not in f:
            raise KeyError(f"Dataset {dataset_name} not found in the file.")

        data_read = f[dataset_name][:]

        # Check if the dataset is empty
        if data_read.size == 0:
            raise ValueError(f"The dataset {dataset_name} is empty.")

        print("Data read successfully")

    return data_read

# 12. ReadH5FileV2(): Read h5 file from .h file (for SeT-4 and above version)
def ReadH5FileV2(path:str):
    """Directly read structural info, acc_floor_response, blg_ds from .h5 file

    Args:
        path: path of the .h5 file
    """
    # Check if the file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"The file at {path} does not exist.")

    with h5py.File(path, 'r') as f:
        # Check if the dataset exists
        if "Acc_Floor_Response" not in f:
            raise KeyError(f"Dataset 'Acc_Floor_Response' not found in the file.")

        acc_floor_response = f["Acc_Floor_Response"][:]

        if "Blg_Damage_State" not in f:
            raise KeyError(f"Dataset 'Blg_Damage_State' not found in the file.")

        blg_ds = f["Blg_Damage_State"][:]

        print("Data read successfully")

    return acc_floor_response, blg_ds   # [198018, 3000] & [198018, 1] in SeT-4

# 13 Class dynamic dataset
class DynamicDatasetV1(Dataset):
    """Load h5 files for SeT-4 training

    Args:
        gm_file_path: path to the ground motion file
        building_files_dir: directory containing the building files
        device: device to load the tensors (for the dictionary of building attributes)
    
    """
    def __init__(self, 
                 gm_file_path: str, 
                 building_files_dir: str,
                 device: torch.device):
        
        self.gm_file_path = gm_file_path
        self.building_files = [
            os.path.join(building_files_dir, f)
            for f in os.listdir(building_files_dir) if f.endswith('.h5')
        ]
        self.device = device

        with h5py.File(self.gm_file_path, 'r') as f:
            self.num_gm_samples = f['Acc_GMs'].shape[0]  # Total number of ground motions

        self.num_building_conditions = len(self.building_files)  # Total number of building conditions
        self.total_samples = self.num_gm_samples * self.num_building_conditions  # Total dataset size

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Compute individual indices
        gm_index = idx // self.num_building_conditions
        building_index = idx % self.num_building_conditions
        
        # Get the ground motion data
        with h5py.File(self.gm_file_path, 'r') as f:
            gm_data = f['Acc_GMs'][gm_index]

        # output [3000, 1]
        gm_data = torch.from_numpy(gm_data).float().unsqueeze(-1)

        # Get a specific building file based on the building index
        bldg_file = self.building_files[building_index]

        # Read data for the specific building condition
        with h5py.File(bldg_file, 'r') as f:
            acc_floor_response = f['Acc_Floor_Response'][gm_index]
            blg_damage_state = f['Blg_Damage_State'][gm_index][0]  # Assuming one-dimensional array

            blg_attributes = {}
            for key, dataset in f["Blg_Attributes"].items():
                # Convert the data to a numpy
                data = dataset[:]

                # if the data type is integer
                if dataset.dtype.kind in ['i', 'u']:
                    tensor = torch.from_numpy(data).to(torch.int64)
                # if the data type is float
                elif dataset.dtype.kind == 'f':
                    tensor = torch.from_numpy(data).to(torch.float32)
                else:
                    raise TypeError(f'Unknown data type: {data.dtype}')
                
                # Add the tensor to the dictionary and send to device
                blg_attributes[key] = tensor.to(self.device)

        # output [3000, 1]
        acc_floor_response = torch.from_numpy(acc_floor_response).float().unsqueeze(-1)

        # output [,]
        blg_damage_state = torch.tensor([blg_damage_state], dtype=torch.long)

        '''
        gm_data: [3000, 1]
        blg_attributes: dictionary of tensors(sent to device)
        acc_floor_response: [3000, 1]
        blg_damage_state: [1]
        '''
        return gm_data, blg_attributes, acc_floor_response, blg_damage_state

# 14. custom_collate(): for SeT-4 to load huge data in dynamic way
def custom_collate(batch):
    gm_data_list, building_attributes_list, acc_floor_response_list, blg_damage_state_list = zip(*batch)

    # Stack ground motion data, floor response data, and damage state data
    gm_data_batch = torch.stack(gm_data_list)
    acc_floor_response_batch = torch.stack(acc_floor_response_list)
    blg_damage_state_batch = torch.stack(blg_damage_state_list)

    # Combine building attributes into a batched format
    batched_building_attributes = {}
    for key in building_attributes_list[0].keys():
        key_tensor_list = [d[key] for d in building_attributes_list]
        # Since batch size is 1, we can directly extract the single tensor
        # If batch size were greater than 1, we would use torch.stack(key_tensor_list)
        batched_building_attributes[key] = key_tensor_list[0]

    return gm_data_batch, batched_building_attributes, acc_floor_response_batch, blg_damage_state_batch