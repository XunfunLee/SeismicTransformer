"""
Contains functionality for save, load the model.
1. Set the device to train.
2. Save the model(entire model or parameters only or both).
3. Load the model from parameters.
4. Load the entire model.
5. Count the number of training in record in csv file.
6. Save the results of the model to a csv file.

Author: Jason Jiang (Xunfun Lee)
Date: 2023.12.2
"""

from pathlib import Path
import os
import torch
import csv
from datetime import datetime
import pandas as pd
import numpy as np
from .data_preparation import FastFourierTransform, MaskingData, TransformMaskDataV1, TransformMaskDataV2
from .visualization import SaveAttnHeatMap, SavePosiHeadMap, SaveAttnHeatMapBarChart, SaveGM, SaveGMPatches

def SetDevice() -> torch.device:
    """ Set the device for training

        cuda: Nividia GPU
        mps: apple M-chip
        cpu: CPU
    """
    if torch.cuda.is_available():
        device = "cuda"
        print("GPU: cuda")
        print("CUDA device numbers: ", torch.cuda.device_count())
    elif torch.backends.mps.is_available():
        device = "mps"
        print("GPU: MPS")
    else:
        device = "cpu"
        print("No GPU found, using CPU")

    return device

# save model
def SaveModel(model: torch.nn.Module,
              num_of_training: int,
              hidden_size: int,
              num_of_layer: int,
              num_of_head: int,
              num_of_epoch: int,
              validation_acc: float,
              f1_score: float,
              save_mode: str):
    """ save the model

    Args:
        model: target model you want to save
        hidden_size: size of the embedding size
        num_of_layer: number of the transformer layer
        num_of_head: number of head the multi-head attention
        num_of_epoch: number of train epoch
        save_mode: save the model or parameters only or both, "model" or "params" or "both"
    """
    # turn model into eval() mode
    # model.eval()

    # define the model file name
    file_path = Path("Models")
    # make a dirctory if it is not exist
    file_path.mkdir(parents=True, exist_ok=True)

    # just save in 2 double
    validation_acc = round(float(validation_acc[-1]), 2)
    f1_score = round(float(f1_score), 2)

    num_of_training += 1

    # save the entire model
    if save_mode == "model":
        file_name = f"SeT_{num_of_training}_HS_{hidden_size}_Layer_{num_of_layer}_Head_{num_of_head}_Epoch_{num_of_epoch}_Acc_{validation_acc}_F1_{f1_score}_Model.pth"  
        save_path = os.path.join(file_path, file_name)
        torch.save(model, save_path)
        print(f"Model saved to {save_path}")

    # only save the params of the model
    elif save_mode == "params":
        file_name = f"SeT_{num_of_training}_HS_{hidden_size}_Layer_{num_of_layer}_Head_{num_of_head}_Epoch_{num_of_epoch}_Acc_{validation_acc}_F1_{f1_score}_Params.pth"
        save_path_params = os.path.join(file_path, file_name)
        torch.save(model.state_dict(), save_path_params)
        print(f"Model parameters saved to {save_path_params}")

    elif save_mode == "both":
        # save the model
        model_name = f"SeT_{num_of_training}_HS_{hidden_size}_Layer_{num_of_layer}_Head_{num_of_head}_Epoch_{num_of_epoch}_Acc_{validation_acc}_F1_{f1_score}_Model.pth"  
        save_path = os.path.join(file_path, model_name)
        torch.save(model, save_path)
        print(f"Model saved to {save_path}")
        # save the params
        model_params_name = f"SeT_{num_of_training}_HS_{hidden_size}_Layer_{num_of_layer}_Head_{num_of_head}_Epoch_{num_of_epoch}_Acc_{validation_acc}_F1_{f1_score}_Params.pth"
        save_path_params = os.path.join(file_path, model_params_name)
        torch.save(model.state_dict(), save_path_params)
        print(f"Model parameters saved to {save_path_params}")

    else:
        raise ValueError("Invalid save_mode. Expected 'model' or 'params' or 'both")

def LoadModelParams(model: torch.nn.Module,
                    params_name: str) -> torch.nn.Module:
    """ load the model from params

    Args:
        model: target model you want to save
        params_name: model params name(.pth)
    """
    params_path = os.path.join("Models", params_name)
    model = model.load_state_dict(torch.load(params_path))

    assert os.path.isfile(params_path), f"The file doesn't exists."

    return model

def LoadModel(model_name: str) -> torch.nn.Module:
    """ load the entire model

    Args:
        model_name: model name(.pth)
    """
    model_path = os.path.join("Models", model_name)
    model = torch.load(model_path)

    assert os.path.isfile(model_path), f"The file doesn't exists."

    return model

def CountNumOfTraining() -> int:
    """ Count the number of training

    Return:
        return the number of training counted in the "training_results.csv" files
    """
    filename = "training_results.csv"
    file_path = os.path.join("Models", filename)

    # Check if the file exists
    file_exists = os.path.isfile(file_path)

    # initialize the row number
    row_number = 0

    if file_exists:
        # Open the file to read the last row number
        with open(file_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            # Skip the title row
            next(reader, None)
            # Get the final row number
            for row in reader:
                if row:  # Make sure the row is not empty
                    last_row_number = int(row[0])
            row_number = last_row_number

    return row_number

def RecordResults(GM: str, 
                  num_of_training: int,
                  patch_size: int, 
                  hidden_size: int, 
                  num_of_layer: int,
                  num_of_head: int, 
                  num_of_epoch: int, 
                  batch_size: int,
                  learning_rate: float, 
                  weight_decay: float, 
                  mlp_dropout: float,
                  train_acc: float,
                  validation_acc: float,
                  f1_score: float,
                  recall_score: float,
                  times: float):
    """ save the results and information of the model into a csv file

    Args:
        all the params here is to descript the model and results
    """
    filename = "training_results.csv"
    file_path = os.path.join("Models", filename)

    # Define fieldnames for CSV
    fieldnames = ['No', 'Date', 'GM', 'patch_size', 'hidden_size', 'num_of_layer', 'num_of_head',
                  'num_of_epoch', 'batch_size', 'learning_rate', 'weight_decay', 'mlp_dropout', 
                  'train_acc', 'validation_acc', 'f1_score', 'recall_score','times']

    # take 4 of the number
    f1_score = round(float(f1_score), 4)
    recall_score = round(float(recall_score), 4)
    train_acc = round(float(train_acc[-1]), 4)
    validation_acc = round(float(validation_acc[-1]), 4)

    # Check if the file exists
    file_exists = os.path.isfile(file_path)

    # Open the file to append new data
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # If the file is new, write a title header
        if not file_exists:
            writer.writeheader()

        # Increment row number
        num_of_training += 1

        # Record the training date and time
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

        # Write in the data
        writer.writerow({
            'No': num_of_training,
            'Date': formatted_time,
            'GM': GM,
            'patch_size': patch_size,
            'hidden_size': hidden_size,
            'num_of_layer': num_of_layer,
            'num_of_head': num_of_head,
            'num_of_epoch': num_of_epoch,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'mlp_dropout': mlp_dropout,
            'train_acc': train_acc,
            'validation_acc': validation_acc,
            'f1_score': f1_score,
            'recall_score': recall_score,
            "times": times
        })

    print(f'Result of this training is recorded into "{file_path}"')

def CreateOutputFolder(num_of_training: int,
                    hidden_size: int, 
                    num_of_layer: int,
                    num_of_head: int, 
                    num_of_epoch: int) -> str:
    """ create a csv folder to store the acc and loss value and image during traning

    Args:
        all is similar to the above

    Returns:    
        save_dir: save dirctory
    """
    model_name = f"SeT_{num_of_training}_HS_{hidden_size}_Layer_{num_of_layer}_Head_{num_of_head}_Epoch_{num_of_epoch}"  
    save_dir = os.path.join("Models", "Output", model_name)
    os.makedirs(save_dir, exist_ok=True)

    return save_dir

def CreateLogFile(save_dir: str) -> str:
    """ create a csv file to store the acc and loss value during traning

    Args:
        save_dir: save dirctory

    Returns:
        log_filename: log file name(csv)
    """
    log_filename = os.path.join(save_dir, "training_log.csv")
    with open(log_filename, "w") as log_file:
        log_file.write("epoch,train_loss,validation_loss,train_accuracy,validation_accuracy\n")

    return log_filename

def LogEpochData(epoch:int, 
                 train_loss:float, 
                 validation_loss:float, 
                 train_acc:float,
                 validation_acc:float,
                 log_filename:str):
    """ log each loss and acc in each epoch

    Args:
        don't need to explain
    """

    try:
        with open(log_filename, "a") as log_file:
            log_file.write(f"{epoch},{round(train_loss,4)},{round(validation_loss,4)},{round(train_acc,4)},{round(validation_acc,4)}\n")
    except PermissionError:
        print(f"Permission denied while writing to {log_filename}.")

def LogEpochDataV3(epoch:int, 
                 train_loss:float, 
                 train_acc:float,
                 train_mse:float,
                 validation_loss:float, 
                 validation_acc:float,
                 validation_mse:float,
                 log_filename:str):
    """ log each loss and acc in each epoch
    Adding mse to the log file compared to LogEpochData

    Args:
        don't need to explain
    """

    try:
        with open(log_filename, "a") as log_file:
            log_file.write(f"{epoch},{round(train_loss,4)},{round(validation_loss,4)},{round(train_acc,4)},{round(validation_acc,4)},{round(train_mse,4)},{round(validation_mse,4)}\n")
    except PermissionError:
        print(f"Permission denied while writing to {log_filename}.")


# Test the model with example ground motion data
def TestModelWithExampleGM(file_path: str,
                           model_name: str,
                           model_version: str="V2.0",
                           mask_mode: bool=False,
                           padding_mode:str="front"):
    ''' Using the example ground motion data to test the model

    Args:
        file_path: the path of the example ground motion data
        model_name: the name of the model
        model_version: the version of the model, SeT-1 or SeT-2, 'V1.0' or 'V2.0'
        mask_mode: whether to use the mask data
        padding_mode: how to process the data if the length is not 3000, 'front' or 'back'

    '''

    #########################################  Load the example ground motion  #########################################
    '''
    can load txt, csv, xlsx file to test.
    column 1 must be the time data
    column 2 must be the value data (acceleration)
    header is optional, will be ignore if exist
    '''

    # load the data
    if file_path.lower().endswith('.txt'):
        # assume that the data is in the format of 2 columns
        data = pd.read_csv(file_path, header=None, delimiter="\t")
    elif file_path.lower().endswith(('.xls', '.xlsx', '.csv')):
        # load the first sheet
        data = pd.read_excel(file_path, header=None)
    else:
        raise ValueError("Unsupported file format! Need to be 'txt', 'xls', 'xlsx', 'csv'")

    # remove the title
    if pd.api.types.is_string_dtype(data.iloc[0, 0]):
        data = data.iloc[1:]

    # get two columns
    column1 = data.iloc[:, 0].dropna().values
    column2 = data.iloc[:, 1].dropna().values

    #########################################  Process the example ground motion  #########################################
    '''
    check the length of the data:
    1. standardlize the sample rate to 50Hz
    2. padding or cut the data to 3000 points
    '''

    # check the sample rate, 50Hz is the standard
    sample_interval = column1[1] - column1[0]
    if sample_interval == 0.02:                 # 50Hz
        column1_standardlize = column1
        column2_standardlize = column2
    elif sample_interval == 0.01:               # 100Hz
        column1_standardlize = column1[::2]     # downsample to 50Hz
        column2_standardlize = column2[::2]
    else:
        raise ValueError("Sample rate is not 50Hz or 100Hz! Please check the data or update the code.")
    
    # check the length of X and Y, they should be equal
    assert len(column1_standardlize) == len(column2_standardlize), "The length of X and Y are not equal!"

    # check the number of points, padding or cut
    num_points = len(column2_standardlize)
    if num_points < 3000:
        if padding_mode == "front":
            # padding the data in the front
            padding_needed = 3000 - num_points
            process_data = np.pad(column2_standardlize, (padding_needed, 0), 'constant')
        elif padding_mode == "back":
            # padding the data in the back
            padding_needed = 3000 - num_points
            process_data = np.pad(column2_standardlize, (0, padding_needed), 'constant')
        else:
            raise ValueError("Invalid padding_mode. Expected 'front' or 'back'")
    # cut the data if the length is larger than 3000
    elif num_points > 3000:
        # find the peak value of the gm
        peak_index = np.argmax(column2)
        # get 1500 points before and after the peak
        start_index = max(peak_index - 1500, 0)
        end_index = min(peak_index + 1500, num_points)
        process_data = column2_standardlize[start_index:end_index]
    
    # check the length of the data
    assert len(process_data) == 3000, "The length after process is not 3000!"

    # turn array into tensor    [3000]  --> [1,3000,1]
    seismic_tensor = torch.tensor(process_data, dtype=torch.float).view(1, -1, 1)

    #########################################  Create the FFT and mask data  #########################################
    '''
    1. Generate FFT data
    2. Generate mask data depend on the model version and mask mode
    '''

    # generate the FFT data
    FFT_data = FastFourierTransform(seismic_tensor)
    FFT_data = FFT_data.squeeze(2)
    # convert FFT_data to float32 tensor, MPS can't use float64

    FFT_data = FFT_data.float()
    # generate the mask
    if mask_mode == False:
        mask_data_patches = None
    elif mask_mode == True:
        mask_data = MaskingData(input_array=seismic_tensor.numpy())
        # transform the mask into sequence
        if model_version == "V2.0":
            # [60940, 3000] --> [60940, 14] = [CLS] + 12 patches(time + [TIME]) + 1 patch(frequency + [FREQ])
            mask_data_patches = TransformMaskDataV2(mask_data=mask_data)
        elif model_version == "V1.0":
            # [60940, 3000] --> [60940, 13] = [CLS] + 12 patches(time)
            mask_data_patches = TransformMaskDataV1(mask_data=mask_data)
        else:
            raise ValueError("This function is for V1.0 and V2.0 only! Type in 'V1.0' or 'V2.0'")

    #########################################  pass through the model  #########################################
    '''
    1. Load the model and move every thing to device
    2. Put example ground motion data into model and get the predicted labels
    '''

    # load the model
    device = SetDevice()
    model_path = os.path.join("Models", model_name + '.pth')
    map_location=torch.device(device)           # incase raise error in MacOS
    model = torch.load(model_path, map_location=map_location)
    # send data to device
    seismic_tensor = seismic_tensor.to(device)
    FFT_data = FFT_data.to(device)

    # turn model into eval() mode
    model.eval()

    with torch.inference_mode():  # using torch.inference_mode() to speed up the model
        y_logit = model(x=seismic_tensor,mask=mask_data_patches,frequency=FFT_data)
        probabilities = torch.softmax(y_logit, dim=1)  # calculate the probability
        predicted_labels = torch.argmax(probabilities, dim=1)  # get the predicted labels

        # print the results
        max_probabilities, predicted_labels = probabilities.max(dim=1)
        print("-------------------------------------------------------")
        print("Example GM:", file_path)
        print("Predicted labels:", predicted_labels)
        print("Max probabilities:", max_probabilities)
        print("-------------------------------------------------------")
    
    #########################################  Save the plots  #########################################
    '''
    1. Create test folder and save the ground motion plot and patches plot
    2. Save the attention heat map and position head map
    3. Create attention weights subfolder and save the attn heat map bar chart
    '''

    # Create test folder
    save_dir = os.path.join("Models", model_name, "Test")

    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # [1, 3000]
    SaveGM(gm_timeseries=process_data,
           save_dir=save_dir)
    
    SaveGMPatches(gm_timeseries=process_data,
                    save_dir=save_dir)

    SaveAttnHeatMap(model=model,
                    save_dir=save_dir,
                    num_of_layer=len(model.attention_weights_list))
    
    SavePosiHeadMap(model=model,
                    save_dir=save_dir)

    SaveAttnHeatMapBarChart(model=model,
                            save_dir=save_dir)
    ### ------------------------------------------------------------------------------------------------ ###
