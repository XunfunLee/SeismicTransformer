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

def SetDevice() -> torch.device:
    # device
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
        save_mode: save the model or parameters only or both
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
                  'train_acc', 'validation_acc', 'f1_score', 'times']

    # take 4 of the number
    f1_score = round(float(f1_score), 4)
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