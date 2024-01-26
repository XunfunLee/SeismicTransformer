"""
Contains functionality for training the model.
1. train_step() for training loop.
2. validation_step() for validation loop.
3. train() for the combinition of training and validation loop.
4. test() for testing the model, return loss, accuracy and results for confusion matrix.
5. ReduceLROnPlateau: modify learning rate based on loss in validation set.

Author: Jason Jiang (Xunfun Lee)
Date: 2023.12.2
"""

from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import torch
import numpy as np
from sklearn.metrics import f1_score, recall_score, mean_squared_error
import math
from .utility import LogEpochData, LogEpochDataV3

# define the global step, make sure the warmup step only happen once
global_step = 0
warmup_done = False

# 1. train.step()
def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               lr_scheduler_warmup: torch.optim.lr_scheduler.LambdaLR,
               num_warmup_steps: int,
               device: torch.device) -> Tuple[float, float]:
    """ train loop for each epoch

    Args:
        model: target model you want to train
        dataloader: dataloader of training data
        loss_fn: loss function for the training
        optimizer: optimizer for the training
        device: training decive (e.g. cuda for nvidia, mps for mac)

    Returns:
        train_loss: loss value of the training dataset
        train_acc: accuracy of the training dataset
    """
    # global variable
    global global_step, warmup_done  # global_step is used for lr_scheduler_warmup

    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y, Mask, frequency) in enumerate(dataloader):
        # Send data to target device, always GPU

        # Mask.shape = [batch_size, 1, seq_len]     e.g. [64, 1, 13]  --> [64, 13] or [64, 14]
        # [batch_size, 1, seq_len] --> [batch_size, seq_len] to 2D
        # pytorch will due with the 2D to 4D automatically
        Mask = Mask.squeeze(1)

        
        X, y, Mask, frequency = X.to(device), y.to(device), Mask.to(device), frequency.to(device)
        '''
        X.shape = [batch_size, seq_len, input_dim]     e.g. [batch_size, 3000, 1]
        y.shape = [batch_size]                         e.g. [batch_size]
        Mask.shape = [batch_size, seq_len]             e.g. [batch_size, 14]
        frequency.shape = [batch_size, seq_len]        e.g. [batch_size, 1500]
        '''
        # print(f"X.shape = {X.shape}")
        # print(f"y.shape = {y.shape}")
        # print(f"Mask.shape = {Mask.shape}")
        # print(f"frequency.shape = {frequency.shape}")

        # Forward pass
        y_pred = model(X, mask=Mask, frequency=frequency)

        # Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # Optimizer zero grad
        optimizer.zero_grad()

        # Loss backward
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Step up the learning rate scheduler, warmup strategy
        lr_scheduler_warmup.step()
        global_step += 1

        # print something after warmup
        if not warmup_done and global_step >= num_warmup_steps:
            print(f"Warmup completed at step {global_step}")
            warmup_done = True  # incase print twice

        # if loss = nan, break the training
        if torch.isnan(loss):
            print("Loss is nan, stopping training.")
            break  # jump out of the training

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

# 2. validation_step()
def validation_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """ validation loop for each epoch

    Args:
        model: target model you want to validation
        dataloader: dataloader of validation data
        loss_fn: loss function for the validation
        device: training decive (e.g. cuda for nvidia, mps for mac)

    Returns:
        validation_loss: loss value of the validation dataset
        validation_acc: accuracy of the validation dataset
    """
    # Put model in eval mode
    model.eval() 

    # Setup validation loss and validation accuracy values
    validation_loss, validation_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y, Mask, frequency) in enumerate(dataloader):
            # Send data to target device
            # Mask.shape = [batch_size, 1, seq_len]     e.g. [64, 1, 13]  --> [64, 13]
            # [batch_size, 1, seq_len] --> [batch_size, seq_len] to 2D
            # pytorch will due with the 2D to 4D automatically
            Mask = Mask.squeeze(1)

            X, y, Mask, frequency = X.to(device), y.to(device), Mask.to(device), frequency.to(device)

            # 1. Forward pass
            validation_pred_logits = model(X, mask=Mask, frequency=frequency)

            # 2. Calculate and accumulate loss
            loss = loss_fn(validation_pred_logits, y)
            validation_loss += loss.item()

            # Calculate and accumulate accuracy
            validation_pred_labels = validation_pred_logits.argmax(dim=1)
            validation_acc += ((validation_pred_labels == y).sum().item()/len(validation_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch 
    validation_loss = validation_loss / len(dataloader)
    validation_acc = validation_acc / len(dataloader)
    return validation_loss, validation_acc

# 3. train(): combining the train_step() and validation_step()
def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          validation_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          lr_scheduler_warmup: torch.optim.lr_scheduler.LambdaLR,
          num_warmup_steps: int,
          lr_scheduler_decay: torch.optim.lr_scheduler.ReduceLROnPlateau,
          device: torch.device,
          log_filename: str) -> Dict[str, List]:
    """ whole action(train + validate) for each epoch

    Args:
        model: target model you want to validation
        train_dataloader: dataloader of training data
        train_dataloader: dataloader of validation data
        optimizer: optimizer for the model
        loss_fn: loss function for the validation
        epochs: number of loop for the training and validation
        device: training decive (e.g. cuda for nvidia, mps for mac)

    Returns:
        train_loss: loss value of the training dataset
        train_acc: accuracy of the training dataset
        validation_loss: loss value of the validation dataset
        validation_acc: accuracy of the validation dataset
    """  
    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "validation_loss": [],
               "validation_acc": [],
               "is_nan": []
    }


    # Make sure model on target device
    model.to(device)

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        # train step
        train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          lr_scheduler_warmup=lr_scheduler_warmup,
                                          num_warmup_steps=num_warmup_steps,
                                          device=device)
        # if train loss = nan, break
        if math.isnan(train_loss):
            print(f"Epoch {epoch}:Train loss is NaN. Stopping training.")
            results["is_nan"].append("yes")
            break

        # validation step
        validation_loss, validation_acc = validation_step(model=model,
                                        dataloader=validation_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)
         
        # if validation loss = nan, break
        if math.isnan(validation_loss):
            print(f"Epoch {epoch}:Validation loss is NaN. Stopping training.")
            results["is_nan"].append("yes")
            break

        # put validation_loss to lr_scheduler
        lr_scheduler_decay.step(validation_loss)

        # update log file(csv file)
        LogEpochData(epoch=epoch,
                     train_loss=train_loss,
                     validation_loss=validation_loss,
                     train_acc=train_acc,
                     validation_acc=validation_acc,
                     log_filename=log_filename)

        # Print out what's happening
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"validation_loss: {validation_loss:.4f} | "
          f"validation_acc: {validation_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["validation_loss"].append(validation_loss)
        results["validation_acc"].append(validation_acc)

    # Return the filled results at the end of the epochs
    return results

# 4. test(): evaluate the model using test data
def test(model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device) -> Tuple[Dict[str, float], Tuple[np.ndarray, np.ndarray]]:
    """ validation loop for each epoch

    Args:
        model: target model you want to test
        dataloader: dataloader of test data
        device: test decive (e.g. cuda for nvidia, mps for mac)

    Returns:
        results: accuracy of the test dataset and caculate F1 scores
        y_pred: label of the model prediction
        y_true: label of the true dataset
    """
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    # Initialize the correct predictions count and all predictions
    correct_preds = 0
    total_preds = 0

    y_preds = []
    y_trues = []

    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y, Mask, frequency) in enumerate(dataloader):
            # Send data to target device
            # Mask.shape = [batch_size, 1, seq_len]     e.g. [64, 1, 13]  --> [64, 13]
            # [batch_size, 1, seq_len] --> [batch_size, seq_len] to 2D
            # pytorch will due with the 2D to 4D automatically
            Mask = Mask.squeeze(1)

            X, y, Mask, frequency = X.to(device), y.to(device), Mask.to(device), frequency.to(device)

            # Forward pass
            y_logit = model(X, mask=Mask, frequency=frequency)

            # Predictions
            _, predicted_labels = torch.max(y_logit, 1)

             # Count correct predictions
            correct_preds += (predicted_labels == y).sum().item()
            total_preds += y.size(0)

            # Save predictions and actual labels for metrics calculations
            y_preds.append(predicted_labels.cpu())
            y_trues.append(y.cpu())

    # Convert predictions and actual labels from list to single tensor
    y_pred_tensor = torch.cat(y_preds)
    y_true_tensor = torch.cat(y_trues)

    # Convert tensor to NumPy array
    y_pred_numpy = y_pred_tensor.numpy()
    y_true_numpy = y_true_tensor.numpy()

    # Calculate overall accuracy
    test_accuracy = correct_preds / total_preds if total_preds > 0 else 0.0

    # Calculate F1 score, default is 'binary' which requires binary class labels,
    # for multi-class classification use 'micro', 'macro', or 'weighted'
    f1 = f1_score(y_true_numpy, y_pred_numpy, average='micro')

    # Caculate Recall_Score for classification task
    Rs = recall_score(y_true_numpy, y_pred_numpy, average='micro')

    # Build the results dictionary
    results = {
        'test_accuracy': test_accuracy,
        'f1_score': f1,
        'recall_score': Rs,
    }

    # Return results and prediction values
    return results, (y_pred_numpy, y_true_numpy)


##### ---------------------------------- SeiscmicTransformer V3.0 ---------------------------------- #####
"""
Author: Jason Jiang (Xunfun Lee)

date: 2023.01.14

For the reason that all the modules were rewrited, the train.py is also rewrited in SeT-3.
The main differences between SeT-3 and previous versions are:
1. SeT-3 is a multi-task model, including classification and regression, so we have two loss functions.
2. Frequency embedding, patch embedding and position embedding are taken good modularization.
3. Maksing including key padding masking (encoder) and attention mask (decoder) are also taken good modularization.

The functions in this file are:
1. train_step_set3() for training loop.
2. validation_step_set3() for validation loop.
3. train_set3() for the combinition of training and validation loop.

"""

from .data_preparation import CreateKeyPaddingMask_AllFalse, CreateLookAheadMask


global_stepV3 = 0
warmup_doneV3 = False

# 1. train_step
def train_step_set3(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader, 
               loss_fn_classification: torch.nn.Module, 
               loss_fn_regression: torch.nn.Module,
               loss_fn_weight_classification: float,
               optimizer: torch.optim.Optimizer, 
               lr_scheduler_warmup: torch.optim.lr_scheduler.LambdaLR, 
               num_warmup_steps: int, 
               teacher_forcing_ratio: float,
               device: torch.device) -> Tuple[float, float]:
    
    """Train step for SeismicTransformer V3.0
    
    Args:
        model: target model you want to train
        dataloader: dataloader of training data
        loss_fn_classification: loss function for the classification
        loss_fn_regression: loss function for the regression
        loss_fn_weight_classification: weight for the classification loss
        optimizer: optimizer for the model
        lr_scheduler_warmup: learning rate scheduler for warmup
        num_warmup_steps: number of warmup steps
        device: training decive (e.g. cuda for nvidia, mps for mac)
    """

    global global_stepV3, warmup_doneV3

    # Set model to training mode
    model.train()

    # Initialize the loss and classification accuracy
    train_loss, train_acc_classification, train_mse_regression = 0.0, 0.0, 0.0

    for _, (gm_sequence, floor_sequence, label) in enumerate(dataloader):

        batch_size = gm_sequence.size(0)

        # for test reason, seq_len should be parameterized
        key_padding_mask = CreateKeyPaddingMask_AllFalse(batch_size=batch_size, seq_len=14)
        attn_mask = CreateLookAheadMask(seq_len=12)
        
        gm_sequence = gm_sequence.to(device)
        label = label.to(device).long()         # label need to be long() type to run CrossEntropyLoss()
        floor_sequence = floor_sequence.to(device)
        key_padding_mask = key_padding_mask.to(device)
        attn_mask = attn_mask.to(device)

        # Forward pass
        damage_state_pred, dynamic_response = model(encoder_input=gm_sequence,
                                                    decoder_input=floor_sequence,
                                                    key_padding_mask=key_padding_mask,
                                                    attn_mask=attn_mask,
                                                    teacher_forcing_ratio=teacher_forcing_ratio)

        
        # Calculate and accumulate classification accuracy
        y_pred_class = torch.argmax(torch.softmax(damage_state_pred, dim=1), dim=1)
        y_pred_class = y_pred_class.float()
        train_acc_classification += (y_pred_class == label).sum().item() / label.size(0)

        # Check that weight is between 0 and 1
        assert 0 <= loss_fn_weight_classification <= 1, \
            "loss_fn_weight_classification should be between 0 and 1"


        # Accumulate loss
        # Calculate classification and regression losses
        loss_classification = loss_fn_classification(damage_state_pred, label)  # here is direct output of the model
        loss_regression = loss_fn_regression(dynamic_response, floor_sequence)
        # Combine losses
        loss = loss_fn_weight_classification * loss_classification + \
               (1 - loss_fn_weight_classification) * loss_regression
        train_loss += loss.item()

        # Zero gradients, backward pass, and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update learning rate scheduler
        lr_scheduler_warmup.step()
        global_stepV3 += 1

        # Check for warmup completion
        if not warmup_doneV3 and global_stepV3 >= num_warmup_steps:
            print(f"Warmup completed at step {global_stepV3}")
            warmup_doneV3 = True

        # Early stopping in case of NaN loss
        if torch.isnan(loss):
            print("Loss is nan, stopping training.")
            break

        # Regression MSE
        mse = torch.nn.functional.mse_loss(dynamic_response, floor_sequence, reduction='sum').item()
        train_mse_regression += mse / floor_sequence.numel()

    # Average the accumulated loss and classification accuracy over all batches
    train_loss /= len(dataloader)
    train_acc_classification /= len(dataloader)
    train_mse_regression /= len(dataloader)

    return train_loss, train_acc_classification, train_mse_regression

# 2. validation_step
def validation_step_set3(model: torch.nn.Module,
                         dataloader: torch.utils.data.DataLoader,
                         loss_fn_classification: torch.nn.Module,
                         loss_fn_regression: torch.nn.Module,
                         loss_fn_weight_classification: float,
                         device: torch.device) -> Tuple[float, float, float]:
    
    """Train step for SeismicTransformer V3.0
    
    Args:
        model: target model you want to train
        dataloader: dataloader of training data
        loss_fn_classification: loss function for the classification
        loss_fn_regression: loss function for the regression
        loss_fn_weight_classification: weight for the classification loss
        device: training decive (e.g. cuda for nvidia, mps for mac)
    """
    
    model.eval()
    val_loss, val_acc_classification, val_mse_regression = 0.0, 0.0, 0.0

    # inference mode
    with torch.inference_mode():
        for _, (gm_sequence, floor_sequence, label) in enumerate(dataloader):

            batch_size = gm_sequence.size(0)

            # for test reason, seq_len should be parameterized
            key_padding_mask = CreateKeyPaddingMask_AllFalse(batch_size=batch_size, seq_len=14)
            attn_mask = CreateLookAheadMask(seq_len=12)

            # Move data to device
            gm_sequence = gm_sequence.to(device)
            label = label.to(device).long()         # label need to be long() type to run CrossEntropyLoss()
            floor_sequence = floor_sequence.to(device)
            key_padding_mask = key_padding_mask.to(device)
            attn_mask = attn_mask.to(device)

            # Forward pass
            damage_state_pred, dynamic_response = model(encoder_input=gm_sequence,
                                                        decoder_input=floor_sequence,
                                                        key_padding_mask=key_padding_mask,
                                                        attn_mask=attn_mask,
                                                        teacher_forcing_ratio=0.0)      # can ignore

            # Calculate classification accuracy
            y_pred_class = torch.argmax(torch.softmax(damage_state_pred, dim=1), dim=1)
            y_pred_class = y_pred_class.float()
            val_acc_classification += (y_pred_class == label).sum().item() / label.size(0)
            
            # Calculate classification and regression losses
            loss_classification = loss_fn_classification(damage_state_pred, label)
            loss_regression = loss_fn_regression(dynamic_response, floor_sequence)
            loss = loss_fn_weight_classification * loss_classification + \
                   (1 - loss_fn_weight_classification) * loss_regression
            val_loss += loss.item()
            
            # Regression MSE
            mse = torch.nn.functional.mse_loss(dynamic_response, floor_sequence, reduction='sum').item()
            val_mse_regression += mse / floor_sequence.numel()

    val_loss /= len(dataloader)
    val_acc_classification /= len(dataloader)
    val_mse_regression /= len(dataloader)

    return val_loss, val_acc_classification, val_mse_regression

# 3. train_set3(): combining the train_step() and validation_step()
def train_set3(model: torch.nn.Module,
               train_loader: torch.utils.data.DataLoader,
               val_loader: torch.utils.data.DataLoader,
               loss_fn_classification: torch.nn.Module,
               loss_fn_regression: torch.nn.Module,
               loss_fn_weight_classification: float,
               optimizer: torch.optim.Optimizer,
               lr_scheduler_warmup: torch.optim.lr_scheduler.LambdaLR,
               lr_scheduler_decay: torch.optim.lr_scheduler.ReduceLROnPlateau,
               num_warmup_steps: int,
               num_epochs: int,
               device: torch.device,
               log_filename: str) -> Dict[str, List]:
    
    """train() loop for SeismicTransformer V3.0
    
    Args:
        model: target model you want to train
        train_loader: dataloader of training data
        val_loader: dataloader of validation data
        loss_fn_classification: loss function for the classification
        loss_fn_regression: loss function for the regression
        loss_fn_weight_classification: weight for the classification loss
        optimizer: optimizer for the model
        lr_scheduler_warmup: learning rate scheduler for warmup
        lr_scheduler_decay: learning rate scheduler for decay
        num_warmup_steps: number of warmup steps
        num_epochs: number of epochs
        device: training decive (e.g. cuda for nvidia, mps for mac)
        teacher_forcing_ratio: how many times the model use the ground truth as input
        log_filename: log file name
    """

    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "train_mse": [],
               "validation_loss": [],
               "validation_acc": [],
               "validation_mse": [],
               "is_nan": []
    }

    # epoch
    for epoch in tqdm(range(num_epochs)):

        # teacher_forcing_ratio decay
        # teacher_forcing_ratio = (1 - (epoch / num_epochs)) / 2
        teacher_forcing_ratio = (1 - (epoch / num_epochs))
        print(f"Epoch 00{epoch} : teacher forcing ratio = {teacher_forcing_ratio}")

        # train step
        train_loss, train_acc, train_mse = train_step_set3(model, 
                                                           train_loader,
                                                           loss_fn_classification,
                                                           loss_fn_regression,
                                                           loss_fn_weight_classification,
                                                           optimizer,
                                                           lr_scheduler_warmup,
                                                           num_warmup_steps,
                                                           teacher_forcing_ratio,
                                                           device)
        
        # if train loss = nan, break
        if math.isnan(train_loss):
            print(f"Epoch {epoch}:Train loss is NaN. Stopping training.")
            results["is_nan"].append("yes")
            break
        
        # validation step
        val_loss, val_acc, val_mse = validation_step_set3(model, 
                                                          val_loader,
                                                          loss_fn_classification,
                                                          loss_fn_regression,
                                                          loss_fn_weight_classification,
                                                          device)
        
        # if validation loss = nan, break
        if math.isnan(val_loss):
            print(f"Epoch {epoch}:Validation loss is NaN. Stopping training.")
            results["is_nan"].append("yes")
            break


        # put validation_loss to lr_scheduler
        # lr_scheduler_decay.step(val_loss)


        # update log file(csv file), need to modify and re-import
        LogEpochDataV3(epoch=epoch,
                     train_loss=train_loss,
                     train_acc=train_acc,
                     train_mse=train_mse,
                     validation_loss=val_loss,
                     validation_acc=val_acc,
                     validation_mse=val_mse,
                     log_filename=log_filename)

        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"train_mse: {train_mse:.4f} | "
          f"validation_loss: {val_loss:.4f} | "
          f"validation_acc: {val_acc:.4f} | "
          f"validation_mse: {val_mse:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["train_mse"].append(train_mse)
        results["validation_loss"].append(val_loss)
        results["validation_acc"].append(val_acc)
        results["validation_mse"].append(val_mse)

    return results

# 4. test_set3(): evaluate the model using test data
def test_set3(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              device: torch.device) -> Tuple[Dict[str, float], Tuple[np.ndarray, np.ndarray]]:
    """
    
    for test:
    Cat all the predictions and true values, then compare them all at once, not one by one.
    As long as the length of the two arrays is the same and fix (3000 for example), this method can work.

    Args:
        model: target model you want to test
    """

    model.to(device)
    model.eval()  # Set the model to evaluation mode

    # Initialize the correct predictions count and all predictions for classification
    correct_label_preds = 0
    total_label_preds = 0

    y_label_preds = []
    y_label_trues = []

    # Initialize lists for dynamic response predictions and true values
    dynamic_response_preds = []
    dynamic_response_trues = []
    
    with torch.inference_mode():
        
        for _, (gm_sequence, label, floor_sequence, key_padding_mask, attn_mask) in enumerate(dataloader):

            gm_sequence = gm_sequence.to(device)
            label = label.to(device)
            floor_sequence = floor_sequence.to(device)
            key_padding_mask = key_padding_mask.to(device)
            attn_mask = attn_mask.to(device)

            # Forward pass
            damage_state_pred, dynamic_response = model(encoder_input=gm_sequence,
                                                        decoder_input=floor_sequence,
                                                        key_padding_mask=key_padding_mask,
                                                        attn_mask=attn_mask)
            
            # Calculate the correct predictions count (classification accuracy)
            _, predicted_labels = torch.max(damage_state_pred, dim=1)
            correct_label_preds += (predicted_labels == label).sum().item()
            total_label_preds += label.size(0)

            y_label_preds.append(predicted_labels.cpu())
            y_label_trues.append(label.cpu())

            # Append dynamic response predictions and true values
            dynamic_response_preds.append(dynamic_response.cpu())
            dynamic_response_trues.append(floor_sequence.cpu())
    
    # Convert predictions and actual labels from list to single tensor
    y_label_preds_tensor = torch.cat(y_label_preds)
    y_label_trues_tensor = torch.cat(y_label_trues)

    # convert to numpy array for classification metrics
    y_label_preds_numpy = y_label_preds_tensor.numpy()
    y_label_trues_numpy = y_label_trues_tensor.numpy()

    # Calculate overall accuracy for classification
    test_acc = correct_label_preds / total_label_preds if total_label_preds > 0 else 0.0

    # Calculate F1 score for classification
    test_f1 = f1_score(y_label_trues_numpy, y_label_preds_numpy, average='macro')

    # Calculate Recall score for classification
    test_Recall = recall_score(y_label_trues_numpy, y_label_preds_numpy, average='macro')

    # Convert dynamic response predictions and actual values from list to single tensor
    dynamic_response_preds_tensor = torch.cat(dynamic_response_preds)
    dynamic_response_trues_tensor = torch.cat(dynamic_response_trues)

    # Convert to numpy array for MSE calculation
    dynamic_response_preds_numpy = dynamic_response_preds_tensor.numpy()
    dynamic_response_trues_numpy = dynamic_response_trues_tensor.numpy()

    # Calculate MSE for dynamic response predictions
    test_mse = mean_squared_error(dynamic_response_trues_numpy, dynamic_response_preds_numpy)

    # Build the results dictionary
    results = {
        'test_accuracy': test_acc,
        'f1_score': test_f1,
        'recall_score': test_Recall,
        'mse_dynamic_response': test_mse,
    }

    # Return results and prediction values
    return results, (y_label_preds_numpy, y_label_trues_numpy)