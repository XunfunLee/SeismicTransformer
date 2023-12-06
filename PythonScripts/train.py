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
from sklearn.metrics import f1_score
import math
from .utility import LogEpochData

# 1. train.step()
def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
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
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device, always GPU
        X, y = X.to(device), y.to(device)

        # Forward pass
        y_pred = model(X)

        # Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # Optimizer zero grad
        optimizer.zero_grad()

        # Loss backward
        loss.backward()

        # Optimizer step
        optimizer.step()


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
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            validation_pred_logits = model(X)

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
          lr_scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
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
        lr_scheduler.step(validation_loss)

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

    with torch.inference_mode():  # Disable gradient computation for evaluation
        for X_batch, y_batch in tqdm(dataloader, desc="Evaluating"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            y_logit = model(X_batch)

            # Predictions
            _, predicted_labels = torch.max(y_logit, 1)

            # Count correct predictions
            correct_preds += (predicted_labels == y_batch).sum().item()
            total_preds += y_batch.size(0)

            # Save predictions and actual labels for metrics calculations
            y_preds.append(predicted_labels.cpu())
            y_trues.append(y_batch.cpu())

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

    # Build the results dictionary
    results = {
        'test_accuracy': test_accuracy,
        'f1_score': f1
    }

    # Return results and prediction values
    return results, (y_pred_numpy, y_true_numpy)
