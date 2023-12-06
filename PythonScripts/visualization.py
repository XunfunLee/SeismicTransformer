"""
Contains functionality for visualization.
1. Plot the ground motion time series data.
2. Plot the loss curvers of training.
3. Plot the ground motion pieces after patching.
4. Plot the confusion matrix.
5. Plot the heat map of attention of the model.
6. Plot the heat map of the positional encoding of the model.

Author: Jason Jiang (Xunfun Lee)
Date: 2023.11.30
"""

import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns
from typing import Dict, List
from sklearn.metrics import confusion_matrix
import os
import torch.nn.functional as F

# 1. Plot the ground motion time series data
# 1.1 subplot for each ground motion
def PlotSubplot(gm_timeseries_array:np.ndarray,
                times:list,
                gm_index:int):
    """Plot GM subplot

    Plot each ground motion time series data.

    Args:
    gm_timeseries_array: Array contain ground motion array, supposed to be 2d （e.g. [10, 3000] means 10 ground motions with 3000 points each.)
    Times: X values of the pictures (e.g. [10, 3000] means 3000 points in X value)
    GM_Index: Which ground motion is plot?

    Returns:
    0

    """
    plt.figure(figsize=(15, 5))
    plt.plot(times, gm_timeseries_array[gm_index], marker='o', linestyle='-')
    plt.title('Acceleration over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s^2)')
    plt.grid(True)
    plt.legend(['Acceleration'])
    plt.show()

#  1.2 Plot series of the ground motion time series data
def PlotManyGM(gm_timeseries_array:np.ndarray,
               num_of_subplot:int=1,
               randomness:bool=True):
    """Plot Ground motion records

    Plot ground motion time series data.

    Args:
    GM_TimeSeries_Array: Array contain ground motion array, supposed to be 2d （e.g. [10, 3000] means 10 ground motions with 3000 points each.)
    Random: Is pick the ground motion randomly from ground motion array
    Num_of_subplot: How many ground motion you want to plot anyway?

    Returns:
    0

    """
    assert gm_timeseries_array != [], "GM_TimeSeries_Array is empty, please check the data."
    assert gm_timeseries_array.ndim == 2, "GM_TimeSeries_Array need to be 2D. [10, 3000] means there are 10 ground motion with 3000 points each."

    # X value of the plot
    times = []
    for i in range(0, len(gm_timeseries_array[0])):
        times.append(i)
        i = i + 0.02

    # number of subplot to plot
    for _ in range(0, num_of_subplot):
        if randomness == True:
            gm_index = random.randint(1, len(gm_timeseries_array))
        else:
            gm_index = 1
        PlotSubplot(gm_timeseries_array=gm_timeseries_array, times=times, gm_index=gm_index+100)

# 1.3 Plot just one ground motion 
def PlotGM(gm_timeseries:np.ndarray):
    """Plot one Ground motion records

    Args:
    gm_timeseries_array: Array contain ground motion array

    """
    times = []
    for i in range(0, len(gm_timeseries)):
        times.append(i)
        i = i + 0.02
    
    plt.figure(figsize=(15, 5))
    plt.plot(times, gm_timeseries, marker='o', linestyle='-')
    plt.title('Acceleration over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s^2)')
    plt.grid(True)
    plt.legend(['Acceleration'])
    plt.show()

# 2. Save(plot) the loss curve function
def plot_loss_curves(results: Dict[str, List],
                    plot_mode: bool,
                    save_dir: str):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "validation_loss": [...],
             "validation_acc": [...]}
        plot_mode: if plot the image or not
        save_dir: save directory
    """
    loss = results["train_loss"]
    validation_loss = results["validation_loss"]

    accuracy = results["train_acc"]
    validation_accuracy = results["validation_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, validation_loss, label="validation_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, validation_accuracy, label="validation_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    
    # save the image
    plt.savefig(os.path.join(save_dir, "Acc_Loss_Curve.png"))
    
    # show the image
    # if plot_mode == True:
        # plt.show()

# 4. Save(plot) confusion matrix
def PlotConfusionMatrix(y_true: np.ndarray,
                        y_pred: np.ndarray,
                        class_names: list,
                        plot_mode: bool,
                        save_dir: str):
    """Plots training curves of a results dictionary.

    Args:
        y_true: numpy array from true result
        y_pred: numpy array from prediction result
        class_names: class name
        plot_mode: if plot the image or not
        save_dir: save directory
    """
    # caculate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # seaborn to draw confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)

    # title and labels
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    # save the image
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))

    # show the image
    # if plot_mode == True:
        # plt.show()

# 5. Plot the heat map of the attention weights
def PlotAttnHeatMap(attn_weights_list):
    

    # Pick the final weights to 
    return 0