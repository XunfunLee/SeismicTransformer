"""
Contains functionality for visualization.
1. Plot the ground motion time series data. (test only)
2. Save the loss curvers of training.
3. Plot the ground motion pieces after patching.   (test only)
4. Save the confusion matrix.
5. Save the heat map of attention of the model.(without bar chart)
5.1 Save the heat map of attention of the model.(with bar chart)
6. Save the heat map of the positional embedding.
7. Save the heat map of the positional embedding similarity.

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
import torch
import pandas as pd

# 1. Plot the ground motion time series data
# 1.1 subplot for each ground motion
def PlotSubplot(gm_timeseries_array:np.ndarray,
                times:list,
                gm_index:int,
                fontsize:int=15,
                titlesize:int=18):
    """Plot GM subplot

    Plot each ground motion time series data.

    Args:
    gm_timeseries_array: Array contain ground motion array, supposed to be 2d （e.g. [10, 3000] means 10 ground motions with 3000 points each.)
    Times: X values of the pictures (e.g. [10, 3000] means 3000 points in X value)
    GM_Index: Which ground motion is plot?

    Returns:
    0

    """
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.figure(figsize=(15, 5))
    plt.plot(times, gm_timeseries_array[gm_index], marker='o', linestyle='-')

    plt.title('Acceleration over Time', fontsize=titlesize)
    plt.xlabel('Time (s)', fontsize=fontsize)
    plt.ylabel('Acceleration (m/s^2)', fontsize=fontsize)
    plt.yticks(rotation=0, fontsize=fontsize, fontname='Times New Roman')  # Adjust the font size as needed, rotation=0 ensures that the y-axis labels do not rotate and are horizontal
    plt.xticks(rotation=0, fontsize=fontsize, fontname='Times New Roman')  # Adjust the font size as needed, rotation=0 ensures that the y-axis labels do not rotate and are horizontal
    
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

# 1.4 Save the ground motion
def SaveGM(gm_timeseries:np.ndarray,
           save_dir:str):
    """Save one Ground motion records

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
    # Save the image
    plt.savefig(os.path.join(save_dir, "Example-GM.svg"), format='svg', bbox_inches='tight')
    plt.close()  # Close the plot to free memory

# 1.5 Save the ground motion pieces after patching
def SaveGMPatches(gm_timeseries:np.ndarray,
                  save_dir:str):
    """Save one Ground motion records

    Args:
    gm_timeseries_array: Array contain ground motion array

    """

        # number of sub plots = Length / patch_size
    num_subplots = 3000 / 250

    # each points of the subplot
    points_per_subplot = int(3000 // num_subplots)

    # create the figure and subplot
    fig, axes = plt.subplots(nrows=1, ncols=int(num_subplots), figsize=(20, 2), sharex=True, sharey=True)

    # create every subplot
    for i in range(int(num_subplots)):
        start_index = i * points_per_subplot
        end_index = (i + 1) * points_per_subplot if i < num_subplots - 1 else 3000
        axes[i].plot(gm_timeseries[start_index:end_index])
        axes[i].set_title(f"Subplot {i + 1}")

    # gap between subplot
    plt.tight_layout()
    # Save the image
    plt.savefig(os.path.join(save_dir, "Example-GM-Patches.svg"), format='svg', bbox_inches='tight')
    plt.close()  # Close the plot to free memory


# 2. Save(plot) the loss curve function
def SaveLossAccCurves(results: Dict[str, List],
                      save_dir: str,
                      fontsize:int=15,
                      titlesize:int=18):
    """Plots training curves of a results dictionary and saves the resulting figure.

    Args:
        results (dict): Dictionary containing lists of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "validation_loss": [...],
             "validation_acc": [...]}
        save_dir (str): Directory where to save the training curve images
        fontsize (int): Font size for the text in the plots (default=15)
        titlesize (int): Font size for the titles (default=18)
    """
    loss = results["train_loss"]
    validation_loss = results["validation_loss"]

    accuracy = results["train_acc"]
    validation_accuracy = results["validation_acc"]

    epochs = range(len(loss))

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['legend.fontsize'] = fontsize  # Set the legend font size

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="Train Loss")
    plt.plot(epochs, validation_loss, label="Validation Loss")
    plt.title("Loss", fontsize=titlesize)
    plt.xlabel("Epochs", fontsize=fontsize)
    plt.ylabel("Loss", fontsize=fontsize)
    plt.legend()
    plt.tick_params(labelsize=fontsize)  # Set the tick label font size

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="Train Accuracy")
    plt.plot(epochs, validation_accuracy, label="Validation Accuracy")
    plt.title("Accuracy", fontsize=titlesize)
    plt.xlabel("Epochs", fontsize=fontsize)
    plt.ylabel("Accuracy", fontsize=fontsize)
    plt.legend()
    plt.tick_params(labelsize=fontsize)  # Set the tick label font size

    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the image
    plt.savefig(os.path.join(save_dir, "Acc_Loss_Curve.svg"), format='svg', bbox_inches='tight')
    plt.close()  # Close the plot to free memory

# 4. Save(plot) confusion matrix
def SaveConfusionMatrix(y_true: np.ndarray,
                        y_pred: np.ndarray,
                        class_names: list,
                        save_dir: str,
                        fontsize: int = 15,
                        titlesize: int = 18):
    """Plots the confusion matrix and saves it as an SVG image.

    Args:
        y_true: numpy array of true results
        y_pred: numpy array of predicted results
        class_names: list of class names for the axis ticks
        save_dir: directory where to save the confusion matrix image
        fontsize: font size for the text in the matrix (default=15)
        titlesize: font size for the title (default=18)
    """

    # Calculate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Set the font properties
    plt.rcParams['font.family'] = 'Times New Roman'

    # Create a figure for the heatmap
    plt.figure(figsize=(10, 7))

    # Draw the heatmap and store the returned Axes object
    ax = sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
                     xticklabels=class_names, yticklabels=class_names,
                     annot_kws={"size": fontsize})

    # Set the title and labels with the specified fontsize
    plt.xlabel('Predicted', fontsize=fontsize)
    plt.ylabel('True', fontsize=fontsize)
    plt.title('Confusion Matrix', fontsize=titlesize)

    # Set the font size for the axis ticks
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    # Get the color bar object and update its tick label size
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=fontsize)

    # Save the image
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, "confusion_matrix.svg"), format='svg', bbox_inches='tight')
    plt.close()  # Close the plot to free memory

# 5. Save the heat map of the attention weights (old)
def SaveAttnHeatMap(model:torch.nn.Module,
                    save_dir:str,
                    num_of_layer:int=12,
                    fontsize:int=15,
                    titlesize:int=18,
                    annot:bool=False):
    """plot attention heat map of the model

    Args:
        model: model you want to plot
        save_dir: save directory
        num_of_row: number of row of the heat map
        num_of_col: number of column of the heat map
        figsize: size of the figure
    """
    # Get the number of layers in the model
    if num_of_layer == 12:
        num_of_row = 4
        num_of_col = 3
        figsize:tuple=(12, 14)
    elif num_of_layer == 6:
        num_of_row = 2
        num_of_col = 3
        figsize:tuple=(12, 7)
    elif num_of_layer == 4:
        num_of_row = 2
        num_of_col = 2
        figsize:tuple=(12, 10)

    attention_weights_list = model.attention_weights_list
    # find the min and max value of the attention weights
    all_weights = torch.cat([layer[0, 1:, 1:].detach() for layer in attention_weights_list])
    vmin = all_weights.min()
    vmax = all_weights.max()

    # Set the font style
    plt.rcParams['font.family'] = 'Times New Roman'

    # We want to display a heatmap without the weights corresponding to the class token, which we assume is at index 0.
    # Therefore, we select the weight matrix excluding the first row and column.
    fig, axes = plt.subplots(nrows=num_of_row, ncols=num_of_col, figsize=figsize)  # Large figure with 2 rows and 6 columns for subplots

    for layer_idx, layer_attention in enumerate(attention_weights_list):
        # Get attention weights for all heads in the current layer and remove class token
        # Assume there's only one head for now; if there are multiple heads, adjust accordingly.
        attn_matrix = layer_attention[0, 1:, 1:].detach().cpu().numpy()

        # Compute the subplot index; there are 6 subplots per row.
        row_idx = layer_idx // num_of_col
        col_idx = layer_idx % num_of_col

        # Plot the heatmap on the appropriate subplot axis.
        ax = sns.heatmap(attn_matrix, cmap='viridis', annot=annot, fmt=".2f", ax=axes[row_idx, col_idx], vmin=vmin, vmax=vmax)     # annot can decide the number of heat map
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=fontsize)
        # Set the title for the subplot.
        axes[row_idx, col_idx].set_title(f'Layer {layer_idx + 1}', fontsize=titlesize)

        # Set axis labels
        axes[row_idx, col_idx].set_xlabel('Input Sequence',fontsize=fontsize)
        axes[row_idx, col_idx].set_ylabel('Input Sequence',fontsize=fontsize)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(save_dir, "Attn_Weights.svg"), format='svg', bbox_inches='tight')

# 5.1 Save the heat map of the attention weights (new)
def SaveAttnHeatMapBarChart(model: torch.nn.Module,
                            save_dir: str,
                            plot_mode: str = 'single',
                            fontsize: int = 15,
                            titlesize: int = 18):
    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Create the subdirectory 'AttentionWeights'
    attention_weights_dir = os.path.join(save_dir, "AttentionWeights")
    if not os.path.exists(attention_weights_dir):
        os.makedirs(attention_weights_dir)

    attention_weights_list = model.attention_weights_list

    if plot_mode == "multiple":
        # for dozens of evluation plot and get the first
        trimmed_attention_matrix = [aw.squeeze(0).cpu().detach().numpy()[0, 1:, 1:] for aw in attention_weights_list]
    elif plot_mode == "single":
        # for only one evluation plot
        trimmed_attention_matrix = [aw.squeeze(0).cpu().detach().numpy()[1:, 1:] for aw in attention_weights_list]
    else:
        raise ValueError(f"Invalid plot mode: {plot_mode}, need to be 'single' or 'multiple'")

    # Set the font style
    plt.rcParams['font.family'] = 'Times New Roman'

    # Loop for all heat maps and bar charts
    for i, aw in enumerate(trimmed_attention_matrix):
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # print(attention_weights_list[0].shape[1])

        # Heat map
        heatmap = axs[0].imshow(aw, cmap='Reds', aspect='auto', vmin=0, vmax=0.5)
        cbar = fig.colorbar(heatmap, ax=axs[0])
        cbar.ax.tick_params(labelsize=fontsize)
        axs[0].set_xticks(range(attention_weights_list[0].shape[1]-1))
        axs[0].set_xticklabels(range(1, attention_weights_list[0].shape[1]), fontsize=fontsize)
        axs[0].set_yticks(range(attention_weights_list[0].shape[1]-1))
        axs[0].set_yticklabels(range(1, attention_weights_list[0].shape[1]), fontsize=fontsize)
        axs[0].set_title(f'Layer {i+1} Attention Weights', fontsize=titlesize)

        # Bar chart
        means = aw.mean(axis=0)
        axs[1].bar(range(1, attention_weights_list[0].shape[1]), means, color='darkred')
        axs[1].set_xticks(range(1, attention_weights_list[0].shape[1]))
        axs[1].set_xticklabels(range(1, attention_weights_list[0].shape[1]), fontsize=fontsize)
        axs[1].tick_params(axis='y', labelsize=fontsize)
        axs[1].set_ylim(0, 0.5)
        axs[1].set_title(f'Layer {i+1} Attention Weights Mean', fontsize=titlesize)

        plt.tight_layout()
        
        # Save each figure in the 'AttentionWeights' subdirectory
        plt.savefig(os.path.join(attention_weights_dir, f'attention_weights_layer_{i+1}.svg'), format='svg', bbox_inches='tight')
        plt.close()

# 6. Save the heat map of the positional encoding
def SavePosiHeadMap(model:torch.nn.Module,
                    save_dir:str,
                    fontsize:int=15,
                    titlesize:int=18,
                    figsize:tuple=(7, 5)):
    """save positional embedding heat map of the model

    Args:
        model: model you want to plot
        fontsize: font size of the y axis
        figsize: size of the figure

    Returns:
        SVG
    """
    # get the positional embedding
    position_embeddings = model.position_embedding.squeeze(0)
    position_embeddings = position_embeddings.detach().cpu().numpy()

    # plot the heat map
    plt.figure(figsize=figsize)
    heatmap = plt.imshow(position_embeddings, cmap='viridis', aspect='auto', vmin=-4, vmax=4)
    
    # Add a color bar at the right
    cbar = plt.colorbar(heatmap)
    cbar.ax.tick_params(labelsize=fontsize)  # Set the font size of the color bar

    # Set the main title and increase the font size
    plt.title('Position Embeddings Heatmap', fontsize=titlesize, fontname='Times New Roman')
    
    # Adjust the y-axis to make it easier to read
    plt.yticks(rotation=0, fontsize=fontsize, fontname='Times New Roman')  # Adjust the font size as needed, rotation=0 ensures that the y-axis labels do not rotate and are horizontal
    plt.xticks(rotation=0, fontsize=fontsize, fontname='Times New Roman')  # Adjust the font size as needed, rotation=0 ensures that the y-axis labels do not rotate and are horizontal
    plt.xlabel('Embedding Dimension', fontsize=fontsize)
    plt.ylabel('Sequence Position', fontsize=fontsize)

    # Save the plot
    plt.savefig(os.path.join(save_dir, "Position_Embedding.svg"), format='svg', bbox_inches='tight')

def SavePosiSimilarity(model:torch.nn.Module,
                        save_dir:str,
                        fontsize:int=15,
                        titlesize:int=18,
                        figsize:tuple=(7, 5)):
    """save positional embedding similarity map of the model

    Args:
        model: model you want to plot
        fontsize: font size of the y axis
        figsize: size of the figure

    Returns:
        SVG
    """
    # evaluation model
    model.eval()

    # positional embedding
    position_embedding = model.position_embedding

    # caculate the cosine similarity matrix
    normalized_pos_embeddings = F.normalize(position_embedding.squeeze(0), p=2, dim=-1)
    cosine_similarity_matrix = torch.mm(normalized_pos_embeddings, normalized_pos_embeddings.t()).cpu().detach().numpy()

    # create heatmap
    plt.figure(figsize=figsize)
    plt.imshow(cosine_similarity_matrix, cmap='viridis')  # viridis theme
    cbar = plt.colorbar()  # color bar
    cbar.ax.tick_params(labelsize=fontsize)  # Set the fontsize of the color bar
    # Set label and title
    plt.xlabel('Position', fontsize=fontsize)
    plt.ylabel('Position', fontsize=fontsize)
    plt.title('Position Embedding Similarity', fontsize=titlesize)

    plt.xticks(range(position_embedding.shape[0]), fontsize=fontsize)
    plt.yticks(range(position_embedding.shape[0]), fontsize=fontsize)
    plt.savefig(os.path.join(save_dir, "Position_Embedding_Similarity.svg"), format='svg')
