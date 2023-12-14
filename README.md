# Seismic Transformer Documentation

*Author: Jason Jiang (Xunfun Lee)*

*Date: 2023.12.3*

![0b042aafb6ba0395805a07402df2c27](https://github.com/XunfunLee/SeismicTransformer/assets/129706253/ffffd3d4-a447-4761-a73c-c7156d097218)

## 1. Get preparation

First you need to install pytorch for your nividia GPU *(if you are using MacOS, go to https://pytorch.org for pytorch for mps)* and create a virtual environment to install dependencies.

1. For pytorch, run `nividia-smi` and `nvcc-V` in cmd to check your driver version (*`nvcc-V` is showing the cuda toolkit driver version while `nvidia-smi` showing the graphic driver which can be different, pytorch need to match toolkit drviver, not the graphic driver*). Then go to https://pytorch.org find a proper version of pytorch to install. More information plz go to google.
2. For the dependencies, just run below code in your virtual environment in present directory:
   ```cmd
    pip install -r requirements.txt
   ```
3. After getting all requirement you need (maybe it is not all in "requirements.txt", just install what is missing), then you start your transformer journey.


## 2. Document description

### 2.1 Folder directory

#### Seisimic-Transformer(this project)
- Data: contain the time-series data of ground motions and buidling damage state. More infomation of the data is in the readme.md under "Data" folder. These data file can be seen in NAS"Z:\public folder\2020ZJ\xxx".
- PythonScripts: python script for modularization of transformer.
  - `data_preparation.py`: load data, create dataloader...
  - `embedding.py`: patch embedding, projection...
  - `train.py`: train, validation, test...
  - `transformer.py`: module in transformer, encoder for V1.0...
  - `utility.py`: save, load the model...
  - `visualization.py`: plot the loss, accuracy and confusion matrix...
- LearningSource: contains a lot of learning information of the transformer architecture.
- readme.md: this file.
- requirements.txt: dependencies of this project.
- `SeT_Base.ipynb`: *See more detailed introduction below.*
- `SeT_Modular(out_of_date).ipynb`: *See more detailed introduction below.*
- `SeT_Train.py`: *See more detailed introduction below.*
- `SeT_Train_Factory.py`: *See more detailed introduction below.*

### 2.2 Detailed information of file

#### `SeT_Base.ipynb` (**Step 1**): for **beginner** to know how transformer works basically

This is a python notebook for the beginner to learn transformer architecture. This notebook don't rely on other python scripts, all function and class come from pytorch library or defined inside the block.

#### `SeT_Modular.ipynb` (**Step 2**): for **master** to know how transformer works after modularization

This notebook is an upgrade of `SeT_Base.ipynb`, putting all functions, classes into python scripts by modularization, getting ready for training within one line of code. But it can't work due to the change of the python scripts.

#### `SeT_Train.py` (**Step 3**): for **professor** to train your model in a single command line

This python script contains all the process from data preparation to save the model. Functions and classes definition is under "PythonScripts" folder. Using one line of code to run:

```cmd
python SeT_Train.py --patch_size 250 --hidden_size 384 --num_layer 4 --num_head 4 --batch_size 64 --epoch 2 --learning_rate 0.001 --weight_decay 0.1 --mlp_dropout 0.1
```

#### `SeT_Train_Factory.py` (**Step 4**): for **high level professor** to train multiple model in a single command line

A modular and automatic scripts to run batches of training, define different params combinition to run dozens of model in just one line of code.

```cmd
python SeT_Train_Factory.py
```

`batch_size`: according to your GPU VRAM, for `RTX-3090ti-24G`, can reach 1972 `batch_size` max (with a `HIDDEN_SIZE` = 768, `NUM_LAYER` = `NUM_HEAD` = 12, parmas of the model is 86M). Big model with more params require more VRAM.

## 3. About this version

### Stage 1: Seismic Transformer V1.0 (100%)

**Stage 1.1**: Build Seismic Transformer on the fundation of Vision Transformer. Using *@Jie Zheng*'s data to train the based model using ground motion time-seris data. (100%)

**Stage 1.2**: Modularization from notebook to python scripts. Automation by args in single line. (100%)

**Stage 1.3**: Ploting attention weights, positional embedding to see the model's performance. Setup learning rate warmup, caculating F1 score, recall. (100%)

### Stage 2: Seismic Transformer V2.0

on coming...




------------------------------------------------------------------
