# Seismic Transformer Documentation

*Author: Jason Jiang (Xunfun Lee)*

*Created Date: 2023.12.3*

*Final Update: 2023.12.15*

![0b042aafb6ba0395805a07402df2c27](https://github.com/XunfunLee/SeismicTransformer/assets/129706253/ffffd3d4-a447-4761-a73c-c7156d097218)

## 1. Get preparation

First you need to install pytorch for your nividia GPU *(if you are using MacOS, go to https://pytorch.org for pytorch for mps)* and create a virtual environment to install dependencies.

1. For pytorch, run `nividia-smi` and `nvcc-V` in cmd to check your driver version (*`nvcc-V` is showing the cuda toolkit driver version while `nvidia-smi` showing the graphic driver which can be different, pytorch need to match toolkit drviver, not the graphic driver*). Then go to https://pytorch.org find a proper version of pytorch to install. More information plz go to google.
2. For the dependencies, just run below code in your virtual environment in present directory:
   ```cmd
    pip install -r requirements.txt
   ```
3. After getting all requirement you need (maybe it is not all in "requirements.txt", just install what is missing), then you start your transformer journey.

-----

## 2. Document description

### 2.1 Folder directory

#### Seisimic-Transformer(this project)
- Data: contain the time-series data of ground motions and buidling damage state. More infomation of the data is in the readme.md under "Data" folder. These data file can be seen in NAS"Z:\public folder\2020ZJ\xxx".
- PythonScripts: python script for modularization of transformer.
  - `data_preparation.py`: load data, create dataloader...
  - `embedding.py`: patch embedding, projection...
  - `train.py`: train, validation, test...
  - `transformer.py`: module in transformer, encoder...
  - `utility.py`: save, load the model...
  - `visualization.py`: plot the loss, accuracy and confusion matrix...
- LearningSource: contains a lot of learning information of the transformer architecture.
- readme.md: this file.
- requirements.txt: dependencies of this project. (maybe not completed)

----

## 3. SeismicTransformer Version

### 3.1 SeismicTransformer V1.0 (Released)

SeismicTransformer V1.0 (SeT-1) is the first version of SeismicTransformer. Everyone who is new to learn Transformer deep learning framework can start from this version **(release 1.0.0)**.

SeT-1 using 110k training and validation data, 1.7k test data (comes from @Jie Zheng's job). Contains the basic function of the training a completed transformer deep learning model. Including positional embedding, class token embedding, key masking, learning rate warmup, learning rate auto adjustment(decay), recall score, F1 score, attention weights(heatmap and bar chart), positional similarity.

* **input**: time-seires data of ground motion. (key mask is optional, effect is not obivious.)
* **output**: one label (5 classification task).

<img width="825" alt="SeT-1 jpg" src="https://github.com/XunfunLee/SeismicTransformer/assets/129706253/5c9a04a8-7c55-4c9b-8588-1c05d0a8557c">

#### 3.1.1 File Description

#### `SeT_Base.ipynb` (**Step 1**): for **beginner** to know how transformer works basically

This is a python notebook for the beginner to learn transformer architecture. This notebook don't rely on other python scripts, all function and class come from pytorch library or defined inside the block.

#### `SeT_Modular.ipynb` (**Step 2**): for **master** to know how transformer works after modularization

This notebook is an upgrade of `SeT_Base.ipynb`, putting all functions, classes into python scripts by modularization, getting ready for training within one line of code. But it can't work due to the change of the python scripts. Maybe some day some handsome boy will fix it :).

#### `SeT_Train.py` (**Step 3**): for **professor** to train your model in a single command line

This python script contains all the process from data preparation to save the model. Functions and classes definition is under "PythonScripts" folder. Using one line of code to run:

```cmd
python SeT_Train.py --patch_size 250 --hidden_size 768 --num_layer 12 --num_head 12 --batch_size 1972 --epoch 20 --learning_rate 0.001 --weight_decay 0. --mlp_dropout 0.1
```

> **Note**: `batch_size`: according to your GPU VRAM, for `RTX-3090ti-24G`, can reach 1972 `batch_size` max (with a `HIDDEN_SIZE` = 768, `NUM_LAYER` = `NUM_HEAD` = 12, parmas of the model is 86M). Big model with more params require more VRAM.

#### `SeT_Train_Factory.py` (**Step 4**): for **high level professor** to train multiple model in a single command line

A modular and automatic scripts to run batches of training, define different params combinition to run dozens of model in just one line of code.

```cmd
python SeT_Train_Factory.py
```

#### `Test_Laboratory.ipynb` (**Step 5**): for anyone who wants to test the model performance after training by using El-Centro ground motion as test data.

You can plot, save the attention weigths (heatmap and bar chart), positional embedding(heatmap), position similarity(heatmap) of the **El-Centro** ground motion.

#### 3.1.2 SeT-1 Conclusion

Some results of the SeT-1 model are listed below:

| **Model** | **Params** | **Accuracy (train, validation, test)** |  
| ----- | ----- | ----- |
| **SeT-1-Base** | patch_size=250, hidden_size=768, layer=12,	head=12,	epoch=20, batch_size=1972, learning_rate=0.001, weight_decay=0, dropout_mlp=0.1| 99%, 92%, 80% |
| **SeT-1-2** | patch_size=250, hidden_size=768, `layer=4`,	head=12,	epoch=20, batch_size=1972, learning_rate=0.001, weight_decay=0, dropout_mlp=0.1| 99%, 92%, 83% |
| **SeT-1-3** | patch_size=250, `hidden_size=384`, layer=12,	head=12,	epoch=20, batch_size=1972, learning_rate=0.001, weight_decay=0, dropout_mlp=0.1| 99%, 93%, 83% |

In conclusion, SeT-1 is a good start but still lack of training data. A lot of strategy such as key masking still don't increase the testing accuracy, while others can get nice effect such as warmup, learn rate decay. 

#### 3.1.3 Work in this version

**Stage 1.1**: Build Seismic Transformer on the fundation of Vision Transformer. Using *@Jie Zheng*'s data to train the based model using ground motion time-seris data. **(100% Competed)**

**Stage 1.2**: Modularization from notebook to python scripts. Automation by args in single line. **(100% Competed)**

**Stage 1.3**: Ploting attention weights, positional embedding to see the model's performance. Setup learning rate warmup, caculating F1 score, recall. **(100% Competed)**

-----

### 3.2 Seismic Transformer V2.0 (Released)

SeT-2**(v2.0.0)** is an update of SeT-1, which adding frequency information by doing ***Fast Fourier Transfromation***. After comparasion (infact is just thinking), frequency information is treated as an input of the sequences but adding two token to specify time series data and frequency data. Different from 12 patches + 1 class token of SeT-1 (forward function inside the SeT-1-base is `(batch_size, [CLS] + 12 patches time series data, 768)` = `[batch_size, 13, 768]`), while SeT-2-base has 3 tokens and 12 patches (`(batch_size, [CLS] + 12 patches time series data (with [TIME]) + 1 frequency data (with [FREQ]), 768)` = `[batch_size, 14, 768]`).

* **input**: time-seires data of ground motion. 
  * **frequency data** is needed in V2.0(will generate during the data processing) and effect is obvious from the attention weights
  * key mask is optional, effect is not obivious.
* **output**: one label (5 classification task).

![SeT-2](https://github.com/XunfunLee/SeismicTransformer/assets/129706253/99b20116-8405-4e11-82a3-87858b393d68)

#### 3.2.1 File Description

#### `set2_train.py`: training python script based on `SeT_Train.py` which can run directly.

Also can run as:

```cmd
python set2_train.py --patch_size 250 --hidden_size 768 --num_layer 12 --num_head 12 --batch_size 1972 --epoch 20 --learning_rate 0.001 --weight_decay 0. --mlp_dropout 0.1
```

#### `set2_test.ipynb`: test SeT-2 model

Test the **El-Centro** ground motion (or other example) and plot the attention weigths heatmap and bar chart to see the preformance of the model.

#### `gm_fft.ipynb`: test Fast Fourier Transformation of El-Centro ground motion

A notebook to visualize the results after Fast Fourier Transformation.

#### `set2_test.py`: test SeT-2 model

Upgrade of `set2_test.ipynb` by modularization to python script.

#### **PythonScript**

Almost every python script has been upgrade to V2.0. Classes and function in version 1.0 has been duplicated and named V2 in SeT-2. e.g. `SeismicTransformerV2`, `TransformMaskDataV2`...

#### 3.2.2 Conclusion

Some results of the SeT-2 model are listed below:

| **Model** | **Params** | **Accuracy (train, validation, test)** |  
| ----- | ----- | ----- |
| **SeT-2-Base** | patch_size=250, hidden_size=768, layer=12,	head=12,	epoch=20, batch_size=1972, learning_rate=0.001, weight_decay=0, dropout_mlp=0.1| 99%, 92%, 83% |
| **SeT-2-1** | patch_size=250, hidden_size=768, `layer=4`,	head=12,	epoch=20, batch_size=1972, learning_rate=0.001, weight_decay=0, dropout_mlp=0.1| 99%, 92%, 82% |

In conclusion, SeT-2 is just an tiny update of SeT-1 by one-week coding. Although adding frequency information doesn't increase the model performance(while the SeT-1 performance is not bad), but we can obviouly see the frequency info of the attention weights is higher than normal sequence. I am eager to jump into SeismicGPT by adding transformer decoder! See you next version.

------------------------------------------------------------------

### 3.3 SeismicTransformer V3.0 (Released)

SeT-3**(v3.0.0)** is huge update of SeT-2, which adding transformer decoder to the model. SeT-3 is a multi-task models which can not only predict the damage state, but also can predict the  dynamic response of the top floor of target building.

In SeT-3, teacher forcing, scheduled sampling are used to train the model. The process of learning the differences between training mode and inference mode after adding decoder is real tough. GPT-4 is hard to provide a big picture of the model, so you need to know all the fundation of the model, techiniques of the training process.

#### 3.3.1 File Description

#### `SeT_3_Cookbook.ipynb`: build SeT-3 from scratch
A notebook to record the process of building SeT-3, which includes all the module used in SeT-3. All modules are rewrited for the complex structure of SeT-3.

#### `SeT_3_Train.ipynb`: train SeT-3
a notebook to train the SeT-3 model. Including a lot of visualization of the training process.

#### `SeT_3_Test.ipynb`: test SeT-3
A notebook to test the SeT-3 model.

#### `SeT_3_TestDecoder.ipynb`: test decoder of SeT-3
A notebook to test the decoder module of SeT-3. Decoder block is very complex and hard to debug, so I write this notebook to test the decoder module.

#### **Python Scripts:** 

All the python scripts are rewrited for the complex structure of SeT-3. e.g. `SeismicTransformerV3`, `DecoderV1`...

#### 3.3.2 Conclusion

| **Model** | **Params** | **Validation (Acc, MSE)** |  
| ----- | ----- | ----- |
| **SeT-3-Base** | patch_size=250, hidden_size=768, layer=12,	head=12,	epoch=50, batch_size=512, | 79%, 0.60 |
| **SeT-3-Large** | patch_size=250, hidden_size=768, `layer=24`,	`head=24`,	epoch=50, batch_size=256, | 82%, 0.33 |

> note: SeT-3 (796MB) is using AdmaW optimizer with learning rate = 0.001, weight decay = 0.0, drouput_mlp = 0.1, lr_warmup = 20% of total steps. Teacher forcing ratio base = 1.0 with linear decay to 0.0.

In conclusion, SeT-3 is a fast version to adding decoder to achieve multi-task. The performance is not good enough, but the aim of SeT-3 is to accomplish the code. SeT-4 will add multi-structure infomation to train.

------------------------------------------------------------------

### 3.4 SeismicTransformer V4.0 (Released)

SeT-4 is a small update from SeT-3, by adding a series of multi-structure information to the model. The multi-structure information includes 4 parts: stories(1-10, 1), height(3-30, 3), IM(6-8, 1), struct type(framework only).

For SeT-4, focal loss is using to balance the number of each class. The model is using 132GB of data to train, approximately 6 million (5,940,540) of data. A automation method has been developed to get the training data from MDOF procedure. Dataloader has also been rewrited to load the data, as well as the embedding method. Specifically, stories and height belong to the same category, so they are using the same embedding method (`nn.Parameter()`, which used as in token embedding). IM and struct type are using the same embedding method (`nn.Embedding()`, whcih is newly method used in SeT training for type data). More details are shown in the paper.

#### 3.4.1 File Description

#### `SeT_4_Cookbook.ipynb`: build SeT-4 from scratch

A notebook to record the process of building SeT-4, which includes all the module used in SeT-4. All modules are rewrited for the complex structure of SeT-4.

#### `SeT_4_Trainbook.ipynb`: train SeT-4

A notebook to train the SeT-4 model. Including a lot of visualization of the training process.

#### `set4_train.py`: script to train SeT-4

#### `set4_train_factory.py`: script to train SeT-4 with different params

#### `sample_counter.py`: counter the number of each class in the training data

Num_samples_class.txt is the output of `sample_counter.py`, which is used to calculate the weight of each class in focal loss.

#### **Python Scripts:**

All the python scripts are rewrited for the complex structure of SeT-4. e.g. `SeismicTransformerV4`, `EncoderV2`...

#### 3.4.2 Conclusion

Each of the training in SeT-4 cost almost a week in 3090ti-24G. The performance of the model is not good enough with accuracy=66% while the MSE=0.5, you can call this bad performance :p. However, the accuracy of the model is only 66% and doesn't increase after 3 epoches. So SeT-3.5 is coming soon to solve the classification task only. After figuring out the classification task, SeT-4.1 will be back to solve the regression task with large data set. The problem maybe in the focal loss, or the whole model structure is not very good for multi-task.

------------------------------------------------------------------

### 3.5 SeismicTransformer V3.5 (working on...)

SeT-3.5 is a classification model without decoder, adding a series of multi-structure information to the model. The multi-structure information includes 4 parts: stories(1-10, 1), height(3-30, 3), IM(6~8, 1), struct type(0 or 1, for framework and frame-shear structure). Compared to SeT-2, the training data has been enlarge to approximately 6 million. Compared to SeT-4, SeT-3.5 subtract the decoder module to focus on the classification task only.

