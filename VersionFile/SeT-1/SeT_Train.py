"""
Contains all functionality for Seismic Transformer traning.

Author: Jason Jiang (Xunfun Lee)
Date: 2023.11.30
"""
###### ---------------------- Scriptes ---------------------- ######
import argparse
import time

# Create argument parser object
parser = argparse.ArgumentParser(description='Process some hyperparameters.')

# Add hyper-parameter
parser.add_argument('--patch_size', type=int, default=250, help='patch size')
parser.add_argument('--hidden_size', type=int, default=768, help='hidden size')
parser.add_argument('--num_layer', type=int, default=4, help='number of layers')
parser.add_argument('--num_head', type=int, default=12, help='number of heads')
parser.add_argument('--batch_size', type=int, default=1972, help='number of batches')
parser.add_argument('--epoch', type=int, default=20, help='number of epochs')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
parser.add_argument('--mlp_dropout', type=float, default=0.1, help='mlp dropout')

# run the parser parameters
args = parser.parse_args()

###### -------------------------------------------------------------- ######

###### ---------------------- Hyper-Parameters ---------------------- ######
# Data parameter
DATA_SOURCE = "50Hz_60s"
PATCH_SIZE = args.patch_size                # PS
NUM_OF_CLASSES = 5
CLASS_NAMES = ["0", "1", "2", "3", "4"]

# Transformer parameter
HIDDEN_SIZE = args.hidden_size              # HS
NUM_LAYER = args.num_layer                      # L
NUM_HEAD = args.num_head                    # H
DROPOUT_ATTENTION:float = 0.
DROPOUT_MLP:float = args.mlp_dropout
DROPOUT_EMBED:float = 0.

# Training parameter
BATCH_SIZE = args.batch_size
NUM_EPOCH = args.epoch
LEARNING_RATE = args.learning_rate           # LR
WEIGHT_DECAY = args.weight_decay              # WD
TRAIN_RATIO = 0.8                  # split the data into training dataset and validation dataset

# Saving parameter
SAVE_MODE = "model"             # "model", "params", "both"

###### ---------------------- Check if the size is correct -------------------------- ######

assert HIDDEN_SIZE % NUM_HEAD == 0, f"Hidden size ({HIDDEN_SIZE}) must be a multiple of the number of head ({NUM_HEAD})."
assert HIDDEN_SIZE > PATCH_SIZE, f"Hidden size ({HIDDEN_SIZE}) must be greater than patch size ({NUM_HEAD})."

###### -------------------------------------------------------------- ######

import torch
import time
from PythonScripts.utility import SetDevice, SaveModel, RecordResults, CountNumOfTraining, CreateOutputFolder, CreateLogFile
from PythonScripts.data_preparation import LoadData, H5pyToTensor, CreateDataLoader, CreateDataLoadersWithMultiDataset, MaskingData, TransformMaskData
from PythonScripts.data_preparation import MaskingDataFake      # create fake mask data, for comparison
from PythonScripts.embedding import PatchEmbedding, ProjectionModule
from PythonScripts.transformer import SeismicTransformer
from PythonScripts.train import train, test
from PythonScripts.visualization import SaveLossAccCurves, SaveConfusionMatrix, SaveAttnHeatMap, SavePosiHeadMap, SaveAttnHeatMapBarChart, SavePosiSimilarity
from transformers import get_linear_schedule_with_warmup

# Set GPU first
device = SetDevice()

# Print the basic parameter of the training.
print(
    f"Training info: \n"
    f"Device: {device} | "
    f"GM: {DATA_SOURCE} | "
    f"Patch Size: {PATCH_SIZE} | "
    f"Hidden Size: {HIDDEN_SIZE} | "
    f"Layer: {NUM_LAYER} | "
    f"Head: {NUM_HEAD} | "
    f"Epoch: {NUM_EPOCH} | "
    f"Learning Rate: {LEARNING_RATE} | "
    f"Weight Decay: {WEIGHT_DECAY} | "
    f"Dropout mlp: {DROPOUT_MLP}"
)

###### ---------------------- 1. Data Preparation ---------------------- ######

file_name = "cnn_[0.1-dif-20_57spp]_center-50Hz-60s_Mode0_x10_balBySameInd.mat"
traindata_path, valdata_path, testdata_path = LoadData(time_series=DATA_SOURCE,
                                                       file_name=file_name)
train_data, train_labels, validation_data, validation_labels, test_data, test_labels = H5pyToTensor(traindata_path=traindata_path,
                                                                                                    valdata_path=valdata_path,
                                                                                                    testdata_path=testdata_path,
                                                                                                    transpose=True)
# Get the whole points of a single ground motion
GM_Length = train_data.size(1)         # train_data = torch.size([num_of_gms, *length_of_each_gm*, 1])

# Calculate the number of patches
Num_Of_Patch = int(GM_Length / PATCH_SIZE)

# Generate the mask for the dataset instead of using the padding 0
# traindata_mask_list = MaskingData(input_array=train_data.numpy(), min_length=PATCH_SIZE, factor=50)
# validationdata_mask_list = MaskingData(input_array=validation_data.numpy(), min_length=PATCH_SIZE, factor=50)
# testdata_mask_list = MaskingData(input_array=test_data.numpy(), min_length=PATCH_SIZE, factor=50)

# Generate the fake mask which all is not padding, for comparison ------------------------------------
traindata_mask_list = MaskingDataFake(input_array=train_data.numpy())
validationdata_mask_list = MaskingDataFake(input_array=validation_data.numpy())
testdata_mask_list = MaskingDataFake(input_array=test_data.numpy())
# --------------------------------------------------------------------------------------------------

# Switching the mask data from [3000] to patches [12+1] with class token infomation
# e.g. [60940, 3000] --> [60940, 13]
traindata_mask_list_1D = TransformMaskData(mask_data=traindata_mask_list,
                                           num_of_patch=Num_Of_Patch,
                                           patch_size=PATCH_SIZE)
validationdata_mask_list_1D = TransformMaskData(mask_data=validationdata_mask_list,
                                           num_of_patch=Num_Of_Patch,
                                           patch_size=PATCH_SIZE)
testdata_mask_list_1D = TransformMaskData(mask_data=testdata_mask_list,
                                           num_of_patch=Num_Of_Patch,
                                           patch_size=PATCH_SIZE)

## train_data : [num_of_gms, length_of_each_gm] --> torch.tensor
## train_labels : [num_of_labels, 1] --> torch.tensor
## traindata_mask_list : [num_of_gms, length_of_each_gm] --> list

###### ---------------------- 2. Dataloader ---------------------- ######
# ground motion data: add an extra dimension to match the model input
# [num_of_gms, length_of_each_gm] ----> [num_of_gms, length_of_each_gm, 1]
train_data = train_data.unsqueeze(2)                        # [number_of_gm, 3000] --> [number_of_gm, 3000, 1]
validation_data = validation_data.unsqueeze(2)  
test_data = test_data.unsqueeze(2)

# label: remove an extra dimension to match the model input, and make it to int64 for loss function
# [num_of_labels, 1] ----> [num_of_labels]
train_labels = torch.squeeze(train_labels).long()             # [number_of_gm, 1] --> [number_of_gm]
validation_labels = torch.squeeze(validation_labels).long()
test_labels = torch.squeeze(test_labels).long()

# Create dataset list, combine train and validation dataset
dataset_list = [train_data, validation_data]                            # train_data.shape = [num_of_gms, 3000, 1]
labelset_list = [train_labels, validation_labels]                       # train_label.shape = [num_of_gms]
maskset_list_1D = [traindata_mask_list_1D, validationdata_mask_list_1D]          # traindata_mask_list = [number_of_gm, 3000]

# Create dataloader (from dataset list), aiming to split the dataset into training dataset and validation dataset
# it is not very neccesary to create a custom dataset to do this, it is just helpful to split the dataset, but not neccesary!!!
train_dataloader, validation_dataloader = CreateDataLoadersWithMultiDataset(data_list=dataset_list,
                                                                            label_list=labelset_list,
                                                                            mask_list=maskset_list_1D,
                                                                            train_ratio=TRAIN_RATIO,
                                                                            batch_size=BATCH_SIZE)

#################################################################### testing 
# for batch, (data, label, mask) in enumerate(train_dataloader):
#     print(f"train data - Batch: {batch} | Data: {data.shape} | Label: {label.shape} | Mask: {mask.shape}")
#################################################################### testing

# Create dataloader (from single dataset)
# train_dataloader = CreateDataLoader(train_data, train_labels, BATCH_SIZE)
# validation_dataloader = CreateDataLoader(validation_data, validation_labels, BATCH_SIZE)
test_dataloader = CreateDataLoader(input_data=test_data, 
                                   labels=test_labels, 
                                   mask=testdata_mask_list_1D,
                                   batch_size=BATCH_SIZE)
print(f"Length of test dataset({len(test_data)})")

#################################################################### testing
# for batch, (data, label, mask) in enumerate(test_dataloader):
#     print(f"test data - Batch: {batch} | Data: {data.shape} | Label: {label.shape} | Mask: {mask.shape}")
#################################################################### testing

###### ---------------------- 3. Split data into patches and embedding ---------------------- ######

# check if the patch_size is greater than the length of the gm
assert PATCH_SIZE <= GM_Length, f"Patch size ({PATCH_SIZE}) must be smaller than length of the ground motion ({GM_Length})."

Num_Of_Classes = train_labels.size()

# Patch Embedding
patch_embedding_layer = PatchEmbedding(Num_Of_Patch, PATCH_SIZE)

# Projection
projection_layer = ProjectionModule(input_size=PATCH_SIZE, output_size=HIDDEN_SIZE)


###### ---------------------- 4. Build model ---------------------- ######

# Create a Seismic transformer instance
SeT = SeismicTransformer(len_of_gm=GM_Length,
                        patch_size=PATCH_SIZE,
                        num_transformer_layers=NUM_LAYER,
                        embedding_dim=HIDDEN_SIZE,
                        mlp_size=HIDDEN_SIZE*4,
                        num_heads=NUM_HEAD,
                        attn_dropout=DROPOUT_ATTENTION,
                        mlp_dropout=DROPOUT_MLP,
                        embedding_dropout=DROPOUT_EMBED,
                        num_classes=NUM_OF_CLASSES)


###### ---------------------- Utility ---------------------- ######

# Count the number of csv file "training_results.csv"
num_of_training = CountNumOfTraining()

# Create output folder
save_dir = CreateOutputFolder(num_of_training=num_of_training+1,
                              hidden_size=HIDDEN_SIZE,
                              num_of_layer=NUM_LAYER,
                              num_of_head=NUM_HEAD,
                              num_of_epoch=NUM_EPOCH)

# Create log file, a csv file                  
log_filename = CreateLogFile(save_dir=save_dir)


###### ---------------------- 5. Train ---------------------- ######

# Setup the loss function for multi-class classification, can be define everywhere just before the train()
loss_fn = torch.nn.CrossEntropyLoss()

# Define the optimizer for the SeT_Train
optimizer = torch.optim.Adam(params=SeT.parameters(), 
                             lr=LEARNING_RATE,
                             betas=(0.9, 0.999),
                             weight_decay=WEIGHT_DECAY)

# Set up the learning rate warmup scheduler, work inside train_step(), warmup_ratio usually is 0.1 or 0.06
num_training_steps = 10000          # 10k steps

num_warmup_steps = num_training_steps * 0.06  # warmup_ratio usually is 0.1 or 0.06
lr_scheduler_warmup = get_linear_schedule_with_warmup(optimizer,
                                                      num_warmup_steps=num_warmup_steps,
                                                      num_training_steps=num_training_steps)

# Set up the learning rate scheduler for decay, work inside train()
lr_scheduler_decay = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                mode='max', # set the min or max lr
                                                                patience=5, # how many epoch loss don't change
                                                                factor=0.1, # new_lr = old_lr * factor
                                                                threshold=0.1, # loss change
                                                                threshold_mode='rel', # compare mode
                                                                cooldown=5, # how many epoch to wait
                                                                min_lr=1e-7, # minimun of lr
                                                                verbose=True)     # print something if useful

# Caculate the start time of the training
strat_time = time.time()

# train the model
results = train(model=SeT,
                train_dataloader=train_dataloader,
                validation_dataloader=validation_dataloader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                epochs=NUM_EPOCH,
                lr_scheduler_warmup=lr_scheduler_warmup,
                num_warmup_steps=num_warmup_steps,
                lr_scheduler_decay=lr_scheduler_decay,
                device=device,
                log_filename=log_filename)

end_time = time.time()
total_time = end_time - strat_time
print(f"Training time: {total_time:.3f}秒")


###### ---------------------- 6. Test ---------------------- ######

# Test the model, if training loss, skip the test
if results["is_nan"] == []:
    results_test, (y_pred, y_true) = test(model=SeT, dataloader=test_dataloader, device=device)
    print(f"Test accuracy:", results_test['test_accuracy'])
    print(f"Test F1 Score: ", results_test['f1_score'])


    ###### ---------------------- 7. Save model and results ---------------------- ######
    # save the model
    SaveModel(model=SeT,
          num_of_training=num_of_training,
          hidden_size=HIDDEN_SIZE,
          num_of_layer=NUM_LAYER,
          num_of_head=NUM_HEAD,
          num_of_epoch=NUM_EPOCH,
          validation_acc=results['validation_acc'],
          f1_score=results_test['f1_score'],
          save_mode=SAVE_MODE)


    ###### ---------------------- 8. Plot and save image ---------------------- ######

    # Save (plot) the accuracy and loss curve
    SaveLossAccCurves(results=results,
                      save_dir=save_dir) 

    # Save (plot) the confusion matrix
    SaveConfusionMatrix(y_true=y_true, 
                        y_pred=y_pred, 
                        class_names=CLASS_NAMES, 
                        save_dir=save_dir)
        
    # Save (plot) the attention weights heatmap
    SaveAttnHeatMap(model=SeT,
                    save_dir=save_dir,
                    num_of_layer=NUM_LAYER,
                    annot=False)

    # Save the attention weights heat map with bar chart
    SaveAttnHeatMapBarChart(model=SeT,
                            save_dir=save_dir,
                            plot_mode="multiple")

    # Save the potisional head map
    SavePosiHeadMap(model=SeT,
                    save_dir=save_dir)
    
    # Save the positional embeding similarity
    SavePosiSimilarity(model=SeT,
                       save_dir=save_dir)

else:
    results_test = {'f1_score': "nan"}
    SAVE_MODE = "params"

###### ---------------------- Record the results ---------------------- ######
# record the training
RecordResults(GM=DATA_SOURCE,
            num_of_training=num_of_training,
            patch_size=PATCH_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_of_layer=NUM_LAYER,
            num_of_head=NUM_HEAD,
            num_of_epoch=NUM_EPOCH,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            mlp_dropout=DROPOUT_MLP,
            train_acc=results['train_acc'],
            validation_acc=results['validation_acc'],
            f1_score=results_test["f1_score"],
            recall_score=results_test["recall_score"],
            times=round(total_time, 2))

# sleep 5 minutes until next training
# time.sleep(300)

###### ------------------------- End -------------------------- ######
###### ------------------------- End -------------------------- ######
###### ------------------------- End -------------------------- ######