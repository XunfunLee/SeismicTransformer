"""
Contains all functionality for Seismic Transformer traning (version 4).

Author: Jason Jiang (Xunfun Lee)
Date: 2023.02.03
"""

###### ---------------------- Scriptes ---------------------- ######
import argparse
import time

# Create argument parser object
parser = argparse.ArgumentParser(description='Process some hyperparameters.')

# Add hyper-parameter
parser.add_argument('--patch_size', type=int, default=250, help='patch size')
parser.add_argument('--hidden_size', type=int, default=768, help='hidden size')
parser.add_argument('--num_layer', type=int, default=12, help='number of layers')
parser.add_argument('--num_head', type=int, default=12, help='number of heads')
parser.add_argument('--batch_size', type=int, default=640, help='number of batches')
parser.add_argument('--epoch', type=int, default=30, help='number of epochs')
parser.add_argument('--cls_weight', type=float, default=0.3, help='loss weight for classification task')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
parser.add_argument('--mlp_dropout', type=float, default=0.1, help='mlp dropout')
parser.add_argument('--attn_dropout', type=float, default=0.0, help='attn dropout')
parser.add_argument('--embed_dropout', type=float, default=0.0, help='embedding dropout')

# run the parser parameters
args = parser.parse_args()

###### -------------------------------------------------------------- ######

###### ---------------------- Hyper-Parameters ---------------------- ######
# Data parameter
PATCH_SIZE = args.patch_size                # PS
NUM_OF_CLASSES = 5
CLASS_NAMES = ["0", "1", "2", "3", "4"]

# Transformer parameter
HIDDEN_SIZE = args.hidden_size              # HS
NUM_LAYER = args.num_layer                      # L
NUM_HEAD = args.num_head                    # H
DROPOUT_ATTENTION:float = args.attn_dropout
DROPOUT_MLP:float = args.mlp_dropout
DROPOUT_EMBED:float = args.embed_dropout

# Training parameter
BATCH_SIZE = args.batch_size
NUM_EPOCH = args.epoch
LEARNING_RATE = args.learning_rate           # LR
WEIGHT_DECAY = args.weight_decay              # WD
TRAIN_RATIO = 0.8                  # split the data into training dataset and validation dataset
CLASSIFICATION_WEIGHT = args.cls_weight

# Saving parameter
SAVE_MODE = "model"             # "model", "params", "both"

###### ---------------------- Check if the size is correct -------------------------- ######

assert HIDDEN_SIZE % NUM_HEAD == 0, f"Hidden size ({HIDDEN_SIZE}) must be a multiple of the number of head ({NUM_HEAD})."
assert HIDDEN_SIZE > PATCH_SIZE, f"Hidden size ({HIDDEN_SIZE}) must be greater than patch size ({NUM_HEAD})."

###### -------------------------------------------------------------- ######

import torch
from torch.utils.data import DataLoader, random_split
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import get_linear_schedule_with_warmup


from PythonScripts.data_preparation import DynamicDatasetV1, custom_collate
from PythonScripts.utility import SetDevice, CreateOutputFolderV4, CreateLogFileV3, CountNumOfTraining, SaveModelV4
from PythonScripts.transformer import SeismicTransformerV4, FocalLoss
from PythonScripts.train import train_set4


device = SetDevice()


###### ---------------------- data preparation -------------------------- ######

# initialize the dataset
gm_file_path = 'D:\\SeismicTransformerData\\All_GMs\\GMs_knet_3474_AF_57.h5'
building_files_dir = 'D:\\SeismicTransformerData\\SeT-4.0'

# create dataset
dataset = DynamicDatasetV1(gm_file_path=gm_file_path, 
                           building_files_dir=building_files_dir, 
                           device=device)

# calculate the size of training and validation dataset
train_size = int(TRAIN_RATIO * len(dataset))
validation_size = len(dataset) - train_size

# random split the dataset into training and validation dataset
train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

# create dataloader
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)
validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)

###### -------------------------------------------------------------- ######


###### ---------------------- build the model -------------------------- ######

# loss_fn_classification = CrossEntropyLoss()
weight_each_class = torch.tensor([0.3316, 0.1195, 0.1453, 0.1947, 0.2088]).to(device=device)
loss_fn_classification = FocalLoss(weight=weight_each_class, gamma=2.0, reduction='mean')
loss_fn_regression = MSELoss()

model = SeismicTransformerV4(len_gm=3000,
                              patch_size=PATCH_SIZE,
                              hidden_size=HIDDEN_SIZE,
                              num_heads=NUM_HEAD,
                              num_layers=NUM_LAYER,
                              dropout_attn=DROPOUT_ATTENTION,
                              dropout_mlp=DROPOUT_MLP,
                              dropout_embed=DROPOUT_EMBED,
                              num_of_classes=NUM_OF_CLASSES).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

num_training_steps = (198018*30 / BATCH_SIZE) * NUM_EPOCH         # total steps = len(train_dataset) / batch_size * epochs
num_warmup_steps = num_training_steps * 0.2  # warmup_ratio usually is 20% of the total steps


lr_scheduler_warmup = get_linear_schedule_with_warmup(optimizer,
                                                      num_warmup_steps=num_warmup_steps,
                                                      num_training_steps=num_training_steps)

# Set up the learning rate scheduler for decay, work inside train()
lr_scheduler_decay = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                mode='max', # set the min or max lr
                                                                patience=3, # how many epoch loss don't change
                                                                factor=0.1, # new_lr = old_lr * factor
                                                                threshold=0.1, # loss change
                                                                threshold_mode='rel', # compare mode
                                                                cooldown=3, # how many epoch to wait
                                                                min_lr=1e-7, # minimun of lr
                                                                verbose=True)     # print something if useful

###### -------------------------------------------------------------- ######

###### ---------------------- Training -------------------------- ######

# Count the number of csv file "training_results.csv"
num_of_training = CountNumOfTraining()

# Create output folder
save_dir = CreateOutputFolderV4(num_of_training=num_of_training+1,
                              hidden_size=HIDDEN_SIZE,
                              num_of_layer=NUM_LAYER,
                              num_of_head=NUM_HEAD,
                              num_of_epoch=NUM_EPOCH)

# Create log file, a csv file                  
log_filename = CreateLogFileV3(save_dir=save_dir)

strat_time = time.time()

# Train the model
results = train_set4(model=model,
                    train_loader=train_dataloader,
                    val_loader=validation_dataloader,
                    loss_fn_classification=loss_fn_classification,
                    loss_fn_regression=loss_fn_regression,
                    loss_fn_weight_classification=CLASSIFICATION_WEIGHT,
                    optimizer=optimizer,
                    lr_scheduler_warmup=lr_scheduler_warmup,
                    lr_scheduler_decay=lr_scheduler_decay,
                    num_warmup_steps=num_warmup_steps,
                    num_epochs=NUM_EPOCH,
                    device=device,
                    log_filename=log_filename)

end_time = time.time()
total_time = end_time - strat_time
print(f"Training time: {total_time:.3f}秒")

###### -------------------------------------------------------------- ######

###### ---------------------- Save the model -------------------------- ######

# Save the model
if results["is_nan"] == []:
    ###### ---------------------- 7. Save model and results ---------------------- ######
    # save the model
    SaveModelV4(model=model,
                num_of_training=num_of_training,
                hidden_size=HIDDEN_SIZE,
                num_of_layer=NUM_LAYER,
                num_of_head=NUM_HEAD,
                num_of_epoch=NUM_EPOCH,
                validation_acc=results['validation_acc'],
                validation_mse=results['validation_mse'],
                save_mode=SAVE_MODE)