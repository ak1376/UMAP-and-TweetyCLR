#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 21:10:36 2024

What I want to do is 

@author: akapoor
"""

import numpy as np
import torch
import sys
filepath = '/home/akapoor'
# filepath = '/Users/AnanyaKapoor'
import os
os.chdir(f'{filepath}/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_b_Canary_SSL/UMAP_and_TweetyCLR')
from util.utils_Dataset_Creation import *
from util.utils_functional import *
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle
from torch.utils.data import DataLoader, TensorDataset
import umap
import matplotlib.pyplot as plt
import torch.optim as optim
import inspect
import torch.nn.init as init
import random
from torch.utils.data import Dataset
from torch.utils.data import random_split


# Set random seeds for reproducibility 
torch.manual_seed(295)
np.random.seed(295)
random.seed(295)

# Matplotlib plotting settings
plt.rcParams.update({'font.size': 20})
plt.rcParams['figure.figsize'] = [15, 15]  # width and height should be in inches

# Specify the necessary directories 
bird_dir = f'{filepath}/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_a_Rotations/Gardner_Lab/Canary_Data/llb16_data_matrices/'
audio_files = bird_dir+'llb3_songs'
directory = bird_dir+ 'Python_Files'

# Identify the upstream location where the results will be saved. 
analysis_path = f'{filepath}/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_b_Canary_SSL/TweetyCLR_Repo/'

# # Parameters we set
num_spec = 1452
window_size = 100
stride = 100

# Define the folder name

# I want to have a setting where the user is asked whether they want to log an
# experiment. The user should also provide a brief text description of what the
# experiment is testing (like a Readme file)

log_experiment = True
if log_experiment == True:
    user_input = input("Please enter the experiment name: ")
    folder_name = f'{analysis_path}Num_Spectrograms_{num_spec}_Window_Size_{window_size}_Stride_{stride}/{user_input}'
    
else:
    folder_name = f'{analysis_path}Num_Spectrograms_{num_spec}_Window_Size_{window_size}_Stride_{stride}'


# Organize the files for analysis 
files = os.listdir(directory)
all_songs_data = [f'{directory}/{element}' for element in files if '.npz' in element] # Get the file paths of each numpy file from Yarden's data
all_songs_data.sort()

# Identity any low and high pass filtering 
masking_freq_tuple = (500, 7000)

# Dimensions of the spec slices for analysis 
spec_dim_tuple = (window_size, 151)

# Ground truth label coloring
with open(f'{filepath}/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_b_Canary_SSL/TweetyCLR_End_to_End/Supervised_Task/category_colors.pkl', 'rb') as file:
    category_colors = pickle.load(file)


# In[1]: Creating Dataset

# Object that has a bunch of helper functions and does a bunch of useful things 
simple_tweetyclr_experiment_1 = Tweetyclr(num_spec, window_size, stride, folder_name, all_songs_data, masking_freq_tuple, spec_dim_tuple, category_colors)

simple_tweetyclr = simple_tweetyclr_experiment_1

# Finds the sliding windows
# simple_tweetyclr.first_time_analysis()

# with open(f'{simple_tweetyclr.folder_name}/simple_tweetyclr.pkl', 'wb') as file:
#     pickle.dump(simple_tweetyclr, file)

with open(f'{simple_tweetyclr.folder_name}/simple_tweetyclr.pkl', 'rb') as file:
    simple_tweetyclr = pickle.load(file)

# Documentation code
if log_experiment == True: 
    exp_descp = input('Please give a couple of sentences describing what the experiment is testing: ')
    # Save the input to a text file
    with open(f'{folder_name}/experiment_readme.txt', 'w') as file:
        file.write(exp_descp)


stacked_windows = simple_tweetyclr.stacked_windows.copy()

stacked_windows.shape = (stacked_windows.shape[0], 100, 151)

stacked_windows[:, :, :] = simple_tweetyclr.stacked_labels_for_window[:, :, None]

stacked_windows.shape = (stacked_windows.shape[0],1, 100, 151) 


total_dataset, total_dataloader = create_dataloader(stacked_windows, 64, np.arange(stacked_windows.shape[0]))


# Need to compute the UMAP embedding
reducer = umap.UMAP(metric = 'cosine', random_state=295)

# embed = reducer.fit_transform(simple_tweetyclr.stacked_windows)
embed = np.load(f'{analysis_path}Num_Spectrograms_{num_spec}_Window_Size_{window_size}_Stride_{stride}/embed.npy')
# Preload the embedding 
simple_tweetyclr.umap_embed_init = embed

plt.figure()
plt.scatter(embed[:,0], embed[:,1], s = 10, c = simple_tweetyclr.mean_colors_per_minispec)
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title(f'Total Slices: {embed.shape[0]}')
plt.savefig(f'{simple_tweetyclr.folder_name}/Plots/UMAP_of_all_slices.png')
plt.show()

# np.save(f'{analysis_path}Num_Spectrograms_{num_spec}_Window_Size_{window_size}_Stride_{stride}/embed.npy', embed)

# DEFINE HARD INDICES THROUGH INTERACTION: USER NEEDS TO ZOOM IN ON ROI 

# If we are including multiple hard regions then I should put them in a 
# dictionary and then select hard and easy negative samples according to which
# key we are in in in the dictionary

hard_indices_dict, list_of_hard_indices = simple_tweetyclr.selecting_confused_region(embed)


data_for_analysis = simple_tweetyclr.stacked_windows.copy()
data_for_analysis = data_for_analysis.reshape(data_for_analysis.shape[0], 1, 100, 151)

# list_of_images = []
# for batch_idx, (data) in enumerate(total_dataloader):
#     data = data[0]
    
#     for image in data:
#         list_of_images.append(image)
        
# list_of_images = [tensor.numpy() for tensor in list_of_images]

# embeddable_images = get_images(list_of_images)

# plot_UMAP_embedding(embed, simple_tweetyclr.mean_colors_per_minispec,embeddable_images, f'{simple_tweetyclr.folder_name}/Plots/UMAP_of_all_slices.html', saveflag = True)

hard_indices = hard_indices_dict[0]
hard_dataset = data_for_analysis[hard_indices,:].reshape(len(hard_indices), 1, 100, 151) # Dataset of all the confused spectrogram slices that we want to untangle

hard_dataset, hard_dataloader = create_dataloader(hard_dataset, 64, hard_indices, shuffle_status=False)

# Need to create a train dataset and test dataset on the hard indices. 

# Split the dataset into a training and testing dataset
# Define the split sizes -- what is the train test split ? 

train_perc = 0.5 #
train_size = int(train_perc * len(hard_dataset))  # (100*train_perc)% for training
test_size = len(hard_dataset) - train_size  # 100 - (100*train_perc)% for testing

# ORGANIZE BELOW INTO A DATASET AND DATALOADER CREATION HELPER FUNCTION

from torch.utils.data import random_split

train_hard_dataset, test_hard_dataset = random_split(hard_dataset, [train_size, test_size]) 

# Getting the indices
train_indices = np.array(train_hard_dataset.indices)
test_indices = np.array(test_hard_dataset.indices)
embed_train = embed[hard_indices[train_indices],:]
embed_test = embed[hard_indices[test_indices], :]

# Let's define the set of easy and hard negatives


shuffle_status = True
batch_size = 64
k_neg = 2


easy_negatives, hard_negatives = creating_negatives_set(embed, hard_indices)


training_dataset = Curating_Dataset(k_neg, train_hard_dataset, easy_negatives, total_dataset)
testing_dataset = Curating_Dataset(k_neg, test_hard_dataset, easy_negatives, total_dataset)

train_loader = torch.utils.data.DataLoader(training_dataset, batch_size = batch_size, shuffle = shuffle_status)
test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size = batch_size, shuffle = shuffle_status)

batch, negative_indices = next(iter(train_loader))

# Dimensions of the output: 
    # anchor_img: [batch_size, 1, 1, 100, 151]
    # negative_img: [batch_size, k_neg, 1, 100, 151]
    # negative_indices: [batch_size, k_neg]


class Encoder(nn.Module):
    def __init__(self, fc_dimensionality, dropout_perc):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3,1,padding=1)
        self.conv2 = nn.Conv2d(8, 8, 3,2,padding=1)
        self.conv3 = nn.Conv2d(8, 16,3,1,padding=1)
        self.conv4 = nn.Conv2d(16,16,3,2,padding=1)
        self.conv5 = nn.Conv2d(16,24,3,1,padding=1)
        self.conv6 = nn.Conv2d(24,24,3,2,padding=1)
        self.conv7 = nn.Conv2d(24,32,3,1,padding=1)
        self.conv8 = nn.Conv2d(32,24,3,2,padding=1)
        self.conv9 = nn.Conv2d(24,24,3,1,padding=1)
        self.conv10 = nn.Conv2d(24,16,3,2,padding=1)

        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn3 = nn.BatchNorm2d(16)
        self.bn4 = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(24)
        self.bn6 = nn.BatchNorm2d(24)
        self.bn7 = nn.BatchNorm2d(32)
        self.bn8 = nn.BatchNorm2d(24)
        self.bn9 = nn.BatchNorm2d(24)
        self.bn10 = nn.BatchNorm2d(16)
        
        self.ln1 = nn.LayerNorm([8, 100, 151])
        self.ln2 = nn.LayerNorm([8, 50, 76])
        self.ln3 = nn.LayerNorm([16, 50, 76])
        self.ln4 = nn.LayerNorm([16, 25, 38])
        self.ln5 = nn.LayerNorm([24, 25, 38])
        self.ln6 = nn.LayerNorm([24, 13, 19])
        self.ln7 = nn.LayerNorm([32, 13, 19])
        self.ln8 = nn.LayerNorm([24, 7, 10])
        self.ln9 = nn.LayerNorm([24, 7, 10])
        self.ln10 = nn.LayerNorm([16, 4, 5])

        self.relu = nn.ReLU(inplace = False)
        self.sigmoid = nn.Sigmoid()
        
        # self.fc = nn.Sequential(
        #     nn.Linear(320, 256),
        #     nn.ReLU(inplace=True), 
        #     nn.Linear(256, 1)
        # )  
        self.fc = nn.Sequential(
            nn.Linear(320, fc_dimensionality),
            nn.ReLU(inplace=False), 
        )  
        
        self.dropout = nn.Dropout(p=dropout_perc)
        
        # Initialize convolutional layers with He initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward_once(self, x):
        ''' 
        Redefinition of the forward function since we are not working with
        (anchor, positive, negative) triplets. 
        
        '''
        
        # LayerNorm + Dropout
        x = self.dropout(self.relu(self.ln1(self.conv1(x))))
        x = self.dropout(self.relu(self.ln2(self.conv2(x))))
        x = self.dropout(self.relu(self.ln3(self.conv3(x))))
        x = self.dropout(self.relu(self.ln4(self.conv4(x))))
        x = self.dropout(self.relu(self.ln5(self.conv5(x))))
        x = self.dropout(self.relu(self.ln6(self.conv6(x))))
        x = self.dropout(self.relu(self.ln7(self.conv7(x))))
        x = self.dropout(self.relu(self.ln8(self.conv8(x))))
        x = self.dropout(self.relu(self.ln9(self.conv9(x))))
        x = self.dropout(self.relu(self.ln10(self.conv10(x))))

        x_flattened = x.view(-1, 320)
                
        return x_flattened
    
    def forward(self, x):
        
        features = self.relu(self.fc(self.forward_once(x)))
        
        return features


fc_dimensionality = 256
dropout_perc = 0
model = Encoder(fc_dimensionality=fc_dimensionality, dropout_perc=dropout_perc)
# Check if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    model = nn.DataParallel(model)

# Move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).to(torch.float32)

# =============================================================================
# UNTRAINED MODEL REPRESENTATION
# =============================================================================

# train_hard_loader = torch.utils.data.DataLoader(train_hard_dataset, batch_size = batch_size, shuffle = False)
# model_rep_untrained = create_UMAP_plot(train_hard_loader, simple_tweetyclr, hard_indices[train_indices], model, 'UMAP_rep_of_model_train_region_untrained_model', saveflag = True)

# =============================================================================
# MODEL BUILDING
# =============================================================================

temperature = 1.0
num_augmentations = 2
optimizer = optim.Adam(model.parameters(), lr=1e-3)

training_batch_loss = []
pos_sim_train_batch = []
neg_sim_train_batch = []
training_epoch_loss = []

validation_batch_loss = []
pos_sim_val_batch = []
neg_sim_val_batch = []
validation_epoch_loss = []

num_epochs = 100
model = model.to(device).float()  # Convert model to float32 and move to device once
noise_level = 0.5  # Define noise level outside the loop
num_augmentations = 2  # Define number of augmentations


wn = Augmentations(noise_level = noise_level, num_augmentations=num_augmentations)

train_loader = torch.utils.data.DataLoader(training_dataset, batch_size = batch_size, shuffle = shuffle_status)
test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size = batch_size, shuffle = shuffle_status)

# Overfitting step 1: Make sure I can overfit on a single batch
train_batch, _= next(iter(train_loader))
validation_batch, _ = next(iter(test_loader))

step = 0
max_steps = 100

# Initialize lists for storing metrics
training_batch_loss = []
pos_sim_batch_train = []
neg_sim_batch_train = []


validation_batch_loss = []
pos_sim_batch_val = []
neg_sim_batch_val = []


while step < max_steps:
    # Move batch to device once and ensure it's float32
    batch = train_batch.to(device).to(torch.float32)
    
    augmented_tensor = wn.white_noise(batch) 

    # I need to flattened the augmented_tensor in order to pass it through my model
    augmented_tensor = augmented_tensor.view(-1, 1, 100, 151)  # New shape: [768, 1, 100, 151]
        
    feats = model(augmented_tensor)  # Direct model invocation
    
    total_samples = feats.shape[0]
    group_size = num_augmentations * (k_neg + 1)  # The size of each group

    dynamic_batch_size = total_samples // group_size
    
    feats = feats.view(dynamic_batch_size, num_augmentations*(k_neg+1), fc_dimensionality)

    loss, pos_sim, neg_sim = infonce_loss_function(feats, temperature)

    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Store metrics after each step
    training_batch_loss.append(loss.item())
    pos_sim_batch_train.append(pos_sim.item())
    neg_sim_batch_train.append(neg_sim.item())

    model.eval()
    with torch.no_grad():
        # Fetch the next batch from the validation set
        batch = validation_batch.to(device).to(torch.float32)

        # Apply augmentation

        augmented_tensor = wn.white_noise(batch) 

        # I need to flattened the augmented_tensor in order to pass it through my model
        augmented_tensor = augmented_tensor.view(-1, 1, 100, 151)  # New shape: [768, 1, 100, 151]
            
        feats = model(augmented_tensor)  # Direct model invocation
        
        total_samples = feats.shape[0]
        group_size = num_augmentations * (k_neg + 1)  # The size of each group

        dynamic_batch_size = total_samples // group_size
        
        feats = feats.view(dynamic_batch_size, num_augmentations*(k_neg+1), fc_dimensionality)
        loss, pos_sim, neg_sim = infonce_loss_function(feats, temperature)

        validation_batch_loss.append(loss.item())
        pos_sim_batch_val.append(pos_sim.item())
        neg_sim_batch_val.append(neg_sim.item())

        step +=1





















# TODO: Need to clean up this code. Need to make sure that I am calculating the validation loss at each model update, rather than calculating the validation loss after the model has seen the entire training set. 

step = 0
max_steps = 100

train_iter = iter(train_loader)
test_iter = iter(test_loader)

# Initialize lists for storing metrics
training_batch_loss = []
pos_sim_batch_train = []
neg_sim_batch_train = []


validation_batch_loss = []
pos_sim_batch_val = []
neg_sim_batch_val = []


while step < max_steps:

    try:
        batch, _ = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        batch, _ = next(train_iter)

    # Move batch to device once and ensure it's float32
    batch = batch.to(device).to(torch.float32)
    
    augmented_tensor = wn.white_noise(batch) 

    # I need to flattened the augmented_tensor in order to pass it through my model
    augmented_tensor = augmented_tensor.view(-1, 1, 100, 151)  # New shape: [768, 1, 100, 151]
        
    feats = model(augmented_tensor)  # Direct model invocation
    
    total_samples = feats.shape[0]
    group_size = num_augmentations * (k_neg + 1)  # The size of each group

    dynamic_batch_size = total_samples // group_size
    
    feats = feats.view(dynamic_batch_size, num_augmentations*(k_neg+1), fc_dimensionality)
    loss, pos_sim, neg_sim = infonce_loss_function(feats, temperature)

    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Store metrics after each step
    training_batch_loss.append(loss.item())
    pos_sim_batch_train.append(pos_sim.item())
    neg_sim_batch_train.append(neg_sim.item())

    # Your existing code where validation loss is computed

    model.eval()
    with torch.no_grad():
        try:
            batch, _ = next(test_iter)
        except StopIteration:
            test_iter = iter(test_loader)
            batch, _ = next(test_iter)
            step +=1
            print(f'Epoch {step}, Last Training Batch Loss: {training_batch_loss[-1]}, Last Validation Batch Loss: {validation_batch_loss[-1]}')

        # Fetch the next batch from the validation set
        batch = batch.to(device).to(torch.float32)

        # Apply augmentation

        augmented_tensor = wn.white_noise(batch) 

        # I need to flattened the augmented_tensor in order to pass it through my model
        augmented_tensor = augmented_tensor.view(-1, 1, 100, 151)  # New shape: [768, 1, 100, 151]
            
        feats = model(augmented_tensor)  # Direct model invocation
        
        total_samples = feats.shape[0]
        group_size = num_augmentations * (k_neg + 1)  # The size of each group

        dynamic_batch_size = total_samples // group_size
        
        feats = feats.view(dynamic_batch_size, num_augmentations*(k_neg+1), fc_dimensionality)
        loss, pos_sim, neg_sim = infonce_loss_function(feats, temperature)

        validation_batch_loss.append(loss.item())
        pos_sim_batch_val.append(pos_sim.item())
        neg_sim_batch_val.append(neg_sim.item())


# Let's look at loss curves and diagnostic plots
plt.figure()
plt.plot(training_batch_loss, label = 'Training Loss')
plt.plot(validation_batch_loss, label = 'Validation Loss')
plt.legend()
plt.title("Unsupervised Training")
plt.xlabel("Batch Number")
plt.ylabel("Raw InfoNCE Loss")
plt.savefig(f'{folder_name}/raw_loss_curve.png')
plt.show()

# LOg loss curve
training_batch_loss_array = np.array(training_batch_loss) + 1e-1
validation_batch_loss_array = np.array(validation_batch_loss) + 1e-1

plt.figure()
plt.plot(training_batch_loss_array, label = 'Training Loss')
plt.plot(validation_batch_loss_array, label = 'Validation Loss')
# plt.axhline(y = np.log(1e-1), label = "Minimum Possible Loss")
plt.legend()
plt.xlabel("Batch Number")
plt.ylabel("Log(InfoNCE Loss + 1e-1)")
plt.title("Unsupervised Training")
plt.savefig(f'{folder_name}/log_loss_curve.png')
plt.show()


# NOw let's look at the positive and negative similarities for training and validation

plt.figure()
plt.plot(pos_sim_train_batch, label = 'Positive Similarities')
plt.plot(neg_sim_train_batch, label = 'Negative Similarities')
plt.legend()
plt.xlabel("Batch Number")
plt.ylabel("Cosine Similarity")
plt.title("Training Data Learning Dynamics")
plt.savefig(f'{folder_name}/training_data_sims_plot.png')
plt.show()

plt.figure()
plt.plot(pos_sim_val_batch, label = 'Positive Similarities')
plt.plot(neg_sim_val_batch, label = 'Negative Similarities')
plt.legend()
plt.xlabel("Batch Number")
plt.ylabel("Cosine Similarity")
plt.title("Validation Data Learning Dynamics")
plt.savefig(f'{folder_name}/validation_data_sims_plot.png')
plt.show()

model_form = model.module


model_arch = str(model_form)
forward_method = inspect.getsource(model_form.forward)
forward_once_method = inspect.getsource(model_form.forward_once)

# Splitting the string into an array of lines
model_arch_lines = model_arch.split('\n')
forward_method_lines = forward_method.split('\n')
forward_once_method_lines = forward_once_method.split('\n')


# CHANGE THIS 


experiment_params = {
    "Data_Directory": bird_dir,
    "Window_Size": simple_tweetyclr.window_size, 
    "Stride_Size": simple_tweetyclr.stride, 
    "Num_Spectrograms": simple_tweetyclr.num_spec, 
    "Total_Slices": simple_tweetyclr.stacked_windows.shape[0], 
    "Frequencies_of_Interest": masking_freq_tuple, 
    "Data_Standardization": "None",
    "Optimizer": str(optimizer), 
    "Num_negative_samples": k_neg,
    "Num_Augmentations": num_augmentations,
    "Temperature": temperature,
    "Num_Epochs": num_epochs, 
    "Torch_Random_Seed": 295, 
    "Accumulation_Size": train_perc, 
    "Train_Proportion": train_perc,
    "Model_Architecture": model_arch_lines, 
    "Forward_Method": forward_method_lines, 
    "Forward_Once_Method": forward_once_method_lines,
    "Dataloader_Shuffle": shuffle_status, 
    "Noise_Level": noise_level, 
    "fc_dimensionality": fc_dimensionality, 
    "Dropout_Perc": dropout_perc
    }

import json

with open(f'{simple_tweetyclr.folder_name}/experiment_params.json', 'w') as file:
    json.dump(experiment_params, file, indent=4)
