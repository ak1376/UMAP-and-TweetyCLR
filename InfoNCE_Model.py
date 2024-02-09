#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 21:10:36 2024

Now upon training the model using the triplet loss, I will return to the 
UMAP + TweetyCLR model using the InfoNCE loss function. This will require that
I apply the augmentation scheme to each image in a batch, keep track of the 
positive pairs and reimplement the loss function. 

@author: akapoor
"""

import numpy as np
import torch
import sys
# filepath = '/home/akapoor'
filepath = '/Users/AnanyaKapoor'
import os
# os.chdir('/Users/AnanyaKapoor/Downloads/TweetyCLR')
os.chdir(f'{filepath}/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_b_Canary_SSL/TweetyCLR_End_to_End')
from util import Tweetyclr
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle
from torch.utils.data import DataLoader, TensorDataset
import umap
import matplotlib.pyplot as plt
import torch.optim as optim
import itertools
import inspect
import torch.nn.init as init
import random
from torch.utils.data import Dataset
from collections import defaultdict
import time


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
num_spec = 100
window_size = 100
stride = 10

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
simple_tweetyclr.first_time_analysis()

# Documentation code
if log_experiment == True: 
    exp_descp = input('Please give a couple of sentences describing what the experiment is testing: ')
    # Save the input to a text file
    with open(f'{folder_name}/experiment_readme.txt', 'w') as file:
        file.write(exp_descp)


stacked_windows = simple_tweetyclr.stacked_windows.copy()
# mean = np.mean(stacked_windows, axis=1, keepdims=True)
# std_dev = np.std(stacked_windows, axis=1, keepdims=True)

# # # Perform z-scoring
# z_scored = (stacked_windows - mean) / std_dev

# # # Replace NaNs with 0s
# z_scored = np.nan_to_num(z_scored)

# stacked_windows = z_scored.copy()

stacked_windows.shape = (stacked_windows.shape[0], 100, 151)

stacked_windows[:, :, :] = simple_tweetyclr.stacked_labels_for_window[:, :, None]

stacked_windows.shape = (stacked_windows.shape[0], 100*151) 

# Set up a base dataloader (which we won't directly use for modeling). Also define the batch size of interest
total_dataset = TensorDataset(torch.tensor(simple_tweetyclr.stacked_windows.reshape(simple_tweetyclr.stacked_windows.shape[0], 1, 100, 151)))
batch_size = 1 # This is the upper order batch size. The real batch size when used for training will be the 2 + 2*k_neg 
total_dataloader = DataLoader(total_dataset, batch_size=batch_size , shuffle=False)

import torch
import torch.nn as nn
import torch.optim as optim
# from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

class Encoder(nn.Module):
    def __init__(self):
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
            nn.Linear(320, 256),
            nn.ReLU(inplace=False), 
        )  
        
        self.dropout = nn.Dropout(p=0.5)
        
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

hard_indices_dict = {}
hard_region_coordinates = {}

# Get current axes
ax = plt.gca()
# Get current limits
xlim = ax.get_xlim()
ylim = ax.get_ylim()

hard_indices = np.where((embed[:,0]>=xlim[0])&(embed[:,0]<=xlim[1]) & (embed[:,1]>=ylim[0]) & (embed[:,1]<=ylim[1]))[0]


hard_indices_dict[0] = hard_indices
hard_region_coordinates[0] = [xlim, ylim]

# Now let's zoom out on the matplotlib plot

# Zoom out by changing the limits
# You can adjust these values as needed to zoom out to the desired level
ax.set_xlim([min(embed[:,0]) - 1, max(embed[:,0]) + 1])  # Zoom out on x-axis
ax.set_ylim([min(embed[:,1]) - 1, max(embed[:, 1]) + 1])  # Zoom out on y-axis

# Show the updated plot
plt.show()


for i in np.arange(len(hard_indices_dict)):
    hard_ind = hard_indices_dict[i]

    plt.figure(figsize = (10,10))
    plt.scatter(embed[hard_ind,0], embed[hard_ind,1], s = 10, c = simple_tweetyclr.mean_colors_per_minispec[hard_ind,:])
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    # plt.title("UMAP Decomposition of ")
    plt.suptitle(f'UMAP Representation of Hard Region #{i}')
    plt.title(f'Total Slices: {embed[hard_ind,:].shape[0]}')
    plt.savefig(f'{simple_tweetyclr.folder_name}/Plots/UMAP_of_hard_slices_region_{i}.png')
    plt.show()


data_for_analysis = simple_tweetyclr.stacked_windows.copy()
data_for_analysis = data_for_analysis.reshape(data_for_analysis.shape[0], 1, 100, 151)
total_dataset = TensorDataset(torch.tensor(data_for_analysis.reshape(data_for_analysis.shape[0], 1, simple_tweetyclr.time_dim, simple_tweetyclr.freq_dim)))
total_dataloader = DataLoader(total_dataset, batch_size=batch_size , shuffle=False)

# list_of_images = []
# for batch_idx, (data) in enumerate(total_dataloader):
#     data = data[0]
    
#     for image in data:
#         list_of_images.append(image)
        
# list_of_images = [tensor.numpy() for tensor in list_of_images]

# embeddable_images = simple_tweetyclr.get_images(list_of_images)

# simple_tweetyclr.plot_UMAP_embedding(embed, simple_tweetyclr.mean_colors_per_minispec,embeddable_images, f'{simple_tweetyclr.folder_name}/Plots/UMAP_of_all_slices.html', saveflag = True)

hard_dataset = data_for_analysis[hard_indices,:].reshape(len(hard_indices), 1, 100, 151) # Dataset of all the confused spectrogram slices that we want to untangle

dataset = TensorDataset(torch.tensor(data_for_analysis), torch.tensor(np.arange(data_for_analysis.shape[0]))) # The full dataset of all slices
hard_dataset = TensorDataset(torch.tensor(hard_dataset), torch.tensor(hard_indices)) # The dataset of just the hard indices


# In[2]: I will first do white noise augmentation the simple dumb way (not worry about speed). But if it hampers speed I will fix it.

# Need to create a train dataset and test dataset on the hard indices. 

# Split the dataset into a training and testing dataset
# Define the split sizes -- what is the train test split ? 
train_perc = 0.5 #
train_size = int(train_perc * len(hard_dataset))  # (100*train_perc)% for training
test_size = len(hard_dataset) - train_size  # 100 - (100*train_perc)% for testing

from torch.utils.data import random_split

train_hard_dataset, test_hard_dataset = random_split(hard_dataset, [train_size, test_size]) 

# Getting the indices
train_indices = np.array(train_hard_dataset.indices)
test_indices = np.array(test_hard_dataset.indices)
embed_train = embed[hard_indices[train_indices],:]
embed_test = embed[hard_indices[test_indices], :]

# These are the training hard anchors and the testing hard anchors. I have two options
# 1. Creating another training and testing dataloader where the batches contain the negatives already (batch = t(anchor), t'(anchor), t(negatives), t'(negatives))
# 2. For each sample in the hard train dataloader and hard test dataloader
#   a. select batch_size - 1 negatives
#   b. apply augmentations to each sample in the batch

# I am more inclined to do 1 instead of 2

class Curating_Dataset(Dataset):
    def __init__(self, k_neg, hard_dataset, easy_negatives, dataset):
        '''

        Parameters
        ----------
        k_neg : int
            Number of negative samples for each anchor sample.
            
        easy_negatives: torch tensor
            The set of indices for easy negatives. Note: this is {ALL INDICES}\{HARD INDICES}

        Returns
        -------
        None.

        '''
        
        self.k_neg = k_neg
        self.hard_dataset = hard_dataset
        self.easy_negatives = easy_negatives
        self.dataset = dataset
        
        self.all_features, self.all_indices = zip(*[self.dataset[i] for i in range(len(self.dataset))]) # I will use this to extract the images using the subsequent negative indices
        self.hard_features, self.hard_indices = zip(*[self.hard_dataset[i] for i in range(len(self.hard_dataset))]) # This will be used to create all the hard features

        # Converting lists of tensors to a single tensor
        self.all_features, self.all_indices = torch.stack(self.all_features), torch.stack(self.all_indices)
        self.hard_features, self.hard_indices = torch.stack(self.hard_features), torch.stack(self.hard_indices)
        
        
    def __len__(self):
        return self.hard_indices.shape[0]
        
    def __getitem__(self, index):
        '''
        For each hard index I want to randomly select k_neg negative samples.

        Parameters
        ----------
        index : int
            Batch index.

        Returns
        -------
        None.

        '''

        actual_index = int(self.all_indices[int(self.hard_indices[index])])
        anchor_img = self.all_features[actual_index,:, :, :].unsqueeze(1)

        random_indices = torch.randint(0, self.easy_negatives.size(0), (self.k_neg,))
        
        negative_indices = self.easy_negatives[random_indices]
        
        negative_imgs = self.all_features[negative_indices, :, :, :]
        
        # x = torch.cat((anchor_img, negative_imgs), dim = 0)
        
        return anchor_img, negative_imgs, negative_indices

# Let's define the set of easy negatives

total_indices = np.arange(embed.shape[0])

easy_negatives = np.setdiff1d(total_indices, hard_indices)
easy_negatives = torch.tensor(easy_negatives)
shuffle_status = True
noise_level = 1.0

k_neg = 31

training_dataset = Curating_Dataset(k_neg, train_hard_dataset, easy_negatives, dataset)
testing_dataset = Curating_Dataset(k_neg, test_hard_dataset, easy_negatives, dataset)

train_loader = torch.utils.data.DataLoader(training_dataset, batch_size = batch_size, shuffle = shuffle_status)
test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size = batch_size, shuffle = shuffle_status)

anchor_img, negative_imgs, negative_indices = next(iter(train_loader))

# Dimensions of the output: 
    # anchor_img: [batch_size, 1, 1, 100, 151]
    # negative_img: [batch_size, k_neg, 1, 100, 151]
    # negative_indices: [batch_size, k_neg]

# In[3]: I want to create an object that will do white noise augmentation 

class White_Noise():
    def __init__(self, num_augmentations, noise_level = 1.0):
        self.num_augmentations = num_augmentations
        self.noise_level = noise_level
        
    def augment_with_white_noise_k_times(self, images):
        """
        Apply k white noise augmentations to 5D PyTorch tensors.
    
        Parameters:
        - images: 5D PyTorch tensor of shape [batch_size, k_samples, 1, width, height]
        - k: int, the number of white noise augmentations to apply
        - noise_scale: float, the scale of the noise, relative to the data range
    
        Returns:
        - Augmented images: 6D PyTorch tensor of shape [k, batch_size, k_samples, 1, width, height]
        """
        # Generate noise for all augmentations in one go
        # The noise shape needs to match images shape, except we add k as the first dimension
        k = self.num_augmentations
        noise_shape = (k,) + images.shape
        noise = torch.rand(noise_shape) * self.noise_level
        
        # Expand the original images tensor to match the noise tensor shape for broadcasting
        images_expanded = images.unsqueeze(0).expand(noise_shape)
        
        # Add the noise to the expanded images tensor and clip values to be between 0 and 1
        noisy_images = torch.clamp(images_expanded + noise, 0, 1)
        
        return noisy_images
    
    
    
    def __call__(self, anchor_img, negative_imgs): 
        # Corrected method calls without passing 'self' explicitly
        anchor_augmented = self.augment_with_white_noise_k_times(anchor_img)
        neg_augmented = self.augment_with_white_noise_k_times(negative_imgs)
        
        return anchor_augmented, neg_augmented
        
        
anchor_img_squeezed = anchor_img.squeeze(2)  # Removing the singleton dimension for compatibility

num_augmentations = 2 


wn = White_Noise(num_augmentations, noise_level = 1.0)

anchor_aug, negative_aug = wn(anchor_img, negative_imgs)

reshaped_aug = anchor_aug.squeeze().unsqueeze(1)
reshaped_neg = negative_aug.reshape(negative_aug.shape[0]*negative_aug.shape[1]*negative_aug.shape[2], 1, 100, 151)   
        
# I can then pass reshaped_aug and reshaped_neg both through the network and easily calculate the loss function

model = Encoder()
# Check if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    model = nn.DataParallel(model)

# Move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).to(torch.float32)

# Scratch code for infonce loss function. Written for just 2 augmentations. 

temperature = 1.0
optimizer = optim.Adam(model.parameters(), lr=1e-3)
training_batch_loss = []
num_epochs = 10
for epoch in np.arange(num_epochs):
    model.to(device).to(torch.float32)
    model.train()
    training_loss = 0

    for batch_idx, (anchor_img, negative_imgs, negative_indices) in enumerate(train_loader):
        optimizer.zero_grad()
        anchor_aug, negative_aug = wn(anchor_img, negative_imgs)
    
        reshaped_aug = anchor_aug.squeeze().unsqueeze(1)
        reshaped_neg = negative_aug.reshape(negative_aug.shape[0]*negative_aug.shape[1]*negative_aug.shape[2], 1, 100, 151)   
    
        # ANCHOR AUGS
        conv_output_anchor_aug = model.forward(reshaped_aug.to(torch.float32))
        anchor_aug_features = F.normalize(conv_output_anchor_aug, dim=1)
        anchor_aug_similarity = torch.matmul(anchor_aug_features, anchor_aug_features.T)
        
        anchor_aug_sim_val = torch.exp(anchor_aug_similarity[0,1]/temperature)
        
        # NEGATIVE AUGS
        
        conv_output_neg_aug = model.forward(reshaped_neg.to(torch.float32))
        neg_aug_features = F.normalize(conv_output_neg_aug, dim=1)
        neg_aug_similarity = torch.matmul(neg_aug_features, neg_aug_features.T)
        neg_aug_sim_val = torch.sum(torch.exp(neg_aug_similarity[0,1:-1]/temperature))
        
        loss = -torch.log(anchor_aug_sim_val/neg_aug_sim_val)
        print(loss.item())
        
        training_loss+=loss.item()
        training_batch_loss.append(loss.item())
        
        loss.backward()
        optimizer.step()




































margin_value = 1.0
criterion = nn.TripletMarginLoss(margin=margin_value, p=2)
# optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 100
patience = 30  # Number of epochs to wait for improvement before stopping
min_delta = 0.001  # Minimum change to qualify as an improvement

best_val_loss = float('inf')
epochs_no_improve = 0
early_stop = False







# =============================================================================
# UNTRAINED MODEL REPRESENTATION
# =============================================================================

model_rep_untrained = []
train_loader = torch.utils.data.DataLoader(train_hard_dataset, batch_size = batch_size, shuffle = False)

model = model.to('cpu')
model.eval()
with torch.no_grad():
    for batch_idx, (img, idx) in enumerate(train_loader):
        data = img.to(torch.float32)
        
        
        
        
        
        
        
        
        
        
        
        output = model.module.forward_once(data)
        model_rep_untrained.append(output.numpy())

model_rep_stacked = np.concatenate((model_rep_untrained))

import umap
reducer = umap.UMAP(metric = 'cosine', random_state=295) # For consistency
embed_train = reducer.fit_transform(model_rep_stacked)

plt.figure()
plt.scatter(embed_train[:,0], embed_train[:,1], c = simple_tweetyclr.mean_colors_per_minispec[hard_indices[train_indices],:])
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.suptitle(f'UMAP Representation of Training Hard Region')
plt.title(f'Total Slices: {embed_train.shape[0]}')
plt.show()
plt.savefig(f'{folder_name}/UMAP_rep_of_model_train_region_untrained_model.png')


training_epoch_loss = []
training_batch_loss = []
validation_epoch_loss = []
validation_batch_loss = []

training_pos_aug = []
training_neg_sample = []

validation_pos_aug = []
validation_neg_sample = []


for epoch in np.arange(num_epochs):
    model.to(device).to(torch.float32)
    model.train()
    training_loss = 0
    for batch_idx, (anchor_img, positive_img, negative_img, negative_index) in enumerate(train_loader):
        anchor_img, positive_img, negative_img = anchor_img.to(device, dtype = torch.float32), positive_img.to(device, dtype = torch.float32), negative_img.to(device, dtype = torch.float32)
        
# =============================================================================
#         DEBUGGING STEP FOR NEGATIVE SAMPLES
# =============================================================================

        # negative_img = torch.zeros_like(anchor_img)
        # positive_img = torch.ones_like(anchor_img)

        optimizer.zero_grad()
        anchor_emb, positive_emb, negative_emb = model(anchor_img, positive_img, negative_img)
        # training_pos_aug.append(positive_img.detach().cpu())
        # training_neg_sample.append(negative_img.detach().cpu())
        loss = criterion(anchor_emb, positive_emb, negative_emb)
        training_loss+=loss.item()
        training_batch_loss.append(loss.item())
        loss.backward()
        optimizer.step()

    
    # DEBUGGING STEP: LOOKING AT THE MODEL REPRESENTATIONS
# =============================================================================
#     # TRAINING
# =============================================================================
    
    # if epoch%5 == 0:
    #     model_rep_train = []
    #     model_for_eval = model.module
    #     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = False)
    #     # model_for_eval = model_for_eval.to('cpu')
    #     model_for_eval.eval()
        
    #     with torch.no_grad():
    #         for batch_idx, (anchor_img, positive_img, negative_img) in enumerate(train_loader):
    #             data = anchor_img.to(torch.float32).to(device)
    #             output = model_for_eval.forward_once(data)
    #             model_rep_train.append(output.cpu().detach().numpy())
        
    #     model_rep_stacked = np.concatenate((model_rep_train))
        
    #     reducer = umap.UMAP(metric = 'cosine', random_state=295) # For consistency
    #     embed_train = reducer.fit_transform(model_rep_stacked)
        
    #     plt.figure()
    #     plt.scatter(embed_train[:,0], embed_train[:,1], c = simple_tweetyclr.mean_colors_per_minispec[hard_indices[train_indices],:])
    #     plt.xlabel("UMAP 1")
    #     plt.ylabel("UMAP 2")
    #     plt.suptitle(f'UMAP Representation of Training Hard Region')
    #     plt.title(f'Total Slices: {embed_train.shape[0]}')
    #     plt.show()
    #     plt.savefig(f'{folder_name}/UMAP_rep_of_model_train_epoch_{epoch}.png')

    # model.to(device).to(torch.float32)
    model.eval()
    with torch.no_grad():
        validation_loss = 0
        for batch_idx, (anchor_img, positive_img, negative_img, negative_index) in enumerate(test_loader):
            anchor_img, positive_img, negative_img = anchor_img.to(device, dtype = torch.float32), positive_img.to(device, dtype = torch.float32), negative_img.to(device, dtype = torch.float32)
# =============================================================================
#             debugging
# =============================================================================
            # negative_img = torch.zeros_like(anchor_img)
            # positive_img = torch.ones_like(anchor_img)

            anchor_emb, positive_emb, negative_emb = model(anchor_img, positive_img, negative_img)
            # validation_pos_aug.append(positive_img.detach().cpu())
            # validation_neg_sample.append(negative_img.detach().cpu())
            
            loss = criterion(anchor_emb, positive_emb, negative_emb)
            validation_loss+=loss.item()
            validation_batch_loss.append(loss.item())
        
    training_epoch_loss.append(training_loss / len(train_loader))
    validation_epoch_loss.append(validation_loss / len(test_loader))
    
    
# # =============================================================================
# #     # TESTING
# # =============================================================================

    # if epoch%5 == 0:
    #     model_rep_test = []
    #     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
    #     model_for_eval = model.module
    #     # model_for_eval = model_for_eval.to('cpu')
    #     model_for_eval.eval()
    #     with torch.no_grad():
    #         for batch_idx, (anchor_img, positive_img, negative_img) in enumerate(test_loader):
    #             data = anchor_img.to(torch.float32).to(device)
    #             output = model_for_eval.forward_once(data)
    #             model_rep_test.append(output.cpu().detach().numpy())
        
    #     model_rep_stacked = np.concatenate((model_rep_test))
        
    #     import umap
    #     reducer = umap.UMAP(metric = 'cosine', random_state=295) # For consistency
    #     embed_test = reducer.fit_transform(model_rep_stacked)
        
    #     plt.figure()
    #     plt.scatter(embed_test[:,0], embed_test[:,1], c = simple_tweetyclr.mean_colors_per_minispec[hard_indices[test_indices],:])
    #     plt.xlabel("UMAP 1")
    #     plt.ylabel("UMAP 2")
    #     plt.suptitle("UMAP of the Testing Hard Region")
    #     plt.title(f'Total Slices: {embed_test.shape[0]}')
    #     plt.show()
    #     plt.savefig(f'{folder_name}/UMAP_rep_of_model_test_epoch_{epoch}.png')

    
    # Check for improvement
    if validation_epoch_loss[-1] + min_delta < best_val_loss:
        best_val_loss = validation_epoch_loss[-1]
        epochs_no_improve = 0
        # Save the best model
        best_model_path = f'{simple_tweetyclr.folder_name}/best_model.pth'
        torch.save(model.state_dict(), best_model_path)
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after epoch {epoch}!')
            early_stop = True
            break  # Early stopping
    
    print(f'Epoch {epoch}, Training Loss: {training_epoch_loss[-1]}, Validation Loss {validation_epoch_loss[-1]}')

if not early_stop:
    print('Training completed without early stopping.')

# plt.figure()
# plt.plot(training_epoch_loss, label = 'Training Loss')
# plt.plot(validation_epoch_loss, label = 'Validation Loss')
# plt.legend()
# plt.xlabel("Epoch")
# plt.ylabel("Triplet Loss")
# plt.title("Unsupervised Training")
# plt.savefig(f'{folder_name}/loss_curve.png')
# plt.show()

training_batch_loss_array = np.array(training_batch_loss) + 1e-1
validation_batch_loss_array = np.array(validation_batch_loss) + 1e-1

plt.figure()
plt.plot(training_batch_loss_array, label = 'Training Loss')
plt.plot(validation_batch_loss_array, label = 'Validation Loss')
# plt.axhline(y = np.log(1e-1), label = "Minimum Possible Loss")
plt.legend()
plt.xlabel("Batch Number")
plt.ylabel("Log(Triplet Loss + 1e-1)")
plt.title("Unsupervised Training")
plt.savefig(f'{folder_name}/loss_curve.png')
plt.show()

# =============================================================================
# TRAINING DATA
# =============================================================================

model_rep_train = []
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = False)

model = model.to('cpu')
model.eval()
with torch.no_grad():
    for batch_idx, (anchor_img, positive_img, negative_img, negative_index) in enumerate(train_loader):
        data = anchor_img.to(torch.float32)
        output = model.module.forward_once(data)
        model_rep_train.append(output.numpy())

model_rep_stacked = np.concatenate((model_rep_train))

import umap
reducer = umap.UMAP(metric = 'cosine', random_state=295) # For consistency
embed_train = reducer.fit_transform(model_rep_stacked)

plt.figure()
plt.scatter(embed_train[:,0], embed_train[:,1], c = simple_tweetyclr.mean_colors_per_minispec[hard_indices[train_indices],:])
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.suptitle(f'UMAP Representation of Training Hard Region')
plt.title(f'Total Slices: {embed_train.shape[0]}')
plt.show()
plt.savefig(f'{folder_name}/UMAP_rep_of_model_train.png')

# =============================================================================
# TESTING DATA
# =============================================================================

model_rep_test = []
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

model = model.to('cpu')
model.eval()
with torch.no_grad():
    for batch_idx, (anchor_img, positive_img, negative_img, negative_index) in enumerate(test_loader):
        data = anchor_img.to(torch.float32)
        output = model.module.forward_once(data)
        model_rep_test.append(output.numpy())

model_rep_stacked = np.concatenate((model_rep_test))

import umap
reducer = umap.UMAP(metric = 'cosine', random_state=295) # For consistency
embed_test = reducer.fit_transform(model_rep_stacked)

plt.figure()
plt.scatter(embed_test[:,0], embed_test[:,1], c = simple_tweetyclr.mean_colors_per_minispec[hard_indices[test_indices],:])
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.suptitle("UMAP of the Testing Hard Region")
plt.title(f'Total Slices: {embed_test.shape[0]}')
plt.show()
plt.savefig(f'{folder_name}/UMAP_rep_of_model_test.png')

# =============================================================================
# POSITIVE AUGMENTATIONS
# =============================================================================

training_pos_aug = []
training_neg_sample = []
training_neg_index = []

validation_pos_aug = []
validation_neg_sample = []
validation_neg_index = []

for batch_idx, (anchor_img, positive_img, negative_img, negative_index) in enumerate(train_loader):
    training_pos_aug.append(positive_img.detach().cpu())
    training_neg_sample.append(negative_img.detach().cpu())
    training_neg_index.append(negative_index)

for batch_idx, (anchor_img, positive_img, negative_img, negative_index) in enumerate(test_loader):
    validation_pos_aug.append(positive_img.detach().cpu())
    validation_neg_sample.append(negative_img.detach().cpu())
    validation_neg_index.append(negative_index)

# # Bokeh Plot
# list_of_images = []
# for batch_idx, (data) in enumerate(hard_dataloader):
#     data = data[0]
    
#     for image in data:
#         list_of_images.append(image)
        
# list_of_images = [tensor.numpy() for tensor in list_of_images]

# embeddable_images = simple_tweetyclr.get_images(list_of_images)

# simple_tweetyclr.plot_UMAP_embedding(embed, simple_tweetyclr.mean_colors_per_minispec[hard_indices,:],embeddable_images, f'{simple_tweetyclr.folder_name}/Plots/UMAP_Analysis.html', saveflag = True)

training_pos_aug_arr = torch.cat(training_pos_aug, dim = 0)
lab_train_pos = 0*np.ones((training_pos_aug_arr.shape[0]))

training_neg_sample_arr = torch.cat(training_neg_sample, dim = 0)
lab_train_neg = 1*np.ones((training_pos_aug_arr.shape[0]))

validation_pos_aug_arr = torch.cat(validation_pos_aug, dim = 0)
lab_val_pos = 2*np.ones((validation_pos_aug_arr.shape[0]))

validation_neg_sample_arr = torch.cat(validation_neg_sample, dim = 0)
lab_val_neg = 3*np.ones((validation_neg_sample_arr.shape[0]))

a = torch.cat((training_pos_aug_arr, training_neg_sample_arr, validation_pos_aug_arr, validation_neg_sample_arr), dim = 0)
b = np.concatenate((lab_train_pos,lab_train_neg,lab_val_pos ,lab_val_neg), axis = 0)

label_indicator = ["training positive aug", "training neg sample", "validation positive aug", "validation neg sample"]



aug_and_neg_dataset = TensorDataset(a, torch.tensor(b)) # The dataset of just the hard indices

aug_and_neg_loader = torch.utils.data.DataLoader(aug_and_neg_dataset, batch_size = batch_size, shuffle = False)

model_rep_aug_and_neg = []

model.to('cpu')
model.eval()
with torch.no_grad():
    for batch_idx, (dat, lab) in enumerate(aug_and_neg_loader):
        data = dat.to(torch.float32)
        output = model.module.forward_once(data)
        model_rep_aug_and_neg.append(output.numpy())

model_rep_aug_stacked = np.concatenate((model_rep_aug_and_neg))

import umap
reducer = umap.UMAP(metric = 'cosine', random_state=295) # For consistency

new_embed = reducer.fit_transform(model_rep_aug_stacked)

plt.figure()
unique_values = np.unique(b)
colors = plt.cm.jet(np.linspace(0, 1, len(unique_values)))  # Use a colormap
mean_colors = np.zeros((b.shape[0],4))

for i, color in zip(unique_values, colors):
    indices = np.where(b == i)[0]
    plt.scatter(new_embed[indices, 0], new_embed[indices, 1], color=color, alpha=0.7, label=f'{label_indicator[int(i)]}')
    mean_colors[indices,:] = color
    

plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("UMAP Decomposition")
plt.legend()
plt.savefig(f'{folder_name}/UMAP_embedding_of_augmentations.png')
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
    "Batch_Size": batch_size, 
    "Num_Epochs": num_epochs, 
    "Torch_Random_Seed": 295, 
    "Accumulation_Size": train_perc, 
    "Train_Proportion": train_perc,
    "Criterion": str(criterion), 
    "Model_Architecture": model_arch_lines, 
    "Forward_Method": forward_method_lines, 
    "Forward_Once_Method": forward_once_method_lines,
    "Dataloader_Shuffle": shuffle_status, 
    "Noise_Level": noise_level, 
    "Margin_Value": margin_value
    }

import json

with open(f'{simple_tweetyclr.folder_name}/experiment_params.json', 'w') as file:
    json.dump(experiment_params, file, indent=4)
    

# I am going to try an additional debugging step
# It is clear that UMAP is not able to represent the green and red differently.
# This is because the sample selection is not being tasked to contrast red 
# against green. However, the model should be able to have a good 
# representation for red that is translationally invariant. 
# I will do a UMAP decomposition on red samples only

# hard_labels = simple_tweetyclr.stacked_labels_for_window[hard_indices,:]

# # Find rows that contain 4
# rows_with_4 = np.any(hard_labels == 4, axis=1)

# # Find rows that do not contain 10
# rows_without_10 = ~np.any(hard_labels == 10, axis=1)

# # Combine conditions: rows that contain 4 and do not contain 10
# desired_rows = np.array([rows_with_4 & rows_without_10])
# desired_rows.shape = (desired_rows.shape[1],)

# model_rep_stacked_new = model_rep_stacked[desired_rows,:]
        
# reducer = umap.UMAP(random_state=295) # For consistency
# embed = reducer.fit_transform(model_rep_stacked_new)


# hard_colors = simple_tweetyclr.mean_colors_per_minispec[hard_indices,:]
# hard_colors = hard_colors[desired_rows,:]
# plt.figure()
# plt.scatter(embed[:,0], embed[:,1], c = hard_colors)
# plt.xlabel("UMAP 1")
# plt.ylabel("UMAP 2")
# plt.title("UMAP of the Representation Layer")
# plt.show()
# plt.savefig(f'{folder_name}/UMAP_rep_of_model.png')


# Bokeh Plot
list_of_images = []
for batch_idx, (data) in enumerate(aug_and_neg_loader):
    data = data[0]
    
    for image in data:
        list_of_images.append(image)
        
list_of_images = [tensor.numpy() for tensor in list_of_images]

embeddable_images = simple_tweetyclr.get_images(list_of_images)

simple_tweetyclr.plot_UMAP_embedding(new_embed, mean_colors ,embeddable_images, f'{simple_tweetyclr.folder_name}/Plots/UMAP_Analysis_augs_and_negs.html', saveflag = True)



