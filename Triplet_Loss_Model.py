#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 21:10:36 2024

This model will be a simpler version of UMAP + TweetyCLR. Here are the following modifications: 
    1. New model architecture 
    2. No temporal augmentation. Positive sample will be the spec slice that is closest to the specific hard index
    3. One easy negative sample only
    4. No train/test split for now

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

# stacked_windows[:, :, :] = simple_tweetyclr.stacked_labels_for_window[:, :, None]

stacked_windows.shape = (stacked_windows.shape[0],1, 100, 151) 

batch_size = 64

total_dataset, total_dataloader = create_dataloader(stacked_windows, batch_size, np.arange(stacked_windows.shape[0]))

# Need to compute the UMAP embedding
# reducer = umap.UMAP(metric = 'cosine', random_state=295)

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

hard_dataset, hard_dataloader = create_dataloader(hard_dataset, batch_size, hard_indices, shuffle_status=False)

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


# In[2]: I want to define a class where I select positive and negative samples. Here is the criterion for each
# Positive sample: A spectrogram slice that is closest in proximity to the anchor datapoint
# Negative sample: A spectrogram slice that is outside the hard_indices region

class APP_MATCHER(Dataset):
    def __init__(self, dataset, hard_dataset, total_hard_indices, umap_embedding, noise_scale = 0.5):
        ''' 
        The APP_MATCHER object should take in the entire dataset and the dataset of hard indices only. 
        The datasets should be Pytorch datasets. THe first component of each 
        dataset will be the actual flattened spectrogram slices. The second 
        component will be the index for each slice from the entire dataset. 
        This will be useful when selecting positive and negative samples.
        '''
        super(APP_MATCHER, self).__init__()
        
        # Extracting all slices and indices
        all_features, all_indices = zip(*[dataset[i] for i in range(len(dataset))]) # This will be used to create the all_possible_negatives region 
        
        hard_features, hard_indices = zip(*[hard_dataset[i] for i in range(len(hard_dataset))]) # This will be used to create all the hard features
        
        # Converting lists of tensors to a single tensor
        all_features, all_indices = torch.stack(all_features), torch.stack(all_indices)
        hard_features, hard_indices = torch.stack(hard_features), torch.stack(hard_indices)
        
        
        self.dataset = dataset
        self.hard_dataset = hard_dataset
        
        self.all_features = all_features
        self.all_indices = all_indices
        
        self.hard_features = hard_features
        self.hard_indices = hard_indices
        self.total_hard_indices = total_hard_indices
        self.umap_embedding = torch.tensor(umap_embedding)
        
        # Find the set of "easy" negatives
        mask = ~torch.isin(self.all_indices, torch.tensor(self.total_hard_indices))
        all_possible_negatives = self.all_indices[mask]
        
        self.all_possible_negatives = all_possible_negatives
        
        # Find the positive images
        dict_of_indices = self.select_positive_image()
        self.dict_of_indices = dict_of_indices
        self.noise_scale = noise_scale
        
    def select_positive_image(self):
        # Compute pairwise Euclidean distances
        # distances = torch.cdist(self.umap_embedding, self.umap_embedding, p=2)
        
        # # Dictionary to store the top 5 closest points for each data point
        # # TODO: I should also create a dictionary that goes from 0, ..., 470 and stores the hard_index. That way I can use index in the__getitem__ function to extract the top_5 spectrogram slices.
        # top_dict = {}
        dict_of_indices = {}
        
        # Iterate over each data point
        for i in range(self.umap_embedding.size(0)):
            # Exclude the distance to the point itself by setting it to a high value
            # distances[i, i] = float('inf')
        
            # Get indices of the top 5 closest points
            # _, closest_indices = torch.topk(distances[i], 5, largest=False)
        
            # Store the corresponding points in the dictionary
            # top_dict[i] = closest_indices
            
            dict_of_indices[i] = i 
            
        
        # Subset the dictionary to only see the values for the hard_indices keys
        # Keys you want to keep
        # keys_to_keep = self.hard_indices.tolist()
        
        # New dictionary with only the keys you want
        # selected_dict = {k: top_dict[k] for k in keys_to_keep if k in top_dict}
        # dict_of_indices = {k: dict_of_indices[k] for k in keys_to_keep if k in dict_of_indices}

        
        return dict_of_indices                      
        # return selected_dict, dict_of_indices
    
    def __len__(self):
        return self.hard_indices.shape[0]
    
    def __getitem__(self, index):
        
        ''' 
        The positive sample for each anchor image in the batch will be an augmented version of the anchor slice -- white noise
        The negative sample for each anchor image in the batch will be a randomly chosen spectrogram slice outside the hard region
        '''

        # The positive spectrogram slice will be the spectrogram slice that is closest in UMAP space to the anchor slice.
        dict_of_indices = self.dict_of_indices
        # keys = list(dict_of_indices.keys())
        actual_index = int(self.all_indices[int(self.hard_indices[index])])
        anchor_img = self.all_features[actual_index,:, :, :]
        
        # =============================================================================
        #         Positive Sample
        # =============================================================================
        
        positive_img = anchor_img.clone()
        # positive_img = self.all_features[actual_index+1,:,:,:]

        noisy_tensor = wn.white_noise(positive_img) 

        
        # # Clip values to be between 0 and 1
        positive_img = torch.clamp(noisy_tensor, 0, 1)
        
        # top_5_images = self.positive_dict[actual_index]
        
        # Generate a random index
        # random_index = torch.randint(0, top_5_images.size(0), (1,)).item()
        
        # Select the element at the random index
        # positive_index = top_5_images[random_index].item()
        # positive_img = self.all_features[positive_index, :, :, :]
                
        # =============================================================================
        #         Negative Sample
        # =============================================================================
        
        random_index = torch.randint(0, self.all_possible_negatives.size(0), (1,)).item()
        
        negative_index = self.all_possible_negatives[random_index].item()
        
        negative_img = self.all_features[negative_index, :, :, :]
        
        return anchor_img, positive_img, negative_img, negative_index
        

noise_level = 1.0

wn = Augmentations(noise_level = noise_level, num_augmentations=1)

train_dataset = APP_MATCHER(total_dataset, train_hard_dataset, hard_indices, embed_train, wn)   
test_dataset = APP_MATCHER(total_dataset, test_hard_dataset, hard_indices, embed_test, wn)   

shuffle_status = True

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = shuffle_status)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = shuffle_status)

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

        self.dropout_perc = dropout_perc
        self.fc_dimensionality = fc_dimensionality
        
        # self.fc = nn.Sequential(
        #     nn.Linear(320, 256),
        #     nn.ReLU(inplace=True), 
        #     nn.Linear(256, 1)
        # )  
        self.fc = nn.Sequential(
            nn.Linear(320, self.fc_dimensionality),
            nn.ReLU(inplace=False), 
        )  
        
        self.dropout = nn.Dropout(p=self.dropout_perc)
        
        # Initialize convolutional layers with He initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward_once(self, x):

        # No BatchNorm
        # x = F.relu((self.conv1(x)))
        # x = F.relu((self.conv2(x)))
        # x = F.relu((self.conv3(x)))
        # x = F.relu((self.conv4(x)))
        # x = F.relu((self.conv5(x)))
        # x = F.relu((self.conv6(x)))
        # x = F.relu((self.conv7(x)))
        # x = F.relu((self.conv8(x)))
        # x = F.relu((self.conv9(x)))
        # x = F.relu((self.conv10(x)))

        # BatchNorm 
        # x = self.relu(self.bn1(self.conv1(x)))
        # x = self.relu(self.bn2(self.conv2(x)))
        # x = self.relu(self.bn3(self.conv3(x)))
        # x = self.relu(self.bn4(self.conv4(x)))
        # x = self.relu(self.bn5(self.conv5(x)))
        # x = self.relu(self.bn6(self.conv6(x)))
        # x = self.relu(self.bn7(self.conv7(x)))
        # x = self.relu(self.bn8(self.conv8(x)))
        # x = self.relu(self.bn9(self.conv9(x)))
        # x = self.relu(self.bn10(self.conv10(x)))
        
        # LayerNorm
        # x = (self.relu(self.ln1(self.conv1(x))))
        # x = (self.relu(self.ln2(self.conv2(x))))
        # x = (self.relu(self.ln3(self.conv3(x))))
        # x = (self.relu(self.ln4(self.conv4(x))))
        # x = (self.relu(self.ln5(self.conv5(x))))
        # x = (self.relu(self.ln6(self.conv6(x))))
        # x = (self.relu(self.ln7(self.conv7(x))))
        # x = (self.relu(self.ln8(self.conv8(x))))
        # x = (self.relu(self.ln9(self.conv9(x))))
        # x = (self.relu(self.ln10(self.conv10(x))))
        
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
    
    def forward(self, anchor_img, positive_img, negative_img):
        
        # Pass the two spectrogram slices through a convolutional frontend to get a representation for each slice
        
        anchor_emb = self.relu(self.fc(self.forward_once(anchor_img)))
        positive_emb = self.relu(self.fc(self.forward_once(positive_img)))
        negative_emb = self.relu(self.fc(self.forward_once(negative_img)))
        
        return anchor_emb, positive_emb, negative_emb


fc_dimensionality = 256
dropout_perc = 0.25
model = Encoder(fc_dimensionality=fc_dimensionality, dropout_perc=dropout_perc)
# Check if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    model = nn.DataParallel(model)

# Move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).to(torch.float32)

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

# train_hard_loader = torch.utils.data.DataLoader(train_hard_dataset, batch_size = batch_size, shuffle = False)
# model_rep_untrained = create_UMAP_plot(train_hard_loader, simple_tweetyclr, hard_indices[train_indices], model, 'UMAP_rep_of_model_train_region_untrained_model', saveflag = True)


# =============================================================================
# MODEL BUILDING
# =============================================================================

step = 0
max_steps = 100

train_iter = iter(train_loader)
test_iter = iter(test_loader)

training_epoch_loss = []
training_batch_loss = []
validation_epoch_loss = []
validation_batch_loss = []

training_pos_aug = []
training_neg_sample = []

validation_pos_aug = []
validation_neg_sample = []


while step < max_steps:
    model.to(device).to(torch.float32)

    try:
        anchor_img, positive_img, negative_img, negative_index = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        anchor_img, positive_img, negative_img, negative_index = next(train_iter)


    # Move batch to device once and ensure it's float32
    anchor_img, positive_img, negative_img = anchor_img.to(device, dtype = torch.float32), positive_img.to(device, dtype = torch.float32), negative_img.to(device, dtype = torch.float32)
    
    optimizer.zero_grad()
    anchor_emb, positive_emb, negative_emb = model(anchor_img, positive_img, negative_img)
    # training_pos_aug.append(positive_img.detach().cpu())
    # training_neg_sample.append(negative_img.detach().cpu())
    loss = criterion(anchor_emb, positive_emb, negative_emb)
    training_batch_loss.append(loss.item())
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        try:
            anchor_img, positive_img, negative_img, negative_index = next(test_iter)
        except StopIteration:
            test_iter = iter(test_loader)
            anchor_img, positive_img, negative_img, negative_index = next(test_iter)
            print(f'Epoch {step}, Last Training Batch Loss: {training_batch_loss[-1]}, Last Validation Batch Loss: {validation_batch_loss[-1]}')
            step +=1


        # Move batch to device once and ensure it's float32
        anchor_img, positive_img, negative_img = anchor_img.to(device, dtype = torch.float32), positive_img.to(device, dtype = torch.float32), negative_img.to(device, dtype = torch.float32)
        
        anchor_emb, positive_emb, negative_emb = model(anchor_img, positive_img, negative_img)
        # training_pos_aug.append(positive_img.detach().cpu())
        # training_neg_sample.append(negative_img.detach().cpu())
        loss = criterion(anchor_emb, positive_emb, negative_emb)

        validation_batch_loss.append(loss.item())

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

# # =============================================================================
# # POSITIVE AUGMENTATIONS
# # =============================================================================

# training_pos_aug = []
# training_neg_sample = []
# training_neg_index = []

# validation_pos_aug = []
# validation_neg_sample = []
# validation_neg_index = []

# for batch_idx, (anchor_img, positive_img, negative_img, negative_index) in enumerate(train_loader):
#     training_pos_aug.append(positive_img.detach().cpu())
#     training_neg_sample.append(negative_img.detach().cpu())
#     training_neg_index.append(negative_index)

# for batch_idx, (anchor_img, positive_img, negative_img, negative_index) in enumerate(test_loader):
#     validation_pos_aug.append(positive_img.detach().cpu())
#     validation_neg_sample.append(negative_img.detach().cpu())
#     validation_neg_index.append(negative_index)

# # # Bokeh Plot
# # list_of_images = []
# # for batch_idx, (data) in enumerate(hard_dataloader):
# #     data = data[0]
    
# #     for image in data:
# #         list_of_images.append(image)
        
# # list_of_images = [tensor.numpy() for tensor in list_of_images]

# # embeddable_images = simple_tweetyclr.get_images(list_of_images)

# # simple_tweetyclr.plot_UMAP_embedding(embed, simple_tweetyclr.mean_colors_per_minispec[hard_indices,:],embeddable_images, f'{simple_tweetyclr.folder_name}/Plots/UMAP_Analysis.html', saveflag = True)

# training_pos_aug_arr = torch.cat(training_pos_aug, dim = 0)
# lab_train_pos = 0*np.ones((training_pos_aug_arr.shape[0]))

# training_neg_sample_arr = torch.cat(training_neg_sample, dim = 0)
# lab_train_neg = 1*np.ones((training_pos_aug_arr.shape[0]))

# validation_pos_aug_arr = torch.cat(validation_pos_aug, dim = 0)
# lab_val_pos = 2*np.ones((validation_pos_aug_arr.shape[0]))

# validation_neg_sample_arr = torch.cat(validation_neg_sample, dim = 0)
# lab_val_neg = 3*np.ones((validation_neg_sample_arr.shape[0]))

# a = torch.cat((training_pos_aug_arr, training_neg_sample_arr, validation_pos_aug_arr, validation_neg_sample_arr), dim = 0)
# b = np.concatenate((lab_train_pos,lab_train_neg,lab_val_pos ,lab_val_neg), axis = 0)

# label_indicator = ["training positive aug", "training neg sample", "validation positive aug", "validation neg sample"]



# aug_and_neg_dataset = TensorDataset(a, torch.tensor(b)) # The dataset of just the hard indices

# aug_and_neg_loader = torch.utils.data.DataLoader(aug_and_neg_dataset, batch_size = batch_size, shuffle = False)

# model_rep_aug_and_neg = []

# model.to('cpu')
# model.eval()
# with torch.no_grad():
#     for batch_idx, (dat, lab) in enumerate(aug_and_neg_loader):
#         data = dat.to(torch.float32)
#         output = model.module.forward_once(data)
#         model_rep_aug_and_neg.append(output.numpy())

# model_rep_aug_stacked = np.concatenate((model_rep_aug_and_neg))

# import umap
# reducer = umap.UMAP(metric = 'cosine', random_state=295) # For consistency

# new_embed = reducer.fit_transform(model_rep_aug_stacked)

# plt.figure()
# unique_values = np.unique(b)
# colors = plt.cm.jet(np.linspace(0, 1, len(unique_values)))  # Use a colormap
# mean_colors = np.zeros((b.shape[0],4))

# for i, color in zip(unique_values, colors):
#     indices = np.where(b == i)[0]
#     plt.scatter(new_embed[indices, 0], new_embed[indices, 1], color=color, alpha=0.7, label=f'{label_indicator[int(i)]}')
#     mean_colors[indices,:] = color
    

# plt.xlabel("UMAP 1")
# plt.ylabel("UMAP 2")
# plt.title("UMAP Decomposition")
# plt.legend()
# plt.savefig(f'{folder_name}/UMAP_embedding_of_augmentations.png')
# plt.show()

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
# list_of_images = []
# for batch_idx, (data) in enumerate(aug_and_neg_loader):
#     data = data[0]
    
#     for image in data:
#         list_of_images.append(image)
        
# list_of_images = [tensor.numpy() for tensor in list_of_images]

# embeddable_images = simple_tweetyclr.get_images(list_of_images)

# simple_tweetyclr.plot_UMAP_embedding(new_embed, mean_colors ,embeddable_images, f'{simple_tweetyclr.folder_name}/Plots/UMAP_Analysis_augs_and_negs.html', saveflag = True)



