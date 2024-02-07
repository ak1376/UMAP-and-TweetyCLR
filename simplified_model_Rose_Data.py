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
import os
# os.chdir('/Users/AnanyaKapoor/Downloads/TweetyCLR')
os.chdir('/home/akapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_b_Canary_SSL/TweetyCLR_End_to_End')
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

directory = '/home/akapoor/Dropbox (University of Oregon)/AK_RHV_Analysis/AreaX_lesions/USA5347/13/Python_Files/songs'

# Identify the upstream location where the results will be saved. 
analysis_path = f'{filepath}/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_b_Canary_SSL/TweetyCLR_Repo/'

# # Parameters we set
num_spec = 250
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
    
category_colors = {}
category_colors[0] = np.array([255, 102, 102])/255

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
batch_size = 128
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


# Need to compute the UMAP embedding
reducer = umap.UMAP(metric = 'cosine', random_state=295)
# reducer = umap.UMAP(metric = 'cosine')
# reducer = umap.UMAP(random_state = 295)

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

data_for_analysis = simple_tweetyclr.stacked_windows.copy()
data_for_analysis = data_for_analysis.reshape(data_for_analysis.shape[0], 1, 100, 151)
total_dataset = TensorDataset(torch.tensor(data_for_analysis.reshape(data_for_analysis.shape[0], 1, simple_tweetyclr.time_dim, simple_tweetyclr.freq_dim)))
total_dataloader = DataLoader(total_dataset, batch_size=batch_size , shuffle=False)

list_of_images = []
for batch_idx, (data) in enumerate(total_dataloader):
    data = data[0]
    
    for image in data:
        list_of_images.append(image)
        
list_of_images = [tensor.numpy() for tensor in list_of_images]

embeddable_images = simple_tweetyclr.get_images(list_of_images)

simple_tweetyclr.plot_UMAP_embedding(embed, simple_tweetyclr.mean_colors_per_minispec,embeddable_images, f'{simple_tweetyclr.folder_name}/Plots/UMAP_of_all_slices.html', saveflag = True)


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



hard_dataset = data_for_analysis[hard_indices,:].reshape(len(hard_indices), 1, 100, 151) # Dataset of all the confused spectrogram slices that we want to untangle

dataset = TensorDataset(torch.tensor(data_for_analysis), torch.tensor(np.arange(data_for_analysis.shape[0]))) # The full dataset of all slices
hard_dataset = TensorDataset(torch.tensor(hard_dataset), torch.tensor(hard_indices)) # The dataset of just the hard indices


# In[2]: I want to define a class where I select positive and negative samples. Here is the criterion for each
# Positive sample: A spectrogram slice that is closest in proximity to the anchor datapoint
# Negative sample: A spectrogram slice that is outside the hard_indices region

class APP_MATCHER(Dataset):
    def __init__(self, dataset, hard_dataset, umap_embedding, noise_scale = 0.5):
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
        self.umap_embedding = torch.tensor(umap_embedding)
        
        # Find the set of "easy" negatives
        mask = ~torch.isin(self.all_indices, self.hard_indices)
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
        # TODO: Revise this function so that I don't get KeyErrors anymore
        
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
        
        # Define the noise scale (e.g., 5% of the data range)
        noise_scale = self.noise_scale        
        # # Generate uniform noise and scale it
        noise = torch.rand_like(positive_img) * noise_scale
        
        # # Add the noise to the original tensor
        noisy_tensor = positive_img + noise
        
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
        
        return anchor_img, positive_img, negative_img
        

# Split the dataset into a training and testing dataset
# Define the split sizes -- what is the train test split ? 
train_perc = 0.5 #
train_size = int(train_perc * len(hard_dataset))  # (100*train_perc)% for training
test_size = len(hard_dataset) - train_size  # 100 - (100*train_perc)% for testing

from torch.utils.data import random_split

train_hard_dataset, test_hard_dataset = random_split(hard_dataset, [train_size, test_size]) 


# I want to randomly select training anchors (which should be 25% of the data).
# Upon selecting these anchors I will then find the positive temporal samples 
# for each of these anchors. The set of the training anchor and training 
# positive samples will be excluded from the total set of possible validation
# anchor and positive samples.

# from torch.utils.data import random_split, Subset

# # Define the train percentage
# train_perc = 0.25

# # Calculate the size of the train and test sets
# train_size = int(train_perc * len(hard_dataset))
# test_size = len(hard_dataset) - train_size

# # Split the dataset into train and test sets
# train_hard_dataset, _ = random_split(hard_dataset, [train_size, test_size])

# # Assume hard_indices is a list or tensor of all indices in hard_dataset
# all_indices = set(range(len(hard_dataset)))

# # Get the indices from the train subset; no need to cast to tensor again
# indices_to_exclude = set(train_hard_dataset.indices)

# # Now, assuming you want to exclude the indices of train set and their immediate next indices
# indices_to_exclude.update([i + 1 for i in train_hard_dataset.indices if i + 1 < len(hard_dataset)])

# # Determine the indices to include (all indices minus the ones to exclude)
# indices_to_include = list(all_indices - indices_to_exclude)

# # Create the subset without the excluded indices
# test_hard_dataset = Subset(hard_dataset, indices_to_include)

# Getting the indices
train_indices = np.array(train_hard_dataset.indices)
test_indices = np.array(test_hard_dataset.indices)
embed_train = embed[hard_indices[train_indices],:]
embed_test = embed[hard_indices[test_indices], :]

train_dataset = APP_MATCHER(dataset, train_hard_dataset, embed_train, noise_scale = 1.0)   
test_dataset = APP_MATCHER(dataset, test_hard_dataset, embed_test, noise_scale = 1.0)   

shuffle_status = True

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = shuffle_status)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = shuffle_status)

a = next(iter(test_loader))

model = Encoder()
# Check if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    model = nn.DataParallel(model)

# Move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).to(torch.float32)

criterion = nn.TripletMarginLoss(margin=1.0, p=2)
# optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 100
patience = 30  # Number of epochs to wait for improvement before stopping
min_delta = 0.001  # Minimum change to qualify as an improvement

best_val_loss = float('inf')
epochs_no_improve = 0
early_stop = False

training_epoch_loss = []
training_batch_loss = []
validation_epoch_loss = []
validation_batch_loss = []
for epoch in np.arange(num_epochs):
    model.to(device).to(torch.float32)
    model.train()
    training_loss = 0
    for batch_idx, (anchor_img, positive_img, negative_img) in enumerate(train_loader):
        anchor_img, positive_img, negative_img = anchor_img.to(device, dtype = torch.float32), positive_img.to(device, dtype = torch.float32), negative_img.to(device, dtype = torch.float32)
        
        optimizer.zero_grad()
        anchor_emb, positive_emb, negative_emb = model(anchor_img, positive_img, negative_img)
        loss = criterion(anchor_emb, positive_emb, negative_emb)
        training_loss+=loss.item()
        training_batch_loss.append(loss.item())
        loss.backward()
        optimizer.step()

    
#     # DEBUGGING STEP: LOOKING AT THE MODEL REPRESENTATIONS
# # =============================================================================
# #     # TRAINING
# # =============================================================================
    
#     if epoch%5 == 0:
#         model_rep_train = []
#         model_for_eval = model.module
#         train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = False)
#         # model_for_eval = model_for_eval.to('cpu')
#         model_for_eval.eval()
        
#         with torch.no_grad():
#             for batch_idx, (anchor_img, positive_img, negative_img) in enumerate(train_loader):
#                 data = anchor_img.to(torch.float32).to(device)
#                 output = model_for_eval.forward_once(data)
#                 model_rep_train.append(output.cpu().detach().numpy())
        
#         model_rep_stacked = np.concatenate((model_rep_train))
        
#         reducer = umap.UMAP(metric = 'cosine', random_state=295) # For consistency
#         embed_train = reducer.fit_transform(model_rep_stacked)
        
#         plt.figure()
#         plt.scatter(embed_train[:,0], embed_train[:,1], c = simple_tweetyclr.mean_colors_per_minispec[hard_indices[train_indices],:])
#         plt.xlabel("UMAP 1")
#         plt.ylabel("UMAP 2")
#         plt.suptitle(f'UMAP Representation of Training Hard Region')
#         plt.title(f'Total Slices: {embed_train.shape[0]}')
#         plt.show()
#         plt.savefig(f'{folder_name}/UMAP_rep_of_model_train_epoch_{epoch}.png')

    # model.to(device).to(torch.float32)
    model.eval()
    with torch.no_grad():
        validation_loss = 0
        for batch_idx, (anchor_img, positive_img, negative_img) in enumerate(test_loader):
            anchor_img, positive_img, negative_img = anchor_img.to(device, dtype = torch.float32), positive_img.to(device, dtype = torch.float32), negative_img.to(device, dtype = torch.float32)
            anchor_emb, positive_emb, negative_emb = model(anchor_img, positive_img, negative_img)
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

plt.figure()
plt.plot(training_batch_loss, label = 'Training Loss')
plt.plot(validation_batch_loss, label = 'Validation Loss')
plt.legend()
plt.xlabel("Batch Number")
plt.ylabel("Triplet Loss")
plt.title("Unsupervised Training")
plt.savefig(f'{folder_name}/loss_curve.png')
plt.show()


# model_rep = []
# total_dataset = TensorDataset(torch.tensor(simple_tweetyclr.stacked_windows.reshape(simple_tweetyclr.stacked_windows.shape[0], 1, 100, 151)), torch.tensor(simple_tweetyclr.stacked_labels_for_window))

# hard_stacked_windows = simple_tweetyclr.stacked_windows[hard_indices,:]
# hard_stacked_windows_train = simple_tweetyclr.stacked_windows[train_indices,:]
# hard_stacked_windows_test = simple_tweetyclr.stacked_windows[test_indices,:]


# hard_dataset = TensorDataset(torch.tensor(hard_stacked_windows.reshape(hard_stacked_windows.shape[0], 1, simple_tweetyclr.time_dim, simple_tweetyclr.freq_dim)))
# hard_dataloader = DataLoader(hard_dataset, batch_size=batch_size , shuffle=False)

hard_dataset_train = TensorDataset(torch.tensor(simple_tweetyclr.stacked_windows[hard_indices[train_indices],:].reshape(simple_tweetyclr.stacked_windows[hard_indices[train_indices],:].shape[0], 1, simple_tweetyclr.time_dim, simple_tweetyclr.freq_dim)))
hard_dataloader_train = DataLoader(hard_dataset_train, batch_size=batch_size , shuffle=False)


hard_dataset_test = TensorDataset(torch.tensor(simple_tweetyclr.stacked_windows[hard_indices[test_indices],:].reshape(simple_tweetyclr.stacked_windows[hard_indices[test_indices],:].shape[0], 1, simple_tweetyclr.time_dim, simple_tweetyclr.freq_dim)))
hard_dataloader_test = DataLoader(hard_dataset_test, batch_size=batch_size , shuffle=False)



# total_dat = APP_MATCHER(total_dataset)
# total_dataloader = torch.utils.data.DataLoader(total_dat, batch_size = batch_size, shuffle = shuffle_status)
# model_rep = []

# model = model.to('cpu')
# model.eval()
# with torch.no_grad():
#     for batch_idx, data in enumerate(hard_dataloader):
#         data = data[0]
#         data = data.to(torch.float32)
#         output = model.module.forward_once(data)
#         model_rep.append(output.numpy())

# model_rep_stacked = np.concatenate((model_rep))

# import umap
# reducer = umap.UMAP(random_state=295) # For consistency
# embed = reducer.fit_transform(model_rep_stacked)

# plt.figure()
# plt.scatter(embed[:,0], embed[:,1], c = simple_tweetyclr.mean_colors_per_minispec[hard_indices,:])
# plt.xlabel("UMAP 1")
# plt.ylabel("UMAP 2")
# plt.title("UMAP of the Representation Layer")
# plt.show()
# plt.savefig(f'{folder_name}/UMAP_rep_of_model.png')


# =============================================================================
# TRAINING DATA
# =============================================================================

model_rep_train = []
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = False)

model = model.to('cpu')
model.eval()
with torch.no_grad():
    for batch_idx, (anchor_img, positive_img, negative_img) in enumerate(train_loader):
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
    for batch_idx, (anchor_img, positive_img, negative_img) in enumerate(test_loader):
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


# Bokeh Plot
list_of_images = []
for batch_idx, (data) in enumerate(hard_dataloader_train):
    data = data[0]
    
    for image in data:
        list_of_images.append(image)
        
list_of_images = [tensor.numpy() for tensor in list_of_images]

embeddable_images = simple_tweetyclr.get_images(list_of_images)

simple_tweetyclr.plot_UMAP_embedding(embed_train, simple_tweetyclr.mean_colors_per_minispec[hard_indices[train_indices],:],embeddable_images, f'{simple_tweetyclr.folder_name}/Plots/UMAP_Analysis_hard_train.html', saveflag = True)

# Bokeh Plot
list_of_images = []
for batch_idx, (data) in enumerate(hard_dataloader_test):
    data = data[0]
    
    for image in data:
        list_of_images.append(image)
        
list_of_images = [tensor.numpy() for tensor in list_of_images]

embeddable_images = simple_tweetyclr.get_images(list_of_images)

simple_tweetyclr.plot_UMAP_embedding(embed_test, simple_tweetyclr.mean_colors_per_minispec[hard_indices[test_indices],:],embeddable_images, f'{simple_tweetyclr.folder_name}/Plots/UMAP_Analysis_hard_train.html', saveflag = True)

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
    "Dataloader_Shuffle": shuffle_status
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


