#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:40:41 2023

@author: akapoor
"""

import numpy as np
import torch
import sys
filepath = '/Users/AnanyaKapoor'
sys.path.append(f'{filepath}/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_b_Canary_SSL/TweetyCLR/')
from util import MetricMonitor, SupConLoss
from util import Tweetyclr, Temporal_Augmentation, TwoCropTransform, Custom_Contrastive_Dataset
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle
from torch.utils.data import DataLoader, TensorDataset
import umap
import matplotlib.pyplot as plt
import torch.optim as optim
import matplotlib.colors as mcolors
import matplotlib.cm as cm

plt.rcParams.update({'font.size': 20})
plt.rcParams['figure.figsize'] = [15, 15]  # width and height should be in inches, e.g., [10, 6]

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
        # self.conv11 = nn.Conv2d(16, 8, 3, 1, padding = 1)
        # self.conv12 = nn.Conv2d(8, 8, 3, 2, padding = 1)
        
        
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
        # self.bn11 = nn.BatchNorm2d(16)
        # self.bn12 = nn.BatchNorm2d(8)

        self.relu = nn.ReLU()       
        self.dropout = nn.Dropout2d(
        )
        # self.fc = nn.Linear(320, 32)
        # self._to_linear = 1280
        # self._to_linear = 320
        self._to_linear = 320
        
    def forward(self, x):
         
        # x = F.relu(self.dropout(self.conv1(self.bn1(x))))
        # x = F.relu(self.conv2(self.bn2(x))) 
        # x = F.relu(self.dropout(self.conv3(self.bn3(x))))
        # x = F.relu(self.conv4(self.bn4(x))) 
        # x = F.relu(self.dropout(self.conv5(self.bn5(x))))
        # x = F.relu(self.conv6(self.bn6(x))) 
        # x = F.relu(self.dropout(self.conv7(self.bn7(x))))
        # x = F.relu(self.conv8(self.bn8(x)))
        # x = F.relu(self.dropout(self.conv9(self.bn9(x))))
        # x = F.relu(self.conv10(self.bn10(x)))
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))) 
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x))) 
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x))) 
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))
        # x = F.relu(self.conv11(self.bn11(x)))
        # x = F.relu(self.conv12(self.bn12(x)))

        # x = x.view(-1, 48)
        x = x.view(-1, 320)
        # x = self.fc(x)
        # x = self.relu(x)
        # x = x.view(-1, 32)
        # x = x.view(-1, 1280) #Window size = 500
        
        return x

def pretraining(epoch, model, contrastive_loader_train, contrastive_loader_test, optimizer, criterion, epoch_number, method='SimCLR'):
    "Contrastive pre-training over an epoch. Adapted from XX"
    negative_similarities_for_epoch = []
    ntxent_positive_similarities_for_epoch = []
    mean_pos_cos_sim_for_epoch = []
    mean_ntxent_positive_similarities_for_epoch = []
    metric_monitor = MetricMonitor()
    
    # TRAINING PHASE
    
    model.train()

    for batch_data in enumerate(contrastive_loader_train):
        data_list = []
        label_list = []
        a = batch_data[1]
        for idx in np.arange(len(a)):
            data_list.append(a[idx][0])
        
        
    # for batch_idx, ((data1, labels1), (data2, labels2)) in enumerate(contrastive_loader):
        data = torch.cat((data_list), dim = 0)
        data = data.unsqueeze(1)
        # data = data.reshape(a[idx][0].shape[0], len(a), a[idx][0].shape[1], a[idx][0].shape[2])
        # labels = labels1
        if torch.cuda.is_available():
            data = data.cuda()
        data = torch.autograd.Variable(data,False)
        bsz = a[idx][0].shape[0]
        data = data.to(torch.float32)
        features = model(data)
        norm = features.norm(p=2, dim=1, keepdim=True)
        epsilon = 1e-12
        # Add a small epsilon to prevent division by zero
        normalized_tensor = features / (norm + epsilon)
        split_features = torch.split(normalized_tensor, [bsz]*len(a), dim=0)
        split_features = [split.unsqueeze(1) for split in split_features]

        features = torch.cat(split_features, dim = 1)

        training_loss, training_negative_similarities, training_positive_similarities = criterion(features)

        metric_monitor.update("Training Loss", training_loss.item())
        metric_monitor.update("Learning Rate", optimizer.param_groups[0]['lr'])
        
        if epoch_number !=0:
            optimizer.zero_grad()
            training_loss.backward()
            optimizer.step()

        negative_similarities_for_epoch.append(float(np.mean(training_negative_similarities.clone().detach().cpu().numpy())))

        ntxent_positive_similarities_for_epoch.append(float(np.mean(training_positive_similarities.clone().detach().cpu().numpy())))
        
        # # Calculate the mean cosine similarity of the model feature representation for the positive pairs.
        # # Slice the tensor to separate the two sets of features you want to compare
        
        split_features = torch.split(normalized_tensor, [bsz]*len(a), dim=0)
        split_features = [split.unsqueeze(1) for split in split_features]
        training_features = torch.cat(split_features, dim = 1)
        
        # VALIDATION PHASE
        
        model.eval()

        for batch_data in enumerate(contrastive_loader_test):
            data_list = []
            label_list = []
            a = batch_data[1]
            for idx in np.arange(len(a)):
                data_list.append(a[idx][0])
            
        # for batch_idx, ((data1, labels1), (data2, labels2)) in enumerate(contrastive_loader):
            data = torch.cat((data_list), dim = 0)
            data = data.unsqueeze(1)
            # data = data.reshape(a[idx][0].shape[0], len(a), a[idx][0].shape[1], a[idx][0].shape[2])
            # labels = labels1
            if torch.cuda.is_available():
                data = data.cuda()
            data = torch.autograd.Variable(data,False)
            bsz = a[idx][0].shape[0]
            data = data.to(torch.float32)
            features = model(data)
            norm = features.norm(p=2, dim=1, keepdim=True)
            epsilon = 1e-12
            # Add a small epsilon to prevent division by zero
            normalized_tensor = features / (norm + epsilon)
            split_features = torch.split(normalized_tensor, [bsz]*len(a), dim=0)
            split_features = [split.unsqueeze(1) for split in split_features]

            validation_features = torch.cat(split_features, dim = 1)

            validation_loss, validation_negative_similarities, validation_positive_similarities = criterion(validation_features)
            
            metric_monitor.update("Validation Loss", validation_loss.item())


    # print("[Epoch: {epoch:03d}] Contrastive Pre-train | {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor))
    print(f'Epoch: {epoch:03d} Contrastive Pre-train Loss: {training_loss:.3f}, Validation Loss: {validation_loss:.3f}')
    # return metric_monitor.metrics['Loss']['avg'], metric_monitor.metrics['Learning Rate']['avg'], negative_similarities_for_epoch, features, mean_pos_cos_sim_for_epoch, mean_ntxent_positive_similarities_for_epoch
    return metric_monitor.metrics['Training Loss']['avg'], metric_monitor.metrics['Validation Loss']['avg'], metric_monitor.metrics['Learning Rate']['avg'], training_features, validation_features, negative_similarities_for_epoch, ntxent_positive_similarities_for_epoch

# =============================================================================
#     # Set data parameters
# =============================================================================


# REAL CANARY DATA 

bird_dir = f'{filepath}/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_a_Rotations/Gardner_Lab/Canary_Data/llb16_data_matrices/'
bird_dir = f'{filepath}/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_a_Rotations/Gardner_Lab/Canary_Data/llb16_data_matrices/'
audio_files = bird_dir+'llb3_songs'
directory = bird_dir+ 'Python_Files'

analysis_path = f'{filepath}/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_b_Canary_SSL/TweetyCLR_Repo/'
# analysis_path = '/home/akapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_b_Canary_SSL/TweetyCLR_Repo/'

# # Parameters we set
num_spec = 80
window_size = 100
stride = 10

# Define the folder name
folder_name = f'{analysis_path}Num_Spectrograms_{num_spec}_Window_Size_{window_size}_Stride_{stride}'


files = os.listdir(directory)
all_songs_data = [f'{directory}/{element}' for element in files if '.npz' in element] # Get the file paths of each numpy file from Yarden's data
all_songs_data.sort()

masking_freq_tuple = (500, 7000)
spec_dim_tuple = (window_size, 151)


with open(f'{filepath}/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_b_Canary_SSL/Canary_SSL_Repo/InfoNCE_Num_Spectrograms_100_Window_Size_100_Stride_10/category_colors.pkl', 'rb') as file:
    category_colors = pickle.load(file)
    
exclude_transitions = False

# SYNTHETIC CANARY DATA

# phrase_repeats = 5
# num_songs = 25
# radius_value = 0.01
# num_syllables = 10

# folderpath = f'{filepath}/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_b_Canary_SSL/TweetyCLR_Repo/'
# songpath = f'{folderpath}num_songs_{num_songs}_num_syllables_{num_syllables}_phrase_repeats_{phrase_repeats}_radius_{radius_value}/'

# # For each spectrogram we will extract
# # 1. Each timepoint's syllable label
# # 2. The spectrogram itself
# stacked_labels = [] 
# stacked_specs = []
# spectrogram_id = [] 

# all_songs_data = [element for element in os.listdir(songpath)  if 'Song' in element] 
# all_songs_data.sort()
# all_songs_data = [f'{songpath}{element}/synthetic_data.npz' for element in all_songs_data]


# # files = os.listdir(directory)
# # all_songs_data = [f'{directory}/{element}' for element in files if '.npz' in element] # Get the file paths of each numpy file from Yarden's data
# # all_songs_data.sort()

# masking_freq_tuple = (500, 13500)
# spec_dim_tuple = (window_size, 151)


# with open(f'{filepath}/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_b_Canary_SSL/Canary_SSL_Repo/InfoNCE_Num_Spectrograms_100_Window_Size_100_Stride_10/category_colors.pkl', 'rb') as file:
#     category_colors = pickle.load(file)
    
# exclude_transitions = False

# num_spec = 10
# folder_name = f'{folderpath}Synthetic_Analysis/Num_Spectrograms_{num_spec}_Window_Size_{window_size}_Stride_{stride}'

# =============================================================================
#     # Set model parameters
# =============================================================================

easy_negatives = 1
hard_negatives = 1
batch_size = easy_negatives + hard_negatives + 1
num_epochs = 10
tau_in_steps = 3
temp_value = 0.02
method = 'SimCLR'
device = 'cuda'
use_scheduler = True

# =============================================================================
#     # Initialize the TweetyCLR object
# =============================================================================
simple_tweetyclr_experiment_1 = Tweetyclr(num_spec, window_size, stride, folder_name, all_songs_data, masking_freq_tuple, spec_dim_tuple, exclude_transitions, category_colors = category_colors)

simple_tweetyclr = simple_tweetyclr_experiment_1
simple_tweetyclr.first_time_analysis()

simple_tweetyclr.easy_negatives = easy_negatives
simple_tweetyclr.hard_negatives = hard_negatives 


# Let's colorize the entire stacked spectrogram by the ground truth labels at each point

dx = simple_tweetyclr.stacked_window_times[0,1] - simple_tweetyclr.stacked_window_times[0,0] 

stacked_times = dx*np.arange(simple_tweetyclr.stacked_specs.shape[1])
stacked_times = stacked_times.reshape(1, stacked_times.shape[0])

# dat = np.load(all_songs_data[0])
dat = np.load(f'{filepath}/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_a_Rotations/Gardner_Lab/Canary_Data/llb16_data_matrices/Python_Files/llb16_0032_2018_05_03_18_22_58.wav.npz')
freq = dat['f']

# Let's get rid of higher order frequencies
mask = (freq<masking_freq_tuple[1])&(freq>masking_freq_tuple[0])
masked_frequencies = freq[mask].reshape(151,1)

# Let's do raw umap

# reducer = umap.UMAP(metric = 'cosine', random_state=295)
reducer = umap.UMAP(metric = 'cosine')

# embed = reducer.fit_transform(simple_tweetyclr.stacked_windows)
# Preload the embedding 
embed = np.load(f'{simple_tweetyclr.folder_name}/embed_80_specs.npy')
simple_tweetyclr.umap_embed_init = embed

plt.figure()
plt.scatter(embed[:,0], embed[:,1], s = 10, c = simple_tweetyclr.mean_colors_per_minispec)
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
# plt.title("UMAP Decomposition of ")
# plt.suptitle("UMAP Representation of All Spectrogram Slices")
plt.title(f'Total Slices: {embed.shape[0]}')
plt.savefig(f'{simple_tweetyclr.folder_name}/Plots/UMAP_of_all_slices.png')
plt.show()

# save the umap embedding for easy future loading
# np.save('/home/akapoor/Desktop/embed.npy', embed)

# DEFINE HARD INDICES THROUGH INTERACTION: USER NEEDS TO ZOOM IN ON ROI 

# Get current axes
ax = plt.gca()
# Get current limits
xlim = ax.get_xlim()
ylim = ax.get_ylim()

hard_indices = np.where((embed[:,0]>=xlim[0])&(embed[:,0]<=xlim[1]) & (embed[:,1]>=ylim[0]) & (embed[:,1]<=ylim[1]))[0]

plt.figure(figsize = (10,10))
plt.scatter(embed[hard_indices,0], embed[hard_indices,1], s = 10, c = simple_tweetyclr.mean_colors_per_minispec[hard_indices,:])
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
# plt.title("UMAP Decomposition of ")
# plt.suptitle("UMAP Representation of All Spectrogram Slices")
plt.title(f'Total Slices: {embed[hard_indices,:].shape[0]}')
plt.savefig(f'{simple_tweetyclr.folder_name}/Plots/UMAP_of_hard_slices.png')
plt.show()

simple_tweetyclr.hard_indices = hard_indices

dataset = simple_tweetyclr.stacked_windows.copy()

# These will give the training and testing ANCHORS
stacked_windows_train, stacked_labels_train, mean_colors_per_minispec_train, hard_indices_train, stacked_windows_test, stacked_labels_test, mean_colors_per_minispec_test, hard_indices_test = simple_tweetyclr.train_test_split(dataset, 0.8, hard_indices)

data_for_analysis = dataset.copy() 

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

# =============================================================================
# # Hand select negative samples
# =============================================================================

# Some negative samples need to be hard. I will do this by for each sample in 
# the bounding box, selecting spectrogram slices within the bounding box that 
# have a low cosine similarity. 

training_indices, stacked_windows_train = simple_tweetyclr.negative_sample_selection(data_for_analysis, hard_indices_train, None)

# Let's eliminate the easy training negatives from the selection of easy negatives for the testing dataset
all_easy_indices = np.array(list(set(np.arange(data_for_analysis.shape[0])) - set(hard_indices)))
training_easy_negatives = np.intersect1d(training_indices, all_easy_indices)

testing_indices, stacked_windows_test = simple_tweetyclr.negative_sample_selection(data_for_analysis, hard_indices_test, training_easy_negatives)

dict_of_spec_slices_with_slice_number = {i: data_for_analysis[i, :] for i in range(data_for_analysis.shape[0])}

items = list(dict_of_spec_slices_with_slice_number.items())

# =============================================================================
#     # Create contrastive dataloaders
# =============================================================================
    
augmentation_object = Temporal_Augmentation(dict_of_spec_slices_with_slice_number , simple_tweetyclr, tau_in_steps=tau_in_steps)

custom_transformation = TwoCropTransform(augmentation_object)

# Training contrastive loader
# # Your new_contrastive_dataset initialization would be:
new_contrastive_dataset = Custom_Contrastive_Dataset(stacked_windows_train, torch.tensor(training_indices), custom_transformation)

# # DataLoader remains the same
contrastive_loader_train = torch.utils.data.DataLoader(new_contrastive_dataset, batch_size=batch_size, shuffle=False)

# Testing contrastive loader
new_contrastive_dataset = Custom_Contrastive_Dataset(stacked_windows_test, torch.tensor(testing_indices), custom_transformation)

# # DataLoader remains the same
contrastive_loader_test = torch.utils.data.DataLoader(new_contrastive_dataset, batch_size=batch_size, shuffle=False)

# =============================================================================
#     # Let's do some local saving to save on computational time
# =============================================================================

aug_tensor_train = torch.empty((tau_in_steps, 0, 1, simple_tweetyclr.window_size, 151))

# first_aug_tensor = torch.empty((0, 1, simple_tweetyclr.window_size, 151))
# second_aug_tensor = torch.empty((0, 1, simple_tweetyclr.window_size, 151))
labels_tensor_train = torch.empty((0))

aug_dict_train = {}

# Iterate through a range of 15 keys
for i in range(tau_in_steps):
    value = []  # Initialize an empty list as the value for each key
    aug_dict_train[i] = value

# aug_list = []
labels_list_train = []
batch_sizes_train = []

for batch_idx, (data, labels) in enumerate(contrastive_loader_train):
    labels_list_train.append(labels)
    
    for i in np.arange(len(data)):
        aug = data[i]
        # aug_tensor[i,:,:,:] = aug
        aug_dict_train[i].append(aug)
    
labels_tensor_train = torch.cat(labels_list_train, dim=0)

flattened_dict_train = {key: [item for sublist in value for item in sublist] for key, value in aug_dict_train.items()}

# Initialize a list to store the dictionaries
dataloader_list_train = []
# filepath_list = []


for i in np.arange(len(flattened_dict_train)):
    
    aug_data = torch.cat(flattened_dict_train[i], dim = 0)
    dataset = TensorDataset(aug_data, labels_tensor_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    dataloader_list_train.append(dataloader)
    
#####
aug_tensor_test = torch.empty((tau_in_steps, 0, 1, simple_tweetyclr.window_size, 151))

# first_aug_tensor = torch.empty((0, 1, simple_tweetyclr.window_size, 151))
# second_aug_tensor = torch.empty((0, 1, simple_tweetyclr.window_size, 151))
labels_tensor_test = torch.empty((0))

aug_dict_test = {}

# Iterate through a range of 15 keys
for i in range(tau_in_steps):
    value = []  # Initialize an empty list as the value for each key
    aug_dict_test[i] = value

# aug_list = []
labels_list_test = []
batch_sizes_test = []

for batch_idx, (data, labels) in enumerate(contrastive_loader_test):
    labels_list_test.append(labels)
    
    for i in np.arange(len(data)):
        aug = data[i]
        # aug_tensor[i,:,:,:] = aug
        aug_dict_test[i].append(aug)
    
labels_tensor_test = torch.cat(labels_list_test, dim=0)

flattened_dict_test = {key: [item for sublist in value for item in sublist] for key, value in aug_dict_test.items()}

# Initialize a list to store the dictionaries
dataloader_list_test = []
# filepath_list = []


for i in np.arange(len(flattened_dict_test)):
    
    aug_data = torch.cat(flattened_dict_test[i], dim = 0)
    dataset = TensorDataset(aug_data, labels_tensor_test)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    dataloader_list_test.append(dataloader)    
    

hard_stacked_windows = simple_tweetyclr.stacked_windows[hard_indices,:]

hard_dataset = TensorDataset(torch.tensor(hard_stacked_windows.reshape(hard_stacked_windows.shape[0], 1, simple_tweetyclr.time_dim, simple_tweetyclr.freq_dim)))
hard_dataloader = DataLoader(hard_dataset, batch_size=batch_size , shuffle=False)



hard_stacked_windows = simple_tweetyclr.stacked_windows[hard_indices_train,:]
hard_dataset_train = TensorDataset(torch.tensor(hard_stacked_windows.reshape(hard_stacked_windows.shape[0], 1, simple_tweetyclr.time_dim, simple_tweetyclr.freq_dim)))
hard_dataloader_train = DataLoader(hard_dataset_train, batch_size=batch_size , shuffle=False)

hard_stacked_windows = simple_tweetyclr.stacked_windows[hard_indices_test,:]
hard_dataset_test = TensorDataset(torch.tensor(hard_stacked_windows.reshape(hard_stacked_windows.shape[0], 1, simple_tweetyclr.time_dim, simple_tweetyclr.freq_dim)))
hard_dataloader_test = DataLoader(hard_dataset_test, batch_size=batch_size , shuffle=False)

list_of_images_train = []
for batch_idx, (data) in enumerate(hard_dataloader_train):
    data = data[0]
    
    for image in data:
        list_of_images_train.append(image)
        
list_of_images_train = [tensor.numpy() for tensor in list_of_images_train]

embeddable_images_train = simple_tweetyclr.get_images(list_of_images_train)


list_of_images_test = []
for batch_idx, (data) in enumerate(hard_dataloader_test):
    data = data[0]
    
    for image in data:
        list_of_images_test.append(image)
        
list_of_images_test = [tensor.numpy() for tensor in list_of_images_test]

embeddable_images_test = simple_tweetyclr.get_images(list_of_images_test)


list_of_images_hard = []
for batch_idx, (data) in enumerate(hard_dataloader):
    data = data[0]
    
    for image in data:
        list_of_images_hard.append(image)
        
list_of_images_hard = [tensor.numpy() for tensor in list_of_images_hard]

embeddable_images_hard = simple_tweetyclr.get_images(list_of_images_hard)




# =============================================================================
#     # Pass data through untrained model and extract representation
# =============================================================================

# Untrained Model Representation
# Ensure the model is on the desired device
model = Encoder().to(torch.float32)
# Check if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    model = nn.DataParallel(model)

# Move your model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)  # Using weight decay with AdamW
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

# model = model.to('cpu')
# original_model = model.module
# original_model.to('cpu')
# model.eval()
# # model = model.to('cpu')
# # Initialize lists to store features and labels
# model_rep_list_untrained = []

criterion = SupConLoss(temperature=temp_value)
# # Iterate over the DataLoaders
# with torch.no_grad():  # Disable gradient computation for efficiency
#     # for data_loader in dataloader_list:
#     for batch_idx, (data) in enumerate(hard_dataloader_train):
#         data = data[0].to(torch.float32)
#         # features = original_model(data)
#         features = original_model(data)
#         model_rep_list_untrained.append(features)

# # Convert lists to tensors
# model_rep_untrained = torch.cat(model_rep_list_untrained, dim=0)

# #  UMAP on the untrained model
# reducer = umap.UMAP(metric = 'cosine')

# a = model_rep_untrained.clone().detach().numpy()
# embed = reducer.fit_transform(a)

# plt.figure()
# plt.suptitle("Training UMAP Representation Through the Untrained Model")
# plt.title(f'Number of Slices: {hard_indices_train.shape[0]}')
# plt.scatter(embed[:,0], embed[:,1], c = simple_tweetyclr.mean_colors_per_minispec[hard_indices_train,:])
# plt.xlabel("UMAP 1")
# plt.ylabel("UMAP 2")
# plt.savefig(f'{simple_tweetyclr.folder_name}/Plots/UMAP_of_untrained_model.png')
# plt.show()

# simple_tweetyclr.plot_UMAP_embedding(embed, simple_tweetyclr.mean_colors_per_minispec[hard_indices_train,:],embeddable_images_train, f'{simple_tweetyclr.folder_name}/Plots/UMAP_of_untrained_model.html', saveflag = True)

# torch.save(model.state_dict(), f'{simple_tweetyclr.folder_name}/untrained_model_state_dict.pth')

training_contrastive_loss, validation_contrastive_loss, contrastive_lr = [], [], []
model = model.to(device)
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()   
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08)
# from torch.optim.lr_scheduler import CosineAnnealingLR
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)  # Using weight decay with AdamW
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
# scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
model.train()

# THIS BELOW TRAINING LOOP NEEDS TO BE MODIFIED SO THAT I AM EVALUATING THE TRAINING AND VALIDATION PERFORMANCE. ALSO INCLUDE EARLY STOPPING. 

initial_negative_sims = []
initial_positive_sims = []

final_negative_sims = []
final_positive_sims = []

for epoch in range(0, num_epochs+1):
    criterion = SupConLoss(temperature=temp_value)
    # loss, lr, negative_similarities_for_epoch, features, mean_pos_cos_sim_for_epoch, mean_ntxent_positive_similarities_for_epoch = pretraining(epoch, model, zip(*dataloader_list), optimizer, criterion, method=method)
    training_loss, validation_loss, lr, training_features, validation_features, negative_sims, positive_sims = pretraining(epoch, model, zip(*dataloader_list_train), zip(*dataloader_list_test), optimizer, criterion, epoch, method=method)
    
    if epoch == 0:
        initial_negative_sims.append(negative_sims)
        initial_positive_sims.append(positive_sims)
    elif epoch == num_epochs:
        final_negative_sims.append(negative_sims)
        final_positive_sims.append(positive_sims)
        
    if (use_scheduler == True) & (epoch !=0):
        scheduler.step()
            
    training_contrastive_loss.append(training_loss)
    validation_contrastive_loss.append(validation_loss)
    contrastive_lr.append(lr)
    
    if epoch%10 == 0:
        torch.save(model.state_dict(), f'{simple_tweetyclr.folder_name}/_model_state_epoch_{epoch}_dict.pth')

# UNTRAINED 

plt.figure()
plt.hist([num * temp_value for num in initial_positive_sims[0]], alpha = 0.7, edgecolor = 'black', color = 'blue', label = 'Positive Sample Similarities')
plt.hist([num * temp_value for num in initial_negative_sims[0]], alpha = 0.7, edgecolor = 'black', color = 'red', label = 'Negative Sample Similarities')
plt.legend()
from scipy.stats import ks_2samp

ks, p = ks_2samp([num * temp_value for num in initial_positive_sims[0]], [num * temp_value for num in initial_negative_sims[0]])
print(ks)
plt.title("Representational Similarities of Untrained Model")
# plt.title(f'KS Value: {ks:.3f}')
plt.xlabel('Similarity Score')
plt.ylabel("Frequency")
plt.show()


# TRAINED

plt.figure()
plt.hist([num * temp_value for num in final_positive_sims[0]], alpha = 0.7, edgecolor = 'black', color = 'blue', label = 'Positive Sample Similarities')
plt.hist([num * temp_value for num in final_negative_sims[0]], alpha = 0.7, edgecolor = 'black', color = 'red', label = 'Negative Sample Similarities')
plt.legend()
from scipy.stats import ks_2samp

ks, p = ks_2samp([num * temp_value for num in final_positive_sims[0]], [num * temp_value for num in final_negative_sims[0]])
print(ks)
plt.title("Representational Similarities of Trained Model")
# plt.title(f'KS Value: {ks:.3f}')
plt.xlabel('Similarity Score')
plt.ylabel("Frequency")
plt.show()

# plt.show()
# model = model.to('cpu')
# original_model = model.module
# original_model.to('cpu')
original_model = model
original_model.eval()

# Initialize lists to store features and labels
model_rep_list_trained = []

# Iterate over the DataLoaders
with torch.no_grad():  # Disable gradient computation for efficiency
    # for data_loader in dataloader_list:
    for batch_idx, (data) in enumerate(hard_dataloader):
        data = data[0].to(torch.float32)
        features = original_model(data)
        model_rep_list_trained.append(features)

# Convert lists to tensors
model_rep_trained = torch.cat(model_rep_list_trained, dim=0)

reducer = umap.UMAP(metric='cosine')

# Compute the mean and standard deviation for each row
mean = model_rep_trained.mean(dim=1, keepdim=True)
std = model_rep_trained.std(dim=1, keepdim=True, unbiased=False)


trained_rep_umap = np.load('/Users/AnanyaKapoor/Downloads/trained_rep_umap_blue_purple_region.npy')
# trained_rep_umap = reducer.fit_transform(model_rep_trained.clone().detach().numpy())

plt.figure()
plt.title("Data UMAP Representation Through the Trained Model")
plt.scatter(trained_rep_umap[:,0], trained_rep_umap[:,1], c = simple_tweetyclr.mean_colors_per_minispec[hard_indices,:])
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.savefig(f'{simple_tweetyclr.folder_name}/UMAP_of_trained_model.png')
plt.show()


# # NOW FOR TESTING DATA

# # Initialize lists to store features and labels
# model_rep_list_trained_validation = []

# # Iterate over the DataLoaders
# with torch.no_grad():  # Disable gradient computation for efficiency
#     # for data_loader in dataloader_list:
#     for batch_idx, (data) in enumerate(hard_dataloader_test):
#         data = data[0].to(torch.float32)
#         features = original_model(data)
#         model_rep_list_trained_validation.append(features)

# # Convert lists to tensors
# model_rep_trained_validation = torch.cat(model_rep_list_trained_validation, dim=0)

# reducer = umap.UMAP(metric='cosine')

# trained_rep_umap_validation = reducer.fit_transform(model_rep_trained_validation.clone().detach().numpy())

# plt.figure()
# plt.title("Data UMAP Representation Through the Trained Model: Validation Data")
# plt.xlabel("UMAP 1")
# plt.ylabel("UMAP 2")
# plt.scatter(trained_rep_umap_validation[:,0], trained_rep_umap_validation[:,1], c = simple_tweetyclr.mean_colors_per_minispec[hard_indices_test,:])
# plt.savefig(f'{simple_tweetyclr.folder_name}/UMAP_of_trained_model_validation_data.png')
# plt.show()


plt.figure()
plt.title("Loss Curve")
plt.plot(training_contrastive_loss[1:], color = 'blue', label = 'training')
plt.plot(validation_contrastive_loss[1:], color = 'orange', label = 'validation')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(f'{simple_tweetyclr.folder_name}/loss_curves.png')
plt.show()

# simple_tweetyclr.plot_UMAP_embedding(trained_rep_umap,  simple_tweetyclr.mean_colors_per_minispec[hard_indices_train,:], embeddable_images_train, f'{simple_tweetyclr.folder_name}/Plots/UMAP_of_trained_model.html', saveflag = True)
# simple_tweetyclr.plot_UMAP_embedding(trained_rep_umap_validation,  simple_tweetyclr.mean_colors_per_minispec[hard_indices_test,:], embeddable_images_test, f'{simple_tweetyclr.folder_name}/Plots/UMAP_of_trained_model_testing.html', saveflag = True)

# =============================================================================
# # DOWNSTREAM EVALUATION
# =============================================================================

# HDBSCAN evaluation

X = trained_rep_umap.copy()

import hdbscan
clusterer = hdbscan.HDBSCAN(min_cluster_size = 100, gen_min_span_tree=True)
# Get the cluster labels

clusterer.fit(X)

# Get the cluster labels
labels = clusterer.labels_

# Number of clusters (excluding noise points)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

print("Number of clusters:", n_clusters)

plt.figure()
plt.scatter(X[:,0], X[:,1], c=clusterer.labels_, cmap='viridis', s=50)
plt.suptitle("HDBSCAN Clustering of Trained UMAP+TweetyCLR Representation of Training Data")
plt.title(f'Number of Clusters Detected: {n_clusters}')
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.savefig(f'{simple_tweetyclr.folder_name}/hdbscan_plot_trained_model_training_data.png')
plt.show()


# HDBSCAN On confused region raw UMAP 

X = simple_tweetyclr.umap_embed_init[hard_indices_train,:].copy()

import hdbscan
clusterer = hdbscan.HDBSCAN(min_cluster_size=100, gen_min_span_tree=True)
# Get the cluster labels

clusterer.fit(X)

# Get the cluster labels
labels = clusterer.labels_

# Number of clusters (excluding noise points)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

print("Number of clusters:", n_clusters)

plt.figure()
plt.scatter(X[:,0], X[:,1], c=clusterer.labels_, cmap='viridis', s=50)
plt.suptitle("HDBSCAN Clustering of UMAP Representation of Training Data")
plt.title(f'Number of Clusters Detected: {n_clusters}')
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.savefig(f'{simple_tweetyclr.folder_name}/hdbscan_plot_UMAP_training_data.png')
plt.show()


# Let's do K-Means + K-Fold CV with Silhouette score as the evaluation metric. This will allow me to determine how many clusers to use. 

# Number of splits for K-Fold
n_splits = 10

# Range of the number of clusters
cluster_range = range(2, 10)

simple_tweetyclr.downstream_clustering(X, n_splits, cluster_range)


# # Validation data
# X = trained_rep_umap_validation.copy()

# clusterer = hdbscan.HDBSCAN(min_cluster_size=100, gen_min_span_tree=True)
# # Get the cluster labels

# clusterer.fit(X)

# # Get the cluster labels
# labels = clusterer.labels_

# # Number of clusters (excluding noise points)
# n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

# print("Number of clusters:", n_clusters)

# plt.figure()
# plt.scatter(X[:,0], X[:,1], c=clusterer.labels_, cmap='viridis', s=50)
# plt.title("HDBSCAN Clustering of Trained UMAP+TweetyCLR Representation of Validation Data")
# plt.xlabel("UMAP 1")
# plt.ylabel("UMAP 2")
# plt.savefig(f'{simple_tweetyclr.folder_name}/hdbscan_plot_trained_model_validation_data.png')
# plt.show()

# # # Number of splits for K-Fold
# # n_splits = 5

# # # Range of the number of clusters
# # cluster_range = range(2, 10)

# # simple_tweetyclr.downstream_clustering(X, n_splits, cluster_range)
















