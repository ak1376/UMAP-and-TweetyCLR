#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:40:41 2023

@author: akapoor
"""

import numpy as np
import torch
import sys
sys.path.append('/home/akapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_b_Canary_SSL/TweetyCLR/')
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

import numpy as np
import torch
import sys
sys.path.append('/home/akapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_b_Canary_SSL/TweetyCLR/')
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

import os
import matplotlib.pyplot as plt
import numpy as np
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import umap
import torch
from bokeh.plotting import figure, show, output_file, save
from bokeh.models import HoverTool, ColumnDataSource
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import base64
import io
from io import BytesIO

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
from matplotlib import cm
from PyQt5.QtCore import Qt
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F 
import sys
sys.path.append('/home/akapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_b_Canary_SSL/TweetyCLR/')
from util import MetricMonitor, SupConLoss
import torch.optim as optim
from util import Tweetyclr, TwoCropTransform, Custom_Contrastive_Dataset

# torch.multiprocessing.set_sharing_strategy('file_system')

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

def pretraining(epoch, model, contrastive_loader, optimizer, criterion, method='SimCLR'):
    "Contrastive pre-training over an epoch. Adapted from XX"
    negative_similarities_for_epoch = []
    ntxent_positive_similarities_for_epoch = []
    mean_pos_cos_sim_for_epoch = []
    mean_ntxent_positive_similarities_for_epoch = []
    metric_monitor = MetricMonitor()
    model.train()

    for batch_data in enumerate(contrastive_loader):
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

        loss, negative_similarities, positive_similarities = criterion(features)

        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("Learning Rate", optimizer.param_groups[0]['lr'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        negative_similarities_for_epoch.append(float(np.mean(negative_similarities.clone().detach().cpu().numpy())))

        ntxent_positive_similarities_for_epoch.append(positive_similarities)
        
        # # Calculate the mean cosine similarity of the model feature representation for the positive pairs.
        # # Slice the tensor to separate the two sets of features you want to compare
        
        split_features = torch.split(normalized_tensor, [bsz]*len(a), dim=0)
        split_features = [split.unsqueeze(1) for split in split_features]
        features = torch.cat(split_features, dim = 1)
        
    print("[Epoch: {epoch:03d}] Contrastive Pre-train | {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor))
    # return metric_monitor.metrics['Loss']['avg'], metric_monitor.metrics['Learning Rate']['avg'], negative_similarities_for_epoch, features, mean_pos_cos_sim_for_epoch, mean_ntxent_positive_similarities_for_epoch
    return metric_monitor.metrics['Loss']['avg'], metric_monitor.metrics['Learning Rate']['avg'], features


# =============================================================================
#     # Set data parameters
# =============================================================================
bird_dir = '/home/akapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_a_Rotations/Gardner_Lab/Canary_Data/llb16_data_matrices/'
# audio_files = bird_dir+'llb3_songs'
directory = bird_dir+ 'Python_Files'
analysis_path = '/home/akapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_b_Canary_SSL/TweetyCLR_Repo/'

# Parameters we set
num_spec = 100
window_size = 100
stride = 10

# Define the folder name
folder_name = f'{analysis_path}Num_Spectrograms_{num_spec}_Window_Size_{window_size}_Stride_{stride}'


files = os.listdir(directory)
all_songs_data = [f'{directory}/{element}' for element in files if '.npz' in element] # Get the file paths of each numpy file from Yarden's data
all_songs_data.sort()

masking_freq_tuple = (500, 7000)
spec_dim_tuple = (window_size, 151)


with open('/home/akapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_b_Canary_SSL/Canary_SSL_Repo/InfoNCE_Num_Spectrograms_100_Window_Size_100_Stride_10/category_colors.pkl', 'rb') as file:
    category_colors = pickle.load(file)
    
exclude_transitions = False


# =============================================================================
#     # Set model parameters
# =============================================================================

easy_negatives = 1
hard_negatives = 1
batch_size = easy_negatives + hard_negatives + 1
num_epochs = 100
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



# Let's do raw umap

# reducer = umap.UMAP(metric = 'cosine', random_state=295)
reducer = umap.UMAP(metric = 'cosine')

embed = reducer.fit_transform(simple_tweetyclr.stacked_windows)
simple_tweetyclr.umap_embed_init = embed

# embed = np.load('/home/akapoor/Desktop/embed.npy')

plt.figure()
plt.scatter(embed[:,0], embed[:,1], s = 10, c = simple_tweetyclr.mean_colors_per_minispec)
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


simple_tweetyclr.hard_indices = hard_indices

dataset = simple_tweetyclr.stacked_windows.copy()

# These will give the training and testing ANCHORS
stacked_windows_train, stacked_labels_train, mean_colors_per_minispec_train, training_indices, stacked_windows_test, stacked_labels_test, mean_colors_per_minispec_test, testing_indices = simple_tweetyclr.train_test_split(dataset, 1.0, hard_indices)

data_for_analysis = dataset.copy() 

total_dataset = TensorDataset(torch.tensor(data_for_analysis.reshape(data_for_analysis.shape[0], 1, simple_tweetyclr.time_dim, simple_tweetyclr.freq_dim)))
total_dataloader = DataLoader(total_dataset, batch_size=batch_size , shuffle=False)

# =============================================================================
# # Hand select negative samples
# =============================================================================

# Some negative samples need to be hard. I will do this by for each sample in 
# the bounding box, selecting spectrogram slices within the bounding box that 
# have a low cosine similarity. 

training_indices, stacked_windows_train = simple_tweetyclr.negative_sample_selection(data_for_analysis, training_indices)

dict_of_spec_slices_with_slice_number = {i: data_for_analysis[i, :] for i in range(data_for_analysis.shape[0])}

items = list(dict_of_spec_slices_with_slice_number.items())

# =============================================================================
#     # Create contrastive dataloaders
# =============================================================================
    
augmentation_object = Temporal_Augmentation(dict_of_spec_slices_with_slice_number , simple_tweetyclr, tau_in_steps=tau_in_steps)

custom_transformation = TwoCropTransform(augmentation_object)

# # Your new_contrastive_dataset initialization would be:
new_contrastive_dataset = Custom_Contrastive_Dataset(stacked_windows_train, torch.tensor(training_indices), custom_transformation)

# # DataLoader remains the same
contrastive_loader = torch.utils.data.DataLoader(new_contrastive_dataset, batch_size=batch_size, shuffle=False)

# =============================================================================
#     # Let's do some local saving to save on computational time
# =============================================================================

aug_tensor = torch.empty((tau_in_steps, 0, 1, simple_tweetyclr.window_size, 151))

# first_aug_tensor = torch.empty((0, 1, simple_tweetyclr.window_size, 151))
# second_aug_tensor = torch.empty((0, 1, simple_tweetyclr.window_size, 151))
labels_tensor = torch.empty((0))

aug_dict = {}

# Iterate through a range of 15 keys
for i in range(tau_in_steps):
    value = []  # Initialize an empty list as the value for each key
    aug_dict[i] = value

# aug_list = []
labels_list = []
batch_sizes = []

for batch_idx, (data, labels) in enumerate(contrastive_loader):
    labels_list.append(labels)
    
    for i in np.arange(len(data)):
        aug = data[i]
        # aug_tensor[i,:,:,:] = aug
        aug_dict[i].append(aug)
    
labels_tensor = torch.cat(labels_list, dim=0)

flattened_dict = {key: [item for sublist in value for item in sublist] for key, value in aug_dict.items()}

# Initialize a list to store the dictionaries
dataloader_list = []
# filepath_list = []



for i in np.arange(len(flattened_dict)):
    
    aug_data = torch.cat(flattened_dict[i], dim = 0)
    dataset = TensorDataset(aug_data, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    dataloader_list.append(dataloader)



hard_stacked_windows = simple_tweetyclr.stacked_windows[hard_indices,:]

hard_dataset = TensorDataset(torch.tensor(hard_stacked_windows.reshape(hard_stacked_windows.shape[0], 1, simple_tweetyclr.time_dim, simple_tweetyclr.freq_dim)))
hard_dataloader = DataLoader(hard_dataset, batch_size=batch_size , shuffle=False)

list_of_images = []
for batch_idx, (data) in enumerate(hard_dataloader):
    data = data[0]
    
    for image in data:
        list_of_images.append(image)
        
list_of_images = [tensor.numpy() for tensor in list_of_images]

embeddable_images = simple_tweetyclr.get_images(list_of_images)

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

model = model.to('cpu')
model.eval()

# Initialize lists to store features and labels
model_rep_list_untrained = []

# Iterate over the DataLoaders
with torch.no_grad():  # Disable gradient computation for efficiency
    # for data_loader in dataloader_list:
    for batch_idx, (data) in enumerate(hard_dataloader):
        data = data[0].to(torch.float32)
        features = model(data)
        model_rep_list_untrained.append(features)

# Convert lists to tensors
model_rep_untrained = torch.cat(model_rep_list_untrained, dim=0)

#  UMAP on the untrained model
reducer = umap.UMAP(metric = 'cosine')

a = model_rep_untrained.clone().detach().numpy()
embed = reducer.fit_transform(a)

plt.figure()
plt.title("Data UMAP Representation Through the Untrained Model")
plt.scatter(embed[:,0], embed[:,1], c = simple_tweetyclr.mean_colors_per_minispec[hard_indices,:])
plt.savefig(f'{simple_tweetyclr.folder_name}/Plots/UMAP_of_untrained_model.png')
plt.show()

simple_tweetyclr.plot_UMAP_embedding(embed, simple_tweetyclr.mean_colors_per_minispec[hard_indices,:],embeddable_images, f'{simple_tweetyclr.folder_name}/Plots/UMAP_of_untrained_model.html', saveflag = True)


dict_of_spec_slices_with_slice_number = {i: data_for_analysis[i, :] for i in range(data_for_analysis.shape[0])}

items = list(dict_of_spec_slices_with_slice_number.items())

# =============================================================================
#     # Create contrastive dataloaders
# =============================================================================
    
augmentation_object = Temporal_Augmentation(dict_of_spec_slices_with_slice_number , simple_tweetyclr, tau_in_steps=tau_in_steps)

custom_transformation = TwoCropTransform(augmentation_object)

# # Your new_contrastive_dataset initialization would be:
new_contrastive_dataset = Custom_Contrastive_Dataset(stacked_windows_train, torch.tensor(training_indices), custom_transformation)

# # DataLoader remains the same
contrastive_loader = torch.utils.data.DataLoader(new_contrastive_dataset, batch_size=batch_size, shuffle=False)

# a = next(iter(contrastive_loader))

# =============================================================================
#     # Let's do some local saving to save on computational time
# =============================================================================

aug_tensor = torch.empty((tau_in_steps, 0, 1, simple_tweetyclr.window_size, 151))

# first_aug_tensor = torch.empty((0, 1, simple_tweetyclr.window_size, 151))
# second_aug_tensor = torch.empty((0, 1, simple_tweetyclr.window_size, 151))
labels_tensor = torch.empty((0))

aug_dict = {}

# Iterate through a range of 15 keys
for i in range(tau_in_steps):
    value = []  # Initialize an empty list as the value for each key
    aug_dict[i] = value

# aug_list = []
labels_list = []
batch_sizes = []

for batch_idx, (data, labels) in enumerate(contrastive_loader):
    labels_list.append(labels)
    
    for i in np.arange(len(data)):
        aug = data[i]
        # aug_tensor[i,:,:,:] = aug
        aug_dict[i].append(aug)
    
labels_tensor = torch.cat(labels_list, dim=0)

flattened_dict = {key: [item for sublist in value for item in sublist] for key, value in aug_dict.items()}

# Initialize a list to store the dictionaries
dataloader_list = []
# filepath_list = []


for i in np.arange(len(flattened_dict)):
    
    aug_data = torch.cat(flattened_dict[i], dim = 0)
    dataset = TensorDataset(aug_data, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    dataloader_list.append(dataloader)


contrastive_loss, contrastive_lr = [], []
model = Encoder().to(torch.float32).to(device)
# if torch.cuda.is_available():
#     model = model.cuda()
    # criterion = criterion.cuda()   
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08)
# from torch.optim.lr_scheduler import CosineAnnealingLR
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)  # Using weight decay with AdamW
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
# scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
model = model.to(device)
model.train()

for epoch in range(1, num_epochs+1):
    criterion = SupConLoss(temperature=temp_value)
    # loss, lr, negative_similarities_for_epoch, features, mean_pos_cos_sim_for_epoch, mean_ntxent_positive_similarities_for_epoch = pretraining(epoch, model, zip(*dataloader_list), optimizer, criterion, method=method)
    loss, lr, features = pretraining(epoch, model, zip(*dataloader_list), optimizer, criterion, method=method)

    if use_scheduler:
        scheduler.step()
            
    contrastive_loss.append(loss)
    contrastive_lr.append(lr)
    
model = model.to('cpu')
model.eval()

# Initialize lists to store features and labels
model_rep_list_trained = []

# Iterate over the DataLoaders
with torch.no_grad():  # Disable gradient computation for efficiency
    # for data_loader in dataloader_list:
    for batch_idx, (data) in enumerate(hard_dataloader):
        data = data[0].to(torch.float32)
        features = model(data)
        model_rep_list_trained.append(features)

# Convert lists to tensors
model_rep_trained = torch.cat(model_rep_list_trained, dim=0)

reducer = umap.UMAP(metric='cosine')

# Compute the mean and standard deviation for each row
mean = model_rep_trained.mean(dim=1, keepdim=True)
std = model_rep_trained.std(dim=1, keepdim=True, unbiased=False)

trained_rep_umap = reducer.fit_transform(model_rep_trained.clone().detach().numpy())

plt.figure()
plt.title("Data UMAP Representation Through the Trained Model")
plt.scatter(trained_rep_umap[:,0], trained_rep_umap[:,1], c = simple_tweetyclr.mean_colors_per_minispec[hard_indices,:])
plt.savefig(f'{simple_tweetyclr.folder_name}/UMAP_of_trained_model.png')
plt.show()


plt.figure()
plt.plot(contrastive_loss)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

simple_tweetyclr.plot_UMAP_embedding(trained_rep_umap,  simple_tweetyclr.mean_colors_per_minispec[hard_indices,:], embeddable_images, f'{simple_tweetyclr.folder_name}/Plots/UMAP_of_trained_model.html', saveflag = True)

# =============================================================================
# # DOWNSTREAM EVALUATION
# =============================================================================

# Let's do K-Means + K-Fold CV with Silhouette score as the evaluation metric. This will allow me to determine how many clusters to use. 

X = trained_rep_umap.copy()


# Number of splits for K-Fold
n_splits = 5

# Range of the number of clusters
cluster_range = range(2, 10)

simple_tweetyclr.downstream_clustering(X, n_splits, cluster_range)


# Let's do supervised classification on non-transition slices only
X = model_rep_trained.clone().detach().cpu().numpy()

# Store indices of rows with 3 or more unique labels
transition_slice_indices = []

for i, row in enumerate(stacked_labels_train):
    if len(np.unique(row)) >= 3:
        transition_slice_indices.append(i)

transition_slices_indices = np.array(transition_slice_indices)

X = np.delete(X, transition_slice_indices, axis = 0)
# X = np.delete(trained_rep_umap, transition_slice_indices, axis = 0)
labels = np.delete(stacked_labels_train, transition_slice_indices, axis = 0)
mcpm = np.delete(simple_tweetyclr.mean_colors_per_minispec[hard_indices,:], transition_slice_indices, axis = 0)

# plt.figure();plt.scatter(X[:,0], X[:,1], c = mcpm)

# Function to find the first nonzero element in each row
def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr != 0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

# Extract first nonzero element from each row
first_nonzero_indices = first_nonzero(labels, axis=1)
first_nonzero_elements = [labels[i, idx] if idx >= 0 else None for i, idx in enumerate(first_nonzero_indices)]

first_nonzero_elements = [0 if x is None else x for x in first_nonzero_elements]

labels = np.array(first_nonzero_elements)

# # test case: delete everything that is not the top 2 categories
# to_delete = np.where((labels != 13) & (labels != 2))[0]

# X = np.delete(X, to_delete, axis = 0)
# labels = np.delete(labels, to_delete, axis = 0)

X = torch.tensor(X)
labels = torch.tensor(labels)

# Create a mapping from original labels to a range 0 to N-1
unique_labels = torch.unique(labels)
label_to_idx = {label.item(): idx for idx, label in enumerate(unique_labels)}

# Map labels to consecutive indices
mapped_labels = torch.tensor([label_to_idx[label.item()] for label in labels])

# Create DataLoader
dataset = TensorDataset(X, mapped_labels)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

class LinearClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)
    
    

# Example usage
input_size = 320  # Number of input features
num_classes = 4  # Number of classes (change as needed)

model = LinearClassifier(input_size, num_classes)
# Loss function and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# =============================================================================
# Calculate Accuracy
# =============================================================================


model.eval()  # Set the model to evaluation mode

correct = 0
total = 0
logit_scores = []
with torch.no_grad():  # No need to track gradients for validation
    for inputs, targets in train_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)  # Get the index of the max log-probability
        logit_scores.append(outputs.data)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

accuracy = correct / total
print(f'Accuracy of the model on the test set: {accuracy * 100:.2f}%')


logit_scores = torch.cat(logit_scores)


# =============================================================================
# Calculate Confusion Matrix
# =============================================================================


from sklearn.metrics import confusion_matrix

# Assuming 'model' is your trained PyTorch model and 'data_loader' is your DataLoader

model.eval()  # Set the model to evaluation mode
all_predictions = []
all_labels = []

with torch.no_grad():  # No need to track gradients
    for inputs, labs in train_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        # Collect the predictions and actual labels
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labs.cpu().numpy())

# Compute the confusion matrix
cm = confusion_matrix(all_labels, all_predictions)

import seaborn as sns

# Plotting the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.ylabel('Actual Labels')
plt.xlabel('Predicted Labels')

# Optional: If you want to display class labels on the axes
class_labels = [str(int(label.numpy())) for label in unique_labels]  # Replace with your class labels
plt.xticks(ticks=np.arange(len(class_labels)) + 0.5, labels=class_labels)
plt.yticks(ticks=np.arange(len(class_labels)) + 0.5, labels=class_labels, rotation=0)

plt.show()

# # =============================================================================
# # AUC Calculation
# # =============================================================================

# # Assuming 'outputs' is your model's raw output and 'true_labels' are your actual labels
# logits = logit_scores
# true_labels = mapped_labels  # Replace with your actual labels

# # Convert logits to probabilities
# probabilities = F.softmax(logits, dim=1).detach().numpy()

# from sklearn.metrics import roc_auc_score

# # Calculate OvA AUC
# auc_score = roc_auc_score(true_labels, probabilities, multi_class='ovr')

# print("One-vs-All AUC:", auc_score)



# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve, auc
# from sklearn.preprocessing import label_binarize

# # Assuming 'y_true' is your true labels and 'y_scores' are the probability scores from your model
# # 'y_true' should be a 1D array of labels, and 'y_scores' should be a 2D array where each column corresponds to a class
# n_classes = len(np.unique(mapped_labels))

# # Binarize the labels for OvR calculation
# y_true_binary = label_binarize(mapped_labels, classes=np.arange(n_classes))

# # Calculate ROC curve and ROC area for each class
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_true_binary[:, i], probabilities[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])

# # Plot all ROC curves
# plt.figure(figsize=(8, 6))
# colors = iter(plt.cm.rainbow(np.linspace(0, 1, n_classes)))
# for i, color in zip(range(n_classes), colors):
#     plt.plot(fpr[i], tpr[i], color=color, lw=2,
#              label='ROC curve of class {0} (area = {1:0.2f})'
#              ''.format(i, roc_auc[i]))

# plt.plot([0, 1], [0, 1], 'k--', lw=2)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Some extension of Receiver operating characteristic to multi-class')
# plt.legend(loc="lower right")
# plt.show()


def top_k_accuracy(y_true, y_pred, k=5):
    """
    Calculate top-k accuracy.
    
    Parameters:
    y_true (Tensor): True labels.
    y_pred (Tensor): Predictions from the model. Assumes that predictions are logits or probabilities.
    k (int): Top k predictions to consider.
    
    Returns:
    float: Top-k accuracy.
    """
    # Get the top k predictions for each sample
    top_k_preds = torch.topk(y_pred, k, dim=1).indices

    # Check if the true labels are in the top k predictions
    correct = top_k_preds.eq(y_true.view(-1, 1).expand_as(top_k_preds))

    # Calculate accuracy
    top_k_accuracy = correct.sum().float() / y_true.size(0)
    return top_k_accuracy.item()

# Example usage
# Assuming y_true are your true labels and y_pred are your model's predictions
y_true = torch.tensor([2, 3, 1, 0, 4])  # Example true labels
y_pred = torch.randn(5, 10)  # Example predictions (random in this case)

k_value = 2
accuracy = top_k_accuracy(mapped_labels, logit_scores, k=k_value)
print(f"Top-{k_value} accuracy: {accuracy * 100:.2f}%")


