# This script will perform supervised contrastive learning using the following selection criteria for positive and negative samples: 
# 1. Randomly batch the data 
# 2. Compute the histogram of labels in each snippet in the batch
# 3. Take the pairwise distance of histograms for each snippet in the batch (this could be the pairwise distance between probability distributions)
# 4. Designate a threshold of distance score (KS). KS values that are below a certain threshold will be positive samples and values that are above a certain threshold will be negative samples. 


# Options for distance metrics: 
# 1. KS distance
# 2. Edit distance

# To make this as close to the disentangling concept as possible, I will do the following: 
# 1. For each hard sample, I will select easy negatives as samples from outside the confused region
# 2. For each hard sample, I will compute the KS distance between the anchor hard and all other hard samples.

# This means that the data will not be randomly batched but will be carefully curated to have a hard sample along with easy negatives and hard negatives

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
import scipy.stats
from torch.utils.data import Subset


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

# with open(f'{analysis_path}Num_Spectrograms_{num_spec}_Window_Size_{window_size}_Stride_{stride}/simple_tweetyclr.pkl', 'wb') as file:
#     pickle.dump(simple_tweetyclr, file)

with open(f'{analysis_path}Num_Spectrograms_{num_spec}_Window_Size_{window_size}_Stride_{stride}/simple_tweetyclr.pkl', 'rb') as file:
    simple_tweetyclr = pickle.load(file)


simple_tweetyclr.folder_name = folder_name


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

stacked_labels_for_window = simple_tweetyclr.stacked_labels_for_window.copy()

batch_size = 64

total_dataset, total_dataloader = create_dataloader(stacked_windows, batch_size, stacked_labels_for_window, np.arange(stacked_windows.shape[0]))

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
hard_labels = stacked_labels_for_window[hard_indices,:].copy()
hard_dataset, hard_dataloader = create_dataloader(hard_dataset, batch_size, hard_labels, hard_indices, shuffle_status=False)

# Need to create a train dataset and test dataset on the hard indices. 

# Split the dataset into a training and testing dataset
# Define the split sizes -- what is the train test split ? 

train_perc = 0.5 #
train_size = int(train_perc * len(hard_dataset))  # (100*train_perc)% for training
test_size = len(hard_dataset) - train_size  # 100 - (100*train_perc)% for testing

# ORGANIZE BELOW INTO A DATASET AND DATALOADER CREATION HELPER FUNCTION

from torch.utils.data import random_split

train_hard_dataset, test_hard_dataset = random_split(hard_dataset, [train_size, test_size]) 
train_hard_loader = torch.utils.data.DataLoader(train_hard_dataset, batch_size = batch_size, shuffle = True)
test_hard_loader = torch.utils.data.DataLoader(test_hard_dataset, batch_size = batch_size, shuffle = True)

# Getting the indices
train_indices = np.array(train_hard_dataset.indices)
test_indices = np.array(test_hard_dataset.indices)
embed_train = embed[hard_indices[train_indices],:]
embed_test = embed[hard_indices[test_indices], :]


# DEBUGGING

# hard_idx = 0 # Indexed over the train_hard_dataset

# unique_labels = np.unique(simple_tweetyclr.stacked_labels_for_window)
# num_unique_labels = unique_labels.shape[0]

# hist_counts, bin_edges = np.histogram(train_hard_dataset[hard_idx][1], bins=np.arange(-0.5, num_unique_labels, 1))

# # I want to stack all the train_hard_labels 
# train_hard_labels = []
# for i in np.arange(len(train_hard_dataset)):
#     train_hard_labels.append(train_hard_dataset[i][1])

# train_hard_labels = np.stack(train_hard_labels)


# hist_counts_list = []

# for i in np.arange(train_hard_labels.shape[0]):          
#     # Compute histogram for the i-th row
#     hist_counts, bin_edges = np.histogram(train_hard_labels[i,:], bins=np.arange(-0.5, num_unique_labels, 1))
#     prob_dist = hist_counts / hist_counts.sum()
#     hist_counts_list.append(hist_counts)

# hist_counts_arr = np.stack(hist_counts_list)

# # ks_list = []
# edit_dist_list = []
# for i in np.arange(train_hard_labels.shape[0]):
#     edit_dist = np.sum(train_hard_labels[hard_idx, :] != train_hard_labels[i, :])
#     # ks, _ = scipy.stats.ks_2samp(hist_counts_arr[hard_idx,:], hist_counts_arr[i,:])
#     # ks_list.append(ks)
#     edit_dist_list.append(edit_dist)

# edit_dist_arr = np.array(edit_dist_list)
# plt.figure()
# plt.hist(edit_dist_arr)
# plt.show()

# Thresholds for positives and negatives: 
# Hard negatives: [80, 100]
# Positives: (0, 40]

# positive_space = np.where((edit_dist_arr>0) & (edit_dist_arr<=50))[0] # Indices from the train_hard_labels.shape[0] that correspond to an edit distance > 0 and <= 40 
# # Let's choose a random index from the positive_space 
# rand_pos_idx = np.random.choice(positive_space)
# print(train_hard_labels[hard_idx, :])
# print(train_hard_labels[rand_pos_idx,:])

# # Hard negative space
# negative_space = np.where((edit_dist_arr>80) & (edit_dist_arr<=100))[0] # Indices from the train_hard_labels.shape[0] that correspond to an edit distance > 0 and <= 40 
# # Let's choose a random index from the positive_space 
# rand_neg_idx = np.random.choice(negative_space)
# print(train_hard_labels[hard_idx, :])
# print(train_hard_labels[rand_neg_idx,:])

# hard_negative_dataset_train = Subset(train_hard_dataset, negative_space)
# positive_dataset_train = Subset(train_hard_dataset, positive_space)


# I want a batch size of 64. I will have 10 positive samples and 54 negative samples. 
k_positive = 10
k_neg_easy = 26
k_neg_hard = 28


# Now let's create a data organizer class that will take an anchor sample and select a certain number of each negatives, a certain number of hard negatives, and a certain number of positive samples. 

class Creating_Our_Dataset(Dataset):
    def __init__(self, k_positive, k_neg_easy, k_neg_hard, hard_dataset, total_dataset, num_unique_labels = 31):
        '''
        I want to select k_positive samples from the hard dataset, k_neg_hard from the hard dataset and k_neg_easy from the total dataset
        '''

        self.k_positive = k_positive
        self.k_neg_easy = k_neg_easy
        self.k_neg_hard = k_neg_hard
        self.hard_dataset = hard_dataset
        self.total_dataset = total_dataset
        self.num_unique_labels = num_unique_labels

        self.all_features, self.all_labels, self.all_indices = zip(*[self.total_dataset[i] for i in range(len(self.total_dataset))]) # I will use this to extract the images using the subsequent negative indices
        self.hard_features, self.hard_labels, self.hard_indices = zip(*[self.hard_dataset[i] for i in range(len(self.hard_dataset))]) # This will be used to create all the hard features

        # Converting lists of tensors to a single tensor
        self.all_features, self.all_labels, self.all_indices = torch.stack(self.all_features), torch.stack(self.all_labels), torch.stack(self.all_indices)
        self.hard_features,self.hard_labels, self.hard_indices = torch.stack(self.hard_features), torch.stack(self.hard_labels), torch.stack(self.hard_indices)
        
    def __len__(self):
        return self.hard_indices.shape[0]
    
    def create_easy_negative_region(self):
            total_indices = np.arange(len(self.total_dataset))

            easy_negatives = np.setdiff1d(total_indices, self.hard_indices.numpy())
            easy_negatives = torch.tensor(easy_negatives)

            return easy_negatives
    
    def create_hard_negative_region(self, edit_dist_arr, lower_limit, upper_limit):
        # Hard negative space
        negative_space = np.where((edit_dist_arr>lower_limit) & (edit_dist_arr<=upper_limit))[0] # Indices from the train_hard_labels.shape[0] that correspond to an edit distance > 0 and <= 40 
        
        # Now return the indices in context of the full dataset
        hard_negative_indices = self.hard_indices[negative_space]
        return hard_negative_indices
    
    def create_positive_region(self, edit_dist_arr, lower_limit, upper_limit):
        positive_space = np.where((edit_dist_arr>lower_limit) & (edit_dist_arr<=upper_limit))[0] # Indices from the train_hard_labels.shape[0] that correspond to an edit distance > 0 and <= 40 
        # Now return the indices in context of the full dataset
        positive_indices = self.hard_indices[positive_space]

        return positive_indices

    def calculate_edit_distance(self, index):
        '''
        Calculates the edit distance between the item in the batch and the rest of the samples in the hard region 
        '''
        
        hist_counts_list = []

        for i in np.arange(self.hard_labels.shape[0]):          
            # Compute histogram for the i-th row
            hist_counts, bin_edges = np.histogram(self.hard_labels[i,:], bins=np.arange(-0.5, self.num_unique_labels, 1))
            hist_counts_list.append(hist_counts)

        hist_counts_arr = np.stack(hist_counts_list)

        # ks_list = []
        edit_dist_list = []
        for i in np.arange(self.hard_labels.shape[0]):
            edit_dist = np.sum(self.all_labels[index, :].numpy() != self.hard_labels[i, :].numpy()) # index is over the entire dataset, not good
            # ks, _ = scipy.stats.ks_2samp(hist_counts_arr[hard_idx,:], hist_counts_arr[i,:])
            # ks_list.append(ks)
            edit_dist_list.append(edit_dist)

        edit_dist_arr = np.array(edit_dist_list)

        return edit_dist_arr
    
    def __getitem__(self, index):
        '''
        For each hard index I want to randomly select k_neg_easy, k_neg_hard, and k_positive samples

        Parameters
        ----------
        index : int
            Batch index.

        Returns
        -------
        None.

        '''

        # Step #1: Extract the anchor image

        actual_index = int(self.all_indices[int(self.hard_indices[index])])
        anchor_img = self.all_features[actual_index,:, :, :].unsqueeze(1)


        # Step #2: Create the edit distance array
        edit_dist_arr = self.calculate_edit_distance(actual_index) # NEED TO CHECK INDICES

        # Step #2: Extract the space of easy negatives

        easy_negative_space = self.create_easy_negative_region()


        # Step #3: Extract the space of hard negatives
        hard_negative_space = self.create_hard_negative_region(edit_dist_arr, lower_limit=80, upper_limit = 100)


        # Step #4: Extract the space of positives
        positive_space = self.create_positive_region(edit_dist_arr, lower_limit = 0, upper_limit=50)

        # Step #5: Let's sample some indices that will be part of the batch

        ## Step 5.1 Sample easy negatives

        random_indices_easy_neg = torch.randint(0, easy_negative_space.size(0), (self.k_neg_easy,))
        easy_neg_imgs = self.all_features[easy_negative_space[random_indices_easy_neg], :, :, :]


        ## Step 5.2 Sample hard negatives
        random_indices_hard_neg = torch.randint(0, hard_negative_space.size(0), (self.k_neg_hard,))
        hard_neg_imgs = self.all_features[hard_negative_space[random_indices_hard_neg], :, :, :]

        ## Step 5.3 Sample positives
        random_indices_pos = torch.randint(0, positive_space.size(0), (self.k_positive,))
        pos_imgs = self.all_features[positive_space[random_indices_pos], :, :, :]

        return [pos_imgs, easy_neg_imgs, hard_neg_imgs], [positive_space[random_indices_pos], easy_negative_space[random_indices_easy_neg], hard_negative_space[random_indices_hard_neg]], anchor_img


example_class = Creating_Our_Dataset(k_positive, k_neg_easy, k_neg_hard, train_hard_dataset, total_dataset, num_unique_labels = 31)
a, b, anchor = next(iter(example_class))

# Next step: Create the InfoNCE loss function for the supervised case. 





