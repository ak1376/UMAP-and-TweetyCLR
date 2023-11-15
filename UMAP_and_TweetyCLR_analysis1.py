#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 18:03:05 2023

New idea after meeting with Tim -- Look at the UMAP embedding of llb16. Find 
the regions where a bunch of syllables are thrown in and then train SimCLR on 
those slices alone

@author: akapoor
"""
import numpy as np
import torch
import sys
sys.path.append('/home/akapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_b_Canary_SSL/TweetyCLR/')
from util import MetricMonitor, DataPlotter, SupConLoss
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
from util import MetricMonitor, DataPlotter, SupConLoss
import torch.optim as optim
from util import Tweetyclr, TwoCropTransform, Custom_Contrastive_Dataset

# class Encoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 8, 3,1,padding=1) 
#         self.conv2 = nn.Conv2d(8, 8, 3,2,padding=1) 
#         self.conv3 = nn.Conv2d(8, 16,3,1,padding=1) 
#         self.conv4 = nn.Conv2d(16,16,3,2,padding=1) 
#         self.conv5 = nn.Conv2d(16,24,3,1,padding=1) 
#         self.conv6 = nn.Conv2d(24,24,3,2,padding=1) 
#         self.conv7 = nn.Conv2d(24,32,3,1,padding=1) 
#         self.conv8 = nn.Conv2d(32,24,3,2,padding=1)
#         self.conv9 = nn.Conv2d(24,24,3,1,padding=1)
#         self.conv10 = nn.Conv2d(24,16,3,2,padding=1)
#         self.conv11 = nn.Conv2d(16, 8, 3, 1, padding = 1)
#         self.conv12 = nn.Conv2d(8, 8, 3, 2, padding = 1)
        
        
#         self.bn1 = nn.BatchNorm2d(1) 
#         self.bn2 = nn.BatchNorm2d(8) 
#         self.bn3 = nn.BatchNorm2d(8) 
#         self.bn4 = nn.BatchNorm2d(16) 
#         self.bn5 = nn.BatchNorm2d(16) 
#         self.bn6 = nn.BatchNorm2d(24) 
#         self.bn7 = nn.BatchNorm2d(24)
#         self.bn8 = nn.BatchNorm2d(32)
#         self.bn9 = nn.BatchNorm2d(24)
#         self.bn10 = nn.BatchNorm2d(24)
#         self.bn11 = nn.BatchNorm2d(16)
#         self.bn12 = nn.BatchNorm2d(8)

#         self.relu = nn.ReLU()       
#         self.dropout = nn.Dropout2d(
#         )
#         # self.fc = nn.Linear(320, 32)
#         # self._to_linear = 1280
#         # self._to_linear = 320
#         self._to_linear = 320
        
#     def forward(self, x):
         
#         # x = F.relu(self.dropout(self.conv1(self.bn1(x))))
#         # x = F.relu(self.conv2(self.bn2(x))) 
#         # x = F.relu(self.dropout(self.conv3(self.bn3(x))))
#         # x = F.relu(self.conv4(self.bn4(x))) 
#         # x = F.relu(self.dropout(self.conv5(self.bn5(x))))
#         # x = F.relu(self.conv6(self.bn6(x))) 
#         # x = F.relu(self.dropout(self.conv7(self.bn7(x))))
#         # x = F.relu(self.conv8(self.bn8(x)))
#         # x = F.relu(self.dropout(self.conv9(self.bn9(x))))
#         # x = F.relu(self.conv10(self.bn10(x)))
        
#         x = F.relu(self.conv1(self.bn1(x)))
#         x = F.relu(self.conv2(self.bn2(x))) 
#         x = F.relu(self.conv3(self.bn3(x)))
#         x = F.relu(self.conv4(self.bn4(x))) 
#         x = F.relu(self.conv5(self.bn5(x)))
#         x = F.relu(self.conv6(self.bn6(x))) 
#         x = F.relu(self.conv7(self.bn7(x)))
#         x = F.relu(self.conv8(self.bn8(x)))
#         x = F.relu(self.conv9(self.bn9(x)))
#         x = F.relu(self.conv10(self.bn10(x)))
#         x = F.relu(self.conv11(self.bn11(x)))
#         x = F.relu(self.conv12(self.bn12(x)))

#         x = x.view(-1, 48)
#         # x = x.view(-1, 320)
#         # x = self.fc(x)
#         # x = self.relu(x)
#         # x = x.view(-1, 32)
#         # x = x.view(-1, 1280) #Window size = 500
        
#         return x

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(8, 1), stride=(8, 1))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(8, 1), stride=(8, 1))
        self.relu = nn.ReLU()
        self.fc = nn.Linear(9408, 1000)
        
        
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layer
        # x = self.fc(x)
        return x 

def pretraining(epoch, model, contrastive_loader, optimizer, criterion, method='SimCLR'):
    "Contrastive pre-training over an epoch. Adapted from XX"
    negative_similarities_for_epoch = []
    ntxent_positive_similarities_for_epoch = []
    mean_pos_cos_sim_for_epoch = []
    mean_ntxent_positive_similarities_for_epoch = []
    metric_monitor = MetricMonitor()
    model.train()
    
    # dummy_dataloader = contrastive_loader[0] # THis will be used for the indices to enumerate
    
    # Use a list comprehension to concatenate data tensors within each tuple along dimension 0
    # data =  [torch.cat([data for data, _ in data_loader], dim=0) for data_loader in contrastive_loader]
    
    # # Concatenate all the data tensors from the list into one tensor
    # data = torch.cat(data, dim=0)


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
        
        
        # f1, f2 = torch.split(normalized_tensor, [bsz, bsz], dim=0)
        # normalized_features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        
        # tensor_a = normalized_features[:, 0, :].clone().detach()
        # tensor_b = normalized_features[:, 1, :].clone().detach()
        
        # # Compute the cosine similarities
        # similarities = F.cosine_similarity(tensor_a, tensor_b, dim=1)
        # mean_pos_cos_sim_for_batch = torch.mean(similarities).clone().detach().cpu().numpy()
        # mean_ntxent_positive_similarities = torch.mean(positive_similarities).clone().detach().cpu().numpy()
        # mean_ntxent_positive_similarities_for_epoch.append(float(mean_ntxent_positive_similarities))
        # mean_pos_cos_sim_for_epoch.append(float(mean_pos_cos_sim_for_batch))

    print("[Epoch: {epoch:03d}] Contrastive Pre-train | {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor))
    # return metric_monitor.metrics['Loss']['avg'], metric_monitor.metrics['Learning Rate']['avg'], negative_similarities_for_epoch, features, mean_pos_cos_sim_for_epoch, mean_ntxent_positive_similarities_for_epoch
    return metric_monitor.metrics['Loss']['avg'], metric_monitor.metrics['Learning Rate']['avg'], features








# =============================================================================
#     # Set data parameters
# =============================================================================
bird_dir = '/home/akapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_a_Rotations/Gardner_Lab/Canary_Data/llb16_data_matrices/'
# audio_files = bird_dir+'llb3_songs'
directory = bird_dir+ 'Python_Files'
analysis_path = '/home/akapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_b_Canary_SSL/Canary_SSL_Repo/'

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

batch_size = 64
num_epochs = 100
tau_in_steps = 5
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

# Let's do raw umap

# reducer = umap.UMAP(metric = 'cosine', random_state=295)
reducer = umap.UMAP(metric = 'cosine')

# embed = reducer.fit_transform(simple_tweetyclr.stacked_windows)

embed = np.load('/home/akapoor/Desktop/embed.npy')

plt.figure()
plt.scatter(embed[:,0], embed[:,1], s = 10, c = simple_tweetyclr.mean_colors_per_minispec)
plt.show()

# save the umap embedding for easy future loading
# np.save('/home/akapoor/Desktop/embed.npy', embed)


# I will define my TweetyCLR set as follows: xlim = (13, 18), ylim = (2, 8)
    
# hard_indices = np.where((embed[:,0]>=-3.35)&(embed[:,0]<=3.35) & (embed[:,1]>=11) & (embed[:,1]<=17))[0]

hard_indices = np.where((embed[:,0]>=7.5)&(embed[:,0]<=11.5) & (embed[:,1]>=5.23) & (embed[:,1]<=12))[0]

stacked_windows_train = simple_tweetyclr.stacked_windows[hard_indices,:].reshape(hard_indices.shape[0], 1, simple_tweetyclr.time_dim, simple_tweetyclr.freq_dim)


data_for_analysis = simple_tweetyclr.stacked_windows 
total_dataset = TensorDataset(torch.tensor(data_for_analysis.reshape(data_for_analysis.shape[0], 1, simple_tweetyclr.time_dim, simple_tweetyclr.freq_dim)))
total_dataloader = DataLoader(total_dataset, batch_size=batch_size , shuffle=False)

# =============================================================================
# # Hand select negative samples
# =============================================================================

# Some negative samples need to be hard. I will do this by for each sample in 
# the bounding box, selecting spectrogram slices within the bounding box that 
# have a low cosine similarity. 

from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(embed[hard_indices,:])

# Rewrite the loop as a list comprehension
n_smallest = 10  # Number of smallest elements to find for each row

# List comprehension to find the indices of the 10 smallest values for each row
smallest_indices_per_row = [np.concatenate((np.array([hard_indices[i]]),
    np.argpartition(cosine_sim[i,:], n_smallest)[:n_smallest][np.argsort(cosine_sim[i,:][np.argpartition(cosine_sim[i,:], n_smallest)[:n_smallest]])]))
    for i in np.arange(len(hard_indices))
]

total_indices = np.arange(data_for_analysis.shape[0])

easy_indices = np.setdiff1d(total_indices, hard_indices)

batch_indices_list = []
batch_array_list = []

all_sampled_indices = [np.random.choice(easy_indices, size=1, replace=False) for i in range(len(hard_indices))]

concatenated_indices = [
    np.concatenate([smallest_indices_per_row[i], all_sampled_indices[i]])
    for i in range(len(smallest_indices_per_row))
]


training_indices = np.stack(concatenated_indices)
training_indices = training_indices.reshape(training_indices.shape[0]*training_indices.shape[1])

stacked_windows_train = data_for_analysis[training_indices,:]

stacked_windows_train = torch.tensor(stacked_windows_train.reshape(stacked_windows_train.shape[0], 1, simple_tweetyclr.time_dim, simple_tweetyclr.freq_dim))


# from scipy.spatial import cKDTree

# # Create a cKDTree for efficient distance queries
# tree = cKDTree(embed)

# # Query the tree for the 64 farthest points from the given point
# # Since cKDTree.query returns the nearest points, we use a large k (e.g., size of dataset)
# # and then take the last 64 points from the results

# negative_samples_list = []

# for i in np.arange(hard_indices.shape[0]):
    
#     _, farthest_indices = tree.query(embed[hard_indices[i],:], k=len(embed))
    
#     # Since we want the farthest points, select the last 64 indices
#     farthest_indices = farthest_indices[-63:]
    
#     negative_samples = np.random.choice(farthest_indices, 63).tolist()
    
#     negative_samples_list.append(negative_samples)
    
    
# Concatenating each element in hard_indices with its respective list in negative_samples_list
# concatenated_arrays = [np.concatenate([[hard_index], negative_samples]) for hard_index, negative_samples in zip(hard_indices, negative_samples_list)]

# # Convert the list of arrays to a single 2D NumPy array
# result_array = np.array(concatenated_arrays)    # This is an array of indices that will be used for actual modeling. This is each hard index and its 63 negative samples.
    
dict_of_spec_slices_with_slice_number = {i: data_for_analysis[i, :] for i in range(data_for_analysis.shape[0])}

items = list(dict_of_spec_slices_with_slice_number.items())

# training_indices = result_array.reshape(result_array.shape[0]*result_array.shape[1])

# stacked_windows_train = simple_tweetyclr.stacked_windows[training_indices,:]






# =============================================================================
#     # Create contrastive dataloaders
# =============================================================================
    
augmentation_object = Temporal_Augmentation(dict_of_spec_slices_with_slice_number , simple_tweetyclr, tau_in_steps=tau_in_steps)

custom_transformation = TwoCropTransform(augmentation_object)

# # Your new_contrastive_dataset initialization would be:
new_contrastive_dataset = Custom_Contrastive_Dataset(stacked_windows_train, torch.tensor(training_indices), custom_transformation)

# # DataLoader remains the same
batch_size = 12
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
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)  # Using weight decay with AdamW
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
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

train_dataset = TensorDataset(torch.tensor(stacked_windows_train.reshape(stacked_windows_train.shape[0], 1, simple_tweetyclr.time_dim, simple_tweetyclr.freq_dim)))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size , shuffle=False)


hard_stacked_windows = simple_tweetyclr.stacked_windows[hard_indices,:]

hard_dataset = TensorDataset(torch.tensor(hard_stacked_windows.reshape(hard_stacked_windows.shape[0], 1, simple_tweetyclr.time_dim, simple_tweetyclr.freq_dim)))
hard_dataloader = DataLoader(hard_dataset, batch_size=batch_size , shuffle=False)


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

mean_colors_per_minispec_train = simple_tweetyclr.mean_colors_per_minispec[hard_indices,:]

plt.figure()
plt.title("Data UMAP Representation Through the Trained Model")
plt.scatter(trained_rep_umap[:,0], trained_rep_umap[:,1], c = mean_colors_per_minispec_train)
# plt.savefig(f'{simple_tweetyclr.folder_name}/UMAP_of_trained_model.png')
plt.show()


plt.figure()
plt.plot(contrastive_loss)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


def embeddable_image(data):
    data = (data.squeeze() * 255).astype(np.uint8)
    # convert to uint8
    data = np.uint8(data)
    image = Image.fromarray(data)
    image = image.convert('RGB')
    # show PIL image
    im_file = BytesIO()
    img_save = image.save(im_file, format='PNG')
    im_bytes = im_file.getvalue()

    img_str = "data:image/png;base64," + base64.b64encode(im_bytes).decode()
    return img_str


def get_images(list_of_images):
    return list(map(embeddable_image, list_of_images))


list_of_images = []
for batch_idx, (data) in enumerate(hard_dataloader):
    data = data[0]
    
    for image in data:
        list_of_images.append(image)
        
list_of_images = [tensor.numpy() for tensor in list_of_images]

embeddable_images = get_images(list_of_images)

def plot_UMAP_embedding(embedding, mean_colors_per_minispec, image_paths, filepath_name, saveflag = False):

    # Specify an HTML file to save the Bokeh image to.
    # output_file(filename=f'{self.folder_name}Plots/{filename_val}.html')
    output_file(filename = f'{filepath_name}')

    # Convert the UMAP embedding to a Pandas Dataframe
    spec_df = pd.DataFrame(embedding, columns=('x', 'y'))


    # Create a ColumnDataSource from the data. This contains the UMAP embedding components and the mean colors per mini-spectrogram
    source = ColumnDataSource(data=dict(x = embedding[:,0], y = embedding[:,1], colors=mean_colors_per_minispec))


    # Create a figure and add a scatter plot
    p = figure(width=800, height=600, tools=('pan, box_zoom, hover, reset'))
    p.scatter(x='x', y='y', size = 7, color = 'colors', source=source)

    hover = p.select(dict(type=HoverTool))
    hover.tooltips = """
        <div>
            <h3>@x, @y</h3>
            <div>
                <img
                    src="@image" height="100" alt="@image" width="100"
                    style="float: left; margin: 0px 15px 15px 0px;"
                    border="2"
                ></img>
            </div>
        </div>
    """

    p.add_tools(HoverTool(tooltips="""
    """))
    
    # Set the image path for each data point
    source.data['image'] = image_paths
    # source.data['image'] = []
    # for i in np.arange(spec_df.shape[0]):
    #     source.data['image'].append(f'{self.folder_name}/Plots/Window_Plots/Window_{i}.png')


    save(p)
    show(p)



plot_UMAP_embedding(trained_rep_umap,  mean_colors_per_minispec_train, embeddable_images, '/home/akapoor/Desktop/UMAP_of_trained_rep.html', saveflag = True)




