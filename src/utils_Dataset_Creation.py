#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 10:06:23 2023

@author: akapoor
"""

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
from tqdm import tqdm

import random

import numpy as np

class Tweetyclr:
    def __init__(self, window_size, stride, folder_name, masking_freq_tuple, spec_dim_tuple, category_colors = None):
        '''
        I want to rewrite this class so that I do the following improvements: 
        1. I want to load the filepaths for training and testing. These should be windowed separately
        2. 
        '''

        self.window_size = window_size
        self.stride = stride
        self.category_colors = category_colors
        self.folder_name = folder_name
        # self.wav_filepaths_train = wav_filepaths_train
        # self.wav_filepaths_validation = wav_filepaths_validation
        self.masking_freq_tuple = masking_freq_tuple
        self.freq_dim = spec_dim_tuple[1]
        self.time_dim = spec_dim_tuple[0]

    def extracting_data(self, songs_folderpath, num_spec):
        '''
        songs_folderpath will be the folder path pointing to the songs we are interested in processing.
        '''
        songs_of_interest = os.listdir(songs_folderpath)
        SOI_paths = [os.path.join(songs_folderpath, element) for element in songs_of_interest]

        # For each spectrogram we will extract
        # 1. Each timepoint's syllable label
        # 2. The spectrogram itself

        #  The songs_folderpath is a very long list -- 80% of the total songs are training and 20% are testing. However, we may want to select only a random subset of these songs for analysis. Therefore, we will need to randomly generate indices to index this list. 

        # indices = [random.randint(0, len(SOI_paths)+1) for _ in range(num_spec)]

        # all_songs_data = [SOI_paths[i] for i in indices]
        all_songs_data = SOI_paths.copy()

        stacked_labels = [] 
        stacked_specs = []
        for i in tqdm(np.arange(num_spec)):
            # Extract the data within the numpy file. We will use this to create the spectrogram
            dat = np.load(all_songs_data[i])
            spec = dat['s']
            times = dat['t']
            frequencies = dat['f']
            labels = dat['labels']
            labels = labels.T


            # Let's get rid of higher order frequencies
            mask = (frequencies<self.masking_freq_tuple[1])&(frequencies>self.masking_freq_tuple[0])
            subsetted_spec = spec[mask.reshape(mask.shape[0],),:]

            # Now let's z-score the spectrogram and take the log
            # Convert to dB and apply z-scoring
            # Sxx_log = 10 * np.log10(subsetted_spec)
            # Sxx_log = np.nan_to_num(Sxx_log, nan=0.0)

            mean = subsetted_spec.mean()
            std = subsetted_spec.std()
            Sxx_z_scored = (subsetted_spec - mean) / std
            Sxx_z_scored = np.nan_to_num(Sxx_z_scored, nan=0.0)

            stacked_labels.append(labels)
            stacked_specs.append(Sxx_z_scored)
            
        stacked_specs = np.concatenate((stacked_specs), axis = 1)
        stacked_labels = np.concatenate((stacked_labels), axis = 0)
        stacked_labels.shape = (stacked_labels.shape[0],1)

        # Get a list of unique categories (syllable labels)
        unique_categories = np.unique(stacked_labels)
        if self.category_colors == None:
            self.category_colors = {category: np.random.rand(3,) for category in unique_categories}
            self.category_colors[0] = np.zeros((3)) # SIlence should be black
            # open a file for writing in binary mode
            # with open(f'{self.folder_name}/category_colors.pkl', 'wb') as f:
            #     # write the dictionary to the file using pickle.dump()
            #     pickle.dump(self.category_colors, f)

        return stacked_specs, stacked_labels
    

    def apply_windowing(self, arr, window_size, stride, flatten_predictions=False):
        """
        Apply windowing to the input array.

        Parameters:
        - arr: The input array to window, expected shape (num_samples, features) for predictions and (num_samples,) for labels.
        - window_size: The size of each window.
        - stride: The stride between windows.
        - flatten_predictions: A boolean indicating whether to flatten the windowed predictions.

        Returns:
        - windowed_arr: The windowed version of the input array.
        """
        num_samples, features = arr.shape if len(arr.shape) > 1 else (arr.shape[0], 1)
        num_windows = (num_samples - window_size) // stride + 1
        windowed_arr = np.lib.stride_tricks.as_strided(
            arr,
            shape=(num_windows, window_size, features),
            strides=(arr.strides[0] * stride, arr.strides[0], arr.strides[-1]),
            writeable=False
        )

        if flatten_predictions and features > 1:
            # Flatten each window for predictions
            windowed_arr = windowed_arr.reshape(num_windows, -1)
        
        return windowed_arr

    def selecting_confused_region(self, embed):

        hard_indices_dict = {}
        hard_region_coordinates = {}

        # Plot the initial scatter plot
        fig, ax = plt.subplots()
        sc = ax.scatter(embed[:,0], embed[:,1], s = 10, c = self.mean_colors_per_minispec)
        plt.title('Zoom in on a region of interest (ROI), then press Enter')

        # Define a callback function that will be called when a key is pressed
        def on_press(event):
            if event.key == 'enter':
                # Get the current axes and limits
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()

                # Find the points within the new limits
                hard_indices = np.where(
                    (embed[:, 0] >= xlim[0]) & (embed[:, 0] <= xlim[1]) &
                    (embed[:, 1] >= ylim[0]) & (embed[:, 1] <= ylim[1])
                )[0]

                # Update the dictionaries with the new hard indices and coordinates
                hard_indices_dict[0] = hard_indices
                hard_region_coordinates[0] = [xlim, ylim]

                # Disconnect the event to prevent multiple captures if Enter is pressed again
                plt.disconnect(cid)

                print("ROI selected. Hard indices captured.")
                # If needed, re-plot here with the zoomed-in area or perform other actions

        # Connect the key press event to the callback function
        cid = plt.connect('key_press_event', on_press)

        # Show the plot with the event connection
        plt.show()

        list_of_hard_indices = []
        for i in np.arange(len(hard_indices_dict)):
            hard_ind = hard_indices_dict[i]
            list_of_hard_indices.append(hard_ind)

            plt.figure(figsize = (10,10))
            plt.scatter(embed[hard_ind,0], embed[hard_ind,1], s = 10, c = self.mean_colors_per_minispec[hard_ind,:])
            plt.xlabel("UMAP 1")
            plt.ylabel("UMAP 2")
            # plt.title("UMAP Decomposition of ")
            plt.suptitle(f'UMAP Representation of Hard Region #{i}')
            plt.title(f'Total Slices: {embed[hard_ind,:].shape[0]}')
            plt.savefig(f'{self.folder_name}/Plots/UMAP_of_hard_slices_region_{i}.png')
            plt.show()

        return hard_indices_dict, list_of_hard_indices
    
class Curating_Dataset:
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
        
        x = torch.cat((anchor_img, negative_imgs), dim = 0)
        
        return x, negative_indices
