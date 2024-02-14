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

import numpy as np

class Tweetyclr:
    def __init__(self, num_spec, window_size, stride, folder_name, all_songs_data, masking_freq_tuple, spec_dim_tuple, exclude_transitions = False, category_colors = None):
        '''The init function should define:
            1. directory for bird
            2. directory for python files
            3. analysis path
            4. folder name 


            Additional tasks
            1. create the folder name if it does not exist already

        '''
        # self.bird_dir = bird_dir
        # self.directory = directory
        self.num_spec = num_spec
        self.window_size = window_size
        self.stride = stride
        # self.analysis_path = analysis_path
        self.category_colors = category_colors
        self.folder_name = folder_name
        self.all_songs_data = all_songs_data
        self.masking_freq_tuple = masking_freq_tuple
        self.freq_dim = spec_dim_tuple[1]
        self.time_dim = spec_dim_tuple[0]
        self.exclude_transitions = exclude_transitions

        # Create the folder if it doesn't already exist
        if not os.path.exists(folder_name+"/Plots/Window_Plots"):
            os.makedirs(folder_name+"/Plots/Window_Plots")
            print(f'Folder "{folder_name}" created successfully.')
        else:
            print(f'Folder "{folder_name}" already exists.')

    def first_time_analysis(self):

        # For each spectrogram we will extract
        # 1. Each timepoint's syllable label
        # 2. The spectrogram itself
        stacked_labels = [] 
        stacked_specs = []
        for i in np.arange(self.num_spec):
            # Extract the data within the numpy file. We will use this to create the spectrogram
            dat = np.load(self.all_songs_data[i])
            spec = dat['s']
            times = dat['t']
            frequencies = dat['f']
            labels = dat['labels']
            labels = labels.T


            # Let's get rid of higher order frequencies
            mask = (frequencies<self.masking_freq_tuple[1])&(frequencies>self.masking_freq_tuple[0])
            masked_frequencies = frequencies[mask]

            subsetted_spec = spec[mask.reshape(mask.shape[0],),:]
            
            stacked_labels.append(labels)
            stacked_specs.append(subsetted_spec)

            
        stacked_specs = np.concatenate((stacked_specs), axis = 1)
        stacked_labels = np.concatenate((stacked_labels), axis = 0)
        stacked_labels.shape = (stacked_labels.shape[0],1)


        # Get a list of unique categories (syllable labels)
        unique_categories = np.unique(stacked_labels)
        if self.category_colors == None:
            self.category_colors = {category: np.random.rand(3,) for category in unique_categories}
            self.category_colors[0] = np.zeros((3)) # SIlence should be black
            # open a file for writing in binary mode
            with open(f'{self.folder_name}/category_colors.pkl', 'wb') as f:
                # write the dictionary to the file using pickle.dump()
                pickle.dump(self.category_colors, f)

        spec_for_analysis = stacked_specs.T
        window_labels_arr = []
        embedding_arr = []
        # Find the exact sampling frequency (the time in miliseconds between one pixel [timepoint] and another pixel)
        print(times.shape)
        dx = np.diff(times)[0,0]

        # We will now extract each mini-spectrogram from the full spectrogram
        stacked_windows = []
        # Find the syllable labels for each mini-spectrogram
        stacked_labels_for_window = []
        # Find the mini-spectrograms onset and ending times 
        stacked_window_times = []

        # The below for-loop will find each mini-spectrogram (window) and populate the empty lists we defined above.
        for i in range(0, spec_for_analysis.shape[0] - self.window_size + 1, self.stride):
            # Find the window
            window = spec_for_analysis[i:i + self.window_size, :]
            # Get the window onset and ending times
            window_times = dx*np.arange(i, i + self.window_size)
            # We will flatten the window to be a 1D vector
            window = window.reshape(1, window.shape[0]*window.shape[1])
            # Extract the syllable labels for the window
            labels_for_window = stacked_labels[i:i+self.window_size, :]
            # Reshape the syllable labels for the window into a 1D array
            labels_for_window = labels_for_window.reshape(1, labels_for_window.shape[0]*labels_for_window.shape[1])
            # Populate the empty lists defined above
            stacked_windows.append(window)
            stacked_labels_for_window.append(labels_for_window)
            stacked_window_times.append(window_times)

        # Convert the populated lists into a stacked numpy array
        stacked_windows = np.stack(stacked_windows, axis = 0)
        stacked_windows = np.squeeze(stacked_windows)

        stacked_labels_for_window = np.stack(stacked_labels_for_window, axis = 0)
        stacked_labels_for_window = np.squeeze(stacked_labels_for_window)

        stacked_window_times = np.stack(stacked_window_times, axis = 0)
        # dict_of_spec_slices_with_slice_number = {i: stacked_windows[i, :] for i in range(stacked_windows.shape[0])}

        # For each mini-spectrogram, find the average color across all unique syllables
        mean_colors_per_minispec = np.zeros((stacked_labels_for_window.shape[0], 3))
        for i in np.arange(stacked_labels_for_window.shape[0]):
            list_of_colors_for_row = [self.category_colors[x] for x in stacked_labels_for_window[i,:]]
            all_colors_in_minispec = np.array(list_of_colors_for_row)
            mean_color = np.mean(all_colors_in_minispec, axis = 0)
            mean_colors_per_minispec[i,:] = mean_color

        self.stacked_windows = stacked_windows
        self.stacked_labels_for_window = stacked_labels_for_window
        self.mean_colors_per_minispec = mean_colors_per_minispec
        self.stacked_window_times = stacked_window_times
        self.masked_frequencies = masked_frequencies
        self.stacked_specs = stacked_specs
        self.stacked_labels = stacked_labels

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
