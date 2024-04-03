import os
import shutil
import numpy as np
from utils_Dataset_Creation import Tweetyclr
from model import TweetyCLR
from trainer import Trainer
from augmentations import White_Noise
from data_loaders import Contrastive_Loader, selecting_confused_region, create_dataloader
import umap
import torch

class Experiment:
    '''
    I essentially want to create a new folder that will store the results.
    '''
    def __init__(self, config) -> None:

        self.config_params = config

    def create_folder_name(self):
        '''
        I want to define the folder name. This should have the same name as the experiment name. Replace all spaces with an underscore.
        Also if there is a folder that has the same name, I want to move the existing folder and its contents to an archive folder location
        '''

        folder_name = self.config_params['experiment_title']

        # Now remove all spaces from the folder name and replace with underscores
        folder_name.replace(' ', '_')
        folder_name = os.path.join(os.getcwd(), folder_name)

        # Define the archived folder directory
        archived_directory_path = os.path.join(os.getcwd(), "archived")
        if os.path.exists(folder_name):
            # Create the "archived" directory if it doesn't exist
            os.makedirs(archived_directory_path, exist_ok=True)
            # Define the new path for the folder within the "archived" directory
            new_folder_path = os.path.join(archived_directory_path, folder_name)
            # Move the folder to the "archived" directory
            shutil.move(folder_name, new_folder_path)
            print(f"'{folder_name}' has been moved to the 'archived' directory.")
        else:
            print(f"The folder '{folder_name}' does not exist.")
            # Now create the folder corresponding to folder_name
            os.makedirs(folder_name)

        self.folder_name = folder_name

    def acquire_slices(self, tweetyclr_obj):
        '''
        This should probably go in the utils folder 
        '''
        stacked_specs_train, stacked_labels_train = tweetyclr_obj.extracting_data(self.config_params['train_directory'], self.config_params['num_spec_train'])
        stacked_specs_validation, stacked_labels_validation = tweetyclr_obj.extracting_data(self.config_params['validation_directory'], self.config_params['num_spec_validation'])

        stacked_windows_train = tweetyclr_obj.apply_windowing(stacked_specs_train.T, self.config_params['window_size'], self.config_params['stride'])
        stacked_windows_train = stacked_windows_train.reshape(stacked_windows_train.shape[0], -1)

        stacked_labels_train = tweetyclr_obj.apply_windowing(stacked_labels_train, self.config_params['window_size'], self.config_params['stride'])
        stacked_labels_train = stacked_labels_train.reshape(stacked_labels_train.shape[0], -1)

        stacked_windows_validation = tweetyclr_obj.apply_windowing(stacked_specs_validation.T, self.config_params['window_size'], self.config_params['stride'])
        stacked_windows_validation = stacked_windows_validation.reshape(stacked_windows_validation.shape[0], -1)

        stacked_labels_validation = tweetyclr_obj.apply_windowing(stacked_labels_validation, self.config_params['window_size'], self.config_params['stride'])
        stacked_labels_validation = stacked_labels_validation.reshape(stacked_labels_validation.shape[0], -1)

        return [stacked_windows_train, stacked_labels_train], [stacked_windows_validation, stacked_labels_validation]
       

    def compute_UMAP(self, arr, metric, seed):
        '''
        This should probably go in the utils folder
        '''
        reducer = umap.UMAP(metric = metric, random_state = seed)
        embed = reducer.fit_transform(arr)

        return embed

    def run(self):
        '''
        This function will essentially train and validate TweetyCLR + UMAP on the Canary Data. 
        In order to train the model I need to run the following: 
        1. Load in the data and do windowing (this should be two separate functions ideally)
        2. Initialize the model 
        3. Initialize the augmentation object
        4. Create a dataloader of triplets
        '''

        # Step 1: Load and process the data
        self.create_folder_name()
        wav_filepaths_train = [os.path.join(self.config_params["train_directory"], filename) for filename in os.listdir(self.config_params["train_directory"])]

        wav_filepaths_validation = [os.path.join(self.config_params["validation_directory"], filename) for filename in os.listdir(self.config_params["validation_directory"])]

        tweetyclr_obj = Tweetyclr(
            window_size = self.config_params['window_size'],
            stride = self.config_params['stride'],
            folder_name = self.folder_name,
            masking_freq_tuple=self.config_params['masking_freq_tuple'], 
            spec_dim_tuple = self.config_params['spec_dim_tuple'],
            category_colors=self.config_params['category_colors']
        )

        # Step 2: Initialize the model

        model = TweetyCLR(
            fc_dimensionality= self.config_params['fc_dimensionality'],
            dropout_perc= self.config_params['dropout_prop']
        )

        # Step 3: Initialize the augmentation object.
        wn = White_Noise(noise_param=0.5)

        # ALL OF STEP 4 SHOULD PROBABLY GO IN A DIFFERENT FILE. 

        # Step 4: Create the contrastive loader

        # Step 4.1: Acquire the training and validation slices
        train_data, validation_data = self.acquire_slices(tweetyclr_obj=tweetyclr_obj)

        # Step 4.2: Compute UMAP on the training and validation slices
        stacked_windows_train = train_data[0]
        stacked_labels_train = train_data[1]

        stacked_windows_validation = validation_data[0]
        stacked_labels_validation = validation_data[1]

        embed_train = self.compute_UMAP(arr = stacked_windows_train, metric = self.config_params['metric'], seed = self.config_params['random_seed'])
        embed_validation = self.compute_UMAP(stacked_windows_validation, metric = self.config_params['metric'], seed = self.config_params['random_seed'])

        # Let's calculate the mean colors
        # TODO: Eventually rewrite this later
            
        # Calculate mean colors
        mean_colors_per_minispec_train = np.zeros((stacked_labels_train.shape[0], 3))
        for i in range(stacked_labels_train.shape[0]):
            colors = np.array([tweetyclr_obj.category_colors[label] for label in stacked_labels_train[i, :]])
            mean_colors_per_minispec_train[i, :] = np.mean(colors, axis=0)

        # Calculate mean colors
        mean_colors_per_minispec_validation = np.zeros((stacked_labels_validation.shape[0], 3))
        for i in range(stacked_labels_validation.shape[0]):
            colors = np.array([tweetyclr_obj.category_colors[label] for label in stacked_labels_validation[i, :]])
            mean_colors_per_minispec_validation[i, :] = np.mean(colors, axis=0)

        # Step 4.3: Let's extract the hard indices for training and validation

        # From the training umap, I want to select ALL the slices in the confused region for training
        train_hard_indices_dict, list_of_train_hard_indices, xlim, ylim = selecting_confused_region(embed = embed_train, mean_colors_per_minispec= mean_colors_per_minispec_train, mode = 'train')

        # Now from the validation UMAP, I want to select ALL the slices in the confused region for validation

        validation_hard_indices_dict, list_of_validation_hard_indices, _, _ = selecting_confused_region(embed = embed_validation, mean_colors_per_minispec=mean_colors_per_minispec_validation, mode = 'train')

        # Step 4.4.1: Create a training dataloader 
        stacked_windows_train = stacked_windows_train.reshape(stacked_windows_train.shape[0], 1, 100, 151)
        total_dataset_train, total_dataloader_train = create_dataloader(stacked_windows_train, batch_size = self.config_params['batch_size'], label_values = stacked_labels_train, indices = np.arange(stacked_windows_train.shape[0]))
        hard_indices_train = train_hard_indices_dict[0]
        hard_dataset_train = stacked_windows_train[hard_indices_train,:].reshape(len(hard_indices_train), 1, self.config_params['window_size'], 151) # Dataset of all the confused spectrogram slices that we want to untangle
        hard_labels_train = stacked_labels_train[hard_indices_train,:].copy()
        hard_dataset_train, hard_dataloader_train = create_dataloader(hard_dataset_train, self.config_params['batch_size'], hard_labels_train, hard_indices_train, shuffle_status=False)

        # Step 4.4.2: Create a validation dataloader
        stacked_windows_validation = stacked_windows_validation.reshape(stacked_windows_validation.shape[0], 1, 100, 151)
        total_dataset_validation, total_dataloader_validation = create_dataloader(stacked_windows_validation, batch_size = self.config_params['batch_size'], label_values = stacked_labels_validation, indices = np.arange(stacked_windows_validation.shape[0]))
        hard_indices_validation = validation_hard_indices_dict[0]
        hard_dataset_validation = stacked_windows_validation[hard_indices_validation,:].reshape(len(hard_indices_validation), 1, self.config_params['window_size'], 151) # Dataset of all the confused spectrogram slices that we want to untangle
        hard_labels_validation = stacked_labels_validation[hard_indices_validation,:].copy()
        hard_dataset_validation, hard_dataloader_validation = create_dataloader(hard_dataset_validation, self.config_params['batch_size'], hard_labels_validation, hard_indices_validation, shuffle_status=False)

        # Step 4.4
        train_dataset = Contrastive_Loader(total_dataset_train, hard_dataset_train, hard_indices_train, embed_train, wn)   
        validation_dataset = Contrastive_Loader(total_dataset_validation, hard_dataset_validation, hard_indices_validation, embed_validation, wn)   

        shuffle_status = True
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = self.config_params['batch_size'], shuffle = shuffle_status)
        validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size = self.config_params['batch_size'], shuffle = shuffle_status)

        anc, pos, neg, idx = next(iter(train_loader))

        # Now I can finally call the trainer object 
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08)
        trainer = Trainer(
            model = model, 
            train_loader = train_loader, 
            validation_loader = validation_loader, 
            optimizer = optimizer, 
            device = self.config_params['device'], 
            max_steps = self.config_params['max_steps'],
            tau = self.config_params['tau'], 
            trailing_avg_window = 1000, 
            patience = 8
            )
        
        trainer.train()






       

















        





