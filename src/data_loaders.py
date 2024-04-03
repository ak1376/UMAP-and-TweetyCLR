import torch 
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset

def create_dataloader(data_values, batch_size, label_values = None, indices = None, shuffle_status = True):
    '''
    Create dataloaders when we want to evaluate the model on the left out regions. This can also be used for creating images for Bokeh plots

    '''
    dataset = TensorDataset(torch.tensor(data_values), torch.tensor(label_values), torch.tensor(indices))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = shuffle_status)

    return dataset, data_loader



def selecting_confused_region(embed, mean_colors_per_minispec, mode='train', xlim=None, ylim=None, folder_name="output"):
    # Your existing setup
    hard_indices_dict = {}
    hard_region_coordinates = {}

    # Plot the initial scatter plot
    fig, ax = plt.subplots()
    sc = ax.scatter(embed[:, 0], embed[:, 1], s=10, c=mean_colors_per_minispec)
    plt.title('Zoom in on a region of interest (ROI), then press Enter')

    # Assuming `embed` and `mean_colors_per_minispec` are defined elsewhere in your code
    # and `mode` is a variable that defines the mode of operation

    if mode == 'train':
        xlim, ylim = None, None  # Initialize xlim and ylim

        # Define a callback function for training mode to capture xlim and ylim
        def on_press(event):
            nonlocal xlim, ylim
            if event.key == 'enter':
                # Get the current axes and limits
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()

                # Disconnect the event to prevent multiple captures
                plt.disconnect(cid)

                plt.close(fig)  # Close the figure to resume the script

                print("ROI selected. xlim and ylim captured.")

        # Connect the key press event to the callback function
        cid = plt.connect('key_press_event', on_press)
        
        plt.show(block=True)  # Show the plot and block execution until it's closed

    # Ensure that xlim and ylim are set
    if xlim and ylim:
        # Now you can safely use xlim and ylim for further computations
        hard_indices = np.where(
            (embed[:, 0] >= xlim[0]) & (embed[:, 0] <= xlim[1]) &
            (embed[:, 1] >= ylim[0]) & (embed[:, 1] <= ylim[1])
        )[0]
        # Further processing using hard_indices...
    else:
        print("xlim and ylim not set. Make sure to press Enter after zooming.")
    hard_indices_dict[0] = hard_indices
    hard_region_coordinates[0] = [xlim, ylim]

    list_of_hard_indices = []
    for i in np.arange(len(hard_indices_dict)):
        hard_ind = hard_indices_dict[i]
        list_of_hard_indices.append(hard_ind)

        plt.figure(figsize=(10, 10))
        plt.scatter(embed[hard_ind, 0], embed[hard_ind, 1], s=10, c=mean_colors_per_minispec[hard_ind, :])
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.suptitle(f'UMAP Representation of Hard Region #{i}')
        plt.title(f'Total Slices: {embed[hard_ind, :].shape[0]}')
        # plt.savefig(f'{folder_name}/Plots/UMAP_of_hard_slices_region_{i}.png')
        plt.show()

    return hard_indices_dict, list_of_hard_indices, xlim, ylim

class Contrastive_Loader(Dataset):
    def __init__(self, dataset, hard_dataset, total_hard_indices, umap_embedding, wn):
        ''' 
        The APP_MATCHER object should take in the entire dataset and the dataset of hard indices only. 
        The datasets should be Pytorch datasets. THe first component of each 
        dataset will be the actual flattened spectrogram slices. The second 
        component will be the index for each slice from the entire dataset. 
        This will be useful when selecting positive and negative samples.
        '''
        super(Contrastive_Loader, self).__init__()
        
        # Extracting all slices and indices
        all_features, all_labels, all_indices = zip(*[dataset[i] for i in range(len(dataset))]) # This will be used to create the all_possible_negatives region 
        
        hard_features, hard_labels, hard_indices = zip(*[hard_dataset[i] for i in range(len(hard_dataset))]) # This will be used to create all the hard features
        
        # Converting lists of tensors to a single tensor
        all_features, all_labels, all_indices = torch.stack(all_features), torch.stack(all_labels), torch.stack(all_indices)
        hard_features, hard_labels, hard_indices = torch.stack(hard_features), torch.stack(hard_labels), torch.stack(hard_indices)
                
        self.dataset = dataset
        self.hard_dataset = hard_dataset
        
        self.all_features = all_features
        self.all_indices = all_indices
        self.all_labels = all_labels


        self.hard_features = hard_features
        self.hard_indices = hard_indices
        self.hard_labels = hard_labels

        self.wn = wn

        self.total_hard_indices = total_hard_indices
        self.umap_embedding = torch.tensor(umap_embedding)
        
        # Find the set of "easy" negatives
        mask = ~torch.isin(self.all_indices, torch.tensor(self.total_hard_indices))
        all_possible_negatives = self.all_indices[mask]
        
        self.all_possible_negatives = all_possible_negatives

        self.unique_labels = np.unique(self.all_labels[self.all_possible_negatives,:]) # For selecting negatives in a supervised way
        
        # Find the positive images
        dict_of_indices = self.select_positive_image()
        self.dict_of_indices = dict_of_indices
        
    def select_positive_image(self):
        dict_of_indices = {}
        
        # Iterate over each data point
        for i in range(self.umap_embedding.size(0)):
            dict_of_indices[i] = i 
            
        return dict_of_indices                      
    
    def __len__(self):
        return self.hard_indices.shape[0]
    
    def __getitem__(self, index):
        
        ''' 
        The positive sample for each anchor image in the batch will be an augmented version of the anchor slice -- white noise
        The negative sample for each anchor image in the batch will be a randomly chosen spectrogram slice outside the hard region
        '''

        # The positive spectrogram slice will be the spectrogram slice that is closest in UMAP space to the anchor slice.
        actual_index = int(self.all_indices[int(self.hard_indices[index])])
        anchor_img = self.all_features[actual_index,:, :, :]
        
        # =============================================================================
        #         Positive Sample
        # =============================================================================
        
        positive_img = anchor_img.clone()

        positive_img = self.wn.forward(positive_img) 

        # =============================================================================
        #         Negative Sample
        # =============================================================================

        random_index = torch.randint(0, self.all_possible_negatives.size(0), (1,)).item()
        
        negative_index = self.all_possible_negatives[random_index].item()
        
        negative_img = self.all_features[negative_index, :, :, :]

        noisy_tensor = self.wn.forward(negative_img) 

        # # Clip values to be between 0 and 1
        negative_img = torch.clamp(noisy_tensor, 0, 1)
        
        return anchor_img, positive_img, negative_img, negative_index
