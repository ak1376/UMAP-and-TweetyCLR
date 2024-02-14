import torch
from torch.utils.data import DataLoader, TensorDataset
from bokeh.plotting import figure, show, output_file, save
from bokeh.models import HoverTool, ColumnDataSource
import pandas as pd
import numpy as np
import umap
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import base64
import io
from io import BytesIO

def create_dataloader(data_values, batch_size, label_values = None, shuffle_status = True):
    '''
    Create dataloaders when we want to evaluate the model on the left out regions. This can also be used for creating images for Bokeh plots

    '''

    dataset = TensorDataset(torch.tensor(data_values), torch.tensor(label_values))

    data_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = shuffle_status)

    return dataset, data_loader

def embeddable_image(data):
    data = (data.squeeze() * 255).astype(np.uint8)
    # convert to uint8
    data = np.uint8(data)
    image = Image.fromarray(data)
    image = image.rotate(90, expand=True) 
    image = image.convert('RGB')
    # show PIL image
    im_file = BytesIO()
    img_save = image.save(im_file, format='PNG')
    im_bytes = im_file.getvalue()

    img_str = "data:image/png;base64," + base64.b64encode(im_bytes).decode()
    return img_str


def get_images(list_of_images):
    return list(map(embeddable_image, list_of_images))

def embeddable_image(data):
    data = (data.squeeze() * 255).astype(np.uint8)
    # convert to uint8
    data = np.uint8(data)
    image = Image.fromarray(data)
    image = image.rotate(90, expand=True) 
    image = image.convert('RGB')
    # show PIL image
    im_file = BytesIO()
    img_save = image.save(im_file, format='PNG')
    im_bytes = im_file.getvalue()

    img_str = "data:image/png;base64," + base64.b64encode(im_bytes).decode()
    return img_str



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

def create_UMAP_plot(data_loader, simple_tweetyclr, indices_of_interest, model, name_of_file, saveflag = True):

    model_rep = []
    model = model.to('cpu')
    with torch.no_grad():
        for batch_idx, (img, idx) in enumerate(data_loader):
            data = img.to(torch.float32)
            
            output = model.module.forward_once(data)
            model_rep.append(output.numpy())

    model_rep_stacked = np.concatenate((model_rep))

    reducer = umap.UMAP(metric = 'cosine', random_state=295) # For consistency
    embed = reducer.fit_transform(model_rep_stacked)

    plt.figure()
    plt.scatter(embed[:,0], embed[:,1], c = simple_tweetyclr.mean_colors_per_minispec[indices_of_interest,:])
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.suptitle(f'UMAP Representation of Training Hard Region')
    plt.title(f'Total Slices: {embed.shape[0]}')
    plt.show()
    if saveflag == True:
        plt.savefig(f'{simple_tweetyclr.folder_name}/{name_of_file}.png')

    return model_rep
            
            
def creating_negatives_set(embed, hard_indices):

    total_indices = np.arange(embed.shape[0])

    easy_negatives = np.setdiff1d(total_indices, hard_indices)
    easy_negatives = torch.tensor(easy_negatives)

    hard_negatives = np.setdiff1d(total_indices, easy_negatives)
    hard_negatives = torch.tensor(hard_negatives)

    return easy_negatives, hard_negatives
    

class Augmentations:
    def __init__(self) -> None:
        pass

    def white_noise(self, batch, num_augmentations):
        '''
        The main features of this function should be that the augmentations are interleaved. 
        That is [batch_size, num_augmented_samples, 1, 100, 151]
        Where [0,0,1,100,151] is the first augmentation of the anchor sample. 
        Where [0,3,1,100,151] is the second augmented of the anchor sample 
        Where [0,1,1,100,151] is the first augmentation of the second sample in batch 0
        '''
        """
        Add white noise augmentations to a batch and interleave augmentations across samples.
        
        Parameters:
        - batch: input tensor of shape [batch_size, num_samples, 1, 100, 151].
        - num_augmentations: total number of augmentations to create for each sample, excluding the original.
        - noise_level: standard deviation of the Gaussian noise to add.
        
        Returns:
        - A tensor where each sample's augmentation is interleaved, 
        with shape [batch_size, num_samples * num_augmentations, 1, 100, 151].
        """
        batch_size, num_samples, channels, height, width = batch.shape
        
        # Generate noise for all augmentations across all samples
        # Shape: [batch_size, num_samples, num_augmentations, channels, height, width]
        noise = torch.randn(batch_size, num_samples, num_augmentations, channels, height, width) * noise_level
        noise = noise.to(batch.device)  # Ensure noise is on the same device as the batch

        # Apply noise to create augmentations
        # We expand the original batch to match the noise shape for addition
        original_expanded = batch.unsqueeze(2).expand(-1, -1, num_augmentations, -1, -1, -1)
        augmented_batch = original_expanded + noise
        
        # Reshape to move the augmentations dimension to the end for easier permutation
        # And then flatten the last three dimensions into one
        batch_noisy = augmented_batch.permute(0, 2, 1, 3, 4, 5).reshape(batch_size, num_augmentations * num_samples, channels, height, width)
            
        return batch_noisy

def infonce_loss_function(feats, temperature):

    batch_size = feats.shape[0]
    
    # Normalize the feature vectors (good practice for cosine similarity)
    feats_norm = F.normalize(feats, p=2, dim=2)
    
    
    # Reshape feats_norm for batch matrix multiplication: [128, 1, 6, 256] x [128, 256, 6, 1]
    # This treats each group of 6 samples as a separate 'batch' for the purpose of multiplication
    feats_reshaped = feats_norm.unsqueeze(1)  # Shape becomes [128, 1, 6, 256]
    transposed_feats = feats_norm.unsqueeze(-1)  # Shape becomes [128, 6, 256, 1]
    
    # Perform batch matrix multiplication
    # torch.matmul can handle broadcasting, so [128, 1, 6, 256] @ [128, 6, 256, 1] results in [128, 6, 6]
    cos_sim_matrices = torch.matmul(feats_reshaped, transposed_feats).squeeze()  # Squeeze to remove dimensions of size 1
    
    nll_grand_mean = 0
    pos_grand_mean = 0
    neg_grand_mean = 0
    for i in np.arange(batch_size):
        cos_sim = cos_sim_matrices[i,:,:]
        
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)

        # For monitoring: Compute mean positive and negative similarities
        pos_sim = cos_sim[pos_mask].mean()
        neg_sim = cos_sim[~pos_mask & ~self_mask].mean()
        
        # InfoNCE loss computation
        cos_sim = cos_sim / temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()
        
        
        nll_grand_mean+=nll
        pos_grand_mean+=pos_sim
        neg_grand_mean+=neg_sim
        
    
    nll_grand_mean/=batch_size
    pos_grand_mean/=batch_size
    neg_grand_mean/=batch_size
    
    return nll_grand_mean, pos_sim, neg_sim
    







