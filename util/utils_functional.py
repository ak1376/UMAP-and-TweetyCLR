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
    def __init__(self, noise_level, num_augmentations):
        self.noise_level = noise_level
        self.num_augmentations = num_augmentations

    def white_noise(self, batch):

        # Define the noise scale (e.g., 5% of the data range)
        noise_scale = self.noise_level        
        # # Generate uniform noise and scale it
        noise = torch.rand_like(batch) * noise_scale
        
        # # Add the noise to the original tensor
        noisy_tensor = batch + noise
        
        # # Clip values to be between 0 and 1
        positive_img = torch.clamp(noisy_tensor, 0, 1)

        return positive_img



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
    







