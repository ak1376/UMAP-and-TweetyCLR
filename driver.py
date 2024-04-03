# I will supply the parameters necessary for this experiment

import numpy as np
import matplotlib.pyplot as plt
import os 
import sys
sys.path.append("src")

from experiment_manager import Experiment

# Matplotlib plotting settings
plt.rcParams.update({'font.size': 20})
plt.rcParams['figure.figsize'] = [15, 15]  # width and height should be in inches

config = {
    'experiment_title': "Canary",
    'train_directory': '/home/akapoor/Desktop/Canary_Training',
    'validation_directory': '/home/akapoor/Desktop/Canary_Testing',
    'random_seed': 295, 
    'num_spec_train': 100, 
    'num_spec_validation': 10,
    'masking_freq_tuple': (500, 7000),
    'spec_dim_tuple' : (100, 151),
    'window_size': 100, 
    'category_colors': None,
    'stride': 100, 
    'batch_size': 64, 
    'metric': 'cosine', 
    'fc_dimensionality': 256, 
    'dropout_prop': 0.1, 
    'max_steps': 10000, 
    'tau': 1.0, 
    'device': 'cuda'
}

# Now I want to initialize the experiment
# I want to pass the config file to the experiment manager. 

experiment_obj = Experiment(config)
experiment_obj.run()