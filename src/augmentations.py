import torch

class White_Noise:
    def __init__(self, noise_param = 0.5) -> None:
        '''
        Random Gaussian noise will be applied to each pixel in each tensor in the batch
        This class should take in a noise parameter (default is 0.5)
        '''
        self.noise_param = noise_param

    def forward(self, spec_tensors):

        # Generate Gaussian noise and scale it
        noise = torch.randn_like(spec_tensors) * self.noise_param

        # Add the noise to the original tensor
        noisy_tensor = spec_tensors + noise

        # Clip values to be between 0 and 1
        noisy_tensor_clipped = torch.clamp(noisy_tensor, 0, 1)

        return noisy_tensor_clipped
    

    