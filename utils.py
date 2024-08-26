import torch
import numpy as np
from torchvision.utils import save_image

# Function to save a batch of generated images
def save_generated_images(images, epoch, batch, directory="./results"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    save_image(images, f"{directory}/epoch_{epoch}_batch_{batch}.png", normalize=True)

# Function to create random noise for the generator
def create_noise(sample_size, nz, device):
    return torch.randn(sample_size, nz, device=device)

# Function to set up seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Normalize an image to the range [0, 1]
def denormalize(image):
    return (image * 0.5) + 0.5
