import torch
import torch.nn as nn
from unet import Unet
from diffussion import Diffusion


# Load your trained model
trained_model = Unet()  # Replace with your model class
trained_model.load_state_dict(torch.load("/home/product_master/Bhumika/stable_diffusion/src/diffusion_model_epoch_100.pth"))
trained_model.eval()
# Initialize the diffusion process
diffusion_model = Diffusion(trained_model, timesteps=1000, device='cuda' if torch.cuda.is_available() else 'cpu')

# Move the model to the correct device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
diffusion_model.to(device)


img_shape = (1, 3, 64, 64)

# Generate a sample image
sampled_image = diffusion_model.sample(img_shape, device)

# Move image to CPU and detach from computation graph for visualization
sampled_image = sampled_image.detach().cpu()

# If you want to visualize the generated image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# Un-normalize if you normalized your images during training
#unorm = transforms.Normalize(
 #  mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
  # std=[1/0.5, 1/0.5, 1/0.5]
#)

#sampled_image = unorm(sampled_image)
sampled_image = sampled_image.squeeze(0).permute(1, 2, 0)  # Reshape for visualization (H, W, C)

plt.imshow(sampled_image)
plt.axis('off')
plt.show()
from PIL import Image
import numpy as np

# Convert tensor to a valid image format
sampled_image_np = (sampled_image.numpy() * 255).astype(np.uint8)
image_pil = Image.fromarray(sampled_image_np)

# Save the image
image_pil.save("generated_sample.png")
