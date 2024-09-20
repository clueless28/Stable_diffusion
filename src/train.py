import torch
from unet import Unet
import torch.nn as nn
from torch.utils.data import DataLoader
from diffussion import Diffusion
import torch.optim as optim
from torchvision import datasets, transforms

# Hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 1e-4
epochs = 100
batch_size = 16
timesteps = 1000

# Define the model and diffusion process
unet_model = Unet().to(device)
diffusion_model = Diffusion(unet_model, timesteps=timesteps, device = device).to(device)
optimizer = optim.Adam(diffusion_model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

# Load dataset (e.g., CIFAR-10 for simplicity)
transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((64, 64))])
dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(epochs):
    diffusion_model.train()
    for images, _ in dataloader:
        images = images.to(device)
        
        # Random time step for each batch
        t = torch.randint(0, timesteps, (batch_size,), device=device).long()
        
        # Forward diffusion (add noise)
        noisy_images, noise = diffusion_model.forward_diffusion(images, t, device)
        
        # Predict the noise using the model
        noise_pred = diffusion_model.reverse_process(noisy_images, t, device)
        
        # Compute loss
        loss = loss_fn(noise_pred, noise)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # Save model checkpoints occasionally
    if (epoch + 1) % 10 == 0:
        torch.save(diffusion_model.state_dict(), f"diffusion_model_epoch_{epoch+1}.pth")
