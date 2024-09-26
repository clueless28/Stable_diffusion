import torch
from unet import Unet
import torch.nn as nn
from torch.utils.data import DataLoader
from diffussion import Diffusion
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 1e-4
epochs = 200
batch_size = 32
timesteps = 300
time_emb_dim = 256  # Dimension of time embedding

# Define the model and diffusion process
unet_model = Unet(in_channels=1, out_channels=1, features=[64, 128, 256, 512], time_dim=time_emb_dim).to(device)
diffusion_model = Diffusion(unet_model, timesteps=timesteps, device = device).to(device)
optimizer = optim.Adam(diffusion_model.parameters(), lr=lr,  weight_decay=1e-5)
mse_loss_fn = nn.MSELoss()
loss_fn_l1 = nn.L1Loss()

# Load dataset (e.g., CIFAR-10 for simplicity)
transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((64, 64))])
dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Function to visualize denoised images
def show_images(images, epoch, title=""):
    images = images.cpu().detach().numpy().transpose(0, 2, 3, 1)
    fig, axs = plt.subplots(1, len(images), figsize=(len(images) * 2, 2))
    for i, img in enumerate(images):
        axs[i].imshow(img.squeeze(), cmap="gray")
        axs[i].axis("off")
    plt.suptitle(title)
    plt.show()
    plt.savefig("/home/drovco/Bhumika/stable_diffusion/assets/" + str(epoch) + ".png" )
    
    
# Training loop
best_val_loss = float('inf')
for epoch in range(epochs):
    diffusion_model.train()
    for images, _ in dataloader:
        images = images.to(device)
        
        # Random time step for each batch
        t = torch.randint(0, timesteps, (batch_size,), device=device).long()
        
        # Forward diffusion (add noise)
        noisy_images, noise = diffusion_model.forward_diffusion(images, t, device)
        
        # Predict the noise using the model
        noise_pred = diffusion_model.reverse_process(noisy_images, t )
         # Compute loss
        loss_fn = mse_loss_fn(noise_pred, noise)
        loss_l1 = loss_fn_l1(noise_pred, noise)
        loss = loss_fn #+ 0.1 * loss_l1  # Weighted sum of MSE and L1 loss
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # Save model checkpoints occasionally
    #if (epoch + 1) % 10 == 0:
    if (loss < best_val_loss) or (epoch + 1) % 10 == 0:
     #   best_val_loss = loss
        torch.save(diffusion_model.state_dict(), "/home/drovco/Bhumika/stable_diffusion/assets/" + "unet_model_epoch_" + str(epoch+1) + ".pth")
        diffusion_model.eval()
        with torch.no_grad():
            sampled_images = diffusion_model.sample((batch_size, 1, 64, 64), t, device)
            show_images(sampled_images[:5], (epoch+1),  title=f"Sampled Images at Epoch {epoch+1}")