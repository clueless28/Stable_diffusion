import torch
import torch.nn as nn
import torch.optim as optim  
import numpy as np
  
# Define the Diffusion model
class Diffusion(nn.Module):
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        super(Diffusion, self).__init__()
        self.model = model
        self.timesteps = timesteps

        # Linearly increasing beta (noise schedule)
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)

        # Precompute alpha values
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0).to(device)

    def forward_diffusion(self, x0, t, device):
        noise = torch.randn_like(x0).to(device)
        alpha_bar_t = self.alpha_bar[t].to(device).reshape(-1, 1, 1, 1)
        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise, noise
    
        noise = torch.randn_like(x0).to(device)
    
        # Ensure t is broadcasted correctly to match batch size
        if t.dim() == 0:
            t = t.unsqueeze(0).repeat(x0.shape[0])  # Repeat t for each sample in the batch
        
        # Now alpha_bar_t should match the batch size
        alpha_bar_t = self.alpha_bar[t].to(device).view(-1, 1, 1, 1)  # Match batch size
        
        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise, noise

    def reverse_process(self, xt, t):
        return self.model(xt, t)

    def sample(self, img_shape, t, device):
        xt = torch.randn(img_shape).to(device)
        
        # Create a time tensor for diffusion steps
        for t in reversed(range(self.timesteps)):
            t_tensor = torch.tensor([t], dtype=torch.float32, device=device)  # Convert t to a tensor
            noise_pred = self.reverse_process(xt, t_tensor)
            xt = self.p_sample(xt, noise_pred, t, device)
        
        return xt
    
    def p_sample(self, xt, noise_pred, t, device, dt=1e-2):
        beta_t = self.betas[t].reshape(-1, 1, 1, 1).to(device)
        alpha_bar_t = self.alpha_bar[t].reshape(-1, 1, 1, 1).to(device)

        # Predicted image using Euler-Maruyama method
        x0_pred = (xt - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)

        # Noise step for stochasticity in reverse process
        noise = torch.randn_like(xt).to(device) if t > 0 else torch.zeros_like(xt)

        # Convert dt to a tensor and move to the appropriate device
        dt_tensor = torch.tensor(dt, device=device)

        # Euler-Maruyama update: introduces stochasticity scaled by sqrt(dt)
        x_next = xt + dt_tensor * (x0_pred - xt) + torch.sqrt(dt_tensor) * torch.sqrt(beta_t) * noise

        return x_next


        