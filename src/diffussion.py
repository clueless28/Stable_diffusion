import torch
import torch.nn as nn
import torch.optim as optim

class Diffusion(nn.Module):
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02, device = "cpu"):
        super(Diffusion, self).__init__()
        self.model = model
        self.timesteps = timesteps
        
        # Linearly increasing beta (noise schedule)
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        
        # Precompute alpha values
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0).to(device)

    def forward_diffusion(self, x0, t, device):
        """
        Adds noise to the original image `x0` at time step `t`.
        """
        noise = torch.randn_like(x0).to(device)
        alpha_bar_t = self.alpha_bar[t].to(device).reshape(-1, 1, 1, 1)
        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise, noise

    def reverse_process(self, xt, t, device):
        """
        Predicts the noise and denoises the image at time step `t`.
        """
        return self.model(xt, t)

    def sample(self, img_shape, device):
        """
        Samples an image by starting from pure noise and reversing the diffusion process.
        """
        xt = torch.randn(img_shape).to(device)
        for t in reversed(range(self.timesteps)):
            noise_pred = self.reverse_process(xt, t, device)     #here model is taking random noisy image as inputl, and it predicts now much noise is present in that image
            xt = self.p_sample(xt, noise_pred, t, device)
        return xt

    def p_sample(self, xt, noise_pred, t, device):
        """
        One step of reverse process.
        """
        beta_t = self.betas[t].reshape(-1, 1, 1, 1).to(device)
        alpha_bar_t = self.alpha_bar[t].reshape(-1, 1, 1, 1).to(device)
        
        # Predicted image
        x0_pred = (xt - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
        
        # Noise added based on beta
        noise = torch.randn_like(xt).to(device) if t > 0 else torch.zeros_like(xt)
        return torch.sqrt(alpha_bar_t) * x0_pred + torch.sqrt(beta_t) * noise

