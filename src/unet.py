import torch
import torch.nn as nn
import torch.nn.functional as F

# Time Embedding class
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super(TimeEmbedding, self).__init__()
        self.linear1 = nn.Linear(1, dim)
        self.act = nn.SiLU()  # Swish activation
        self.linear2 = nn.Linear(dim, dim)
    
    def forward(self, t):
        # Ensure time is of shape [batch_size, 1]
        if len(t.shape) == 1:
            t = t.unsqueeze(-1)
        t = self.act(self.linear1(t))
        return self.linear2(t)

# UNet with Time Embeddings
class Unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512], time_dim=256):
        super(Unet, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.time_mlp = TimeEmbedding(time_dim)  # Time embedding module
        
        # Create the encoder part
        for channels in features:
            self.encoder.append(self._conv_block(in_channels, channels, time_dim))
            in_channels = channels
                    
        # Bottleneck layer
        self.bottleneck = self._conv_block(features[-1], features[-1] * 2, time_dim)
        
        # Create the decoder part
        for channels in reversed(features):
            self.decoder.append(nn.ConvTranspose2d(channels * 2, channels, kernel_size=2, stride=2))
            self.decoder.append(self._conv_block(channels * 2, channels, time_dim))
        
        # Final layer to output the desired number of channels
        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x, t):
        # Convert time t to float and pass through time embedding layers
        t = t.float()  # Ensure t is of type float
        t_embedding = self.time_mlp(t)
        
        # Encoder forward pass
        enc_outs = []
        for layer in self.encoder:
            x = layer(x, t_embedding)  # Pass the time embedding into each conv block
            enc_outs.append(x)  # Save the output of the encoder layers
            x = F.max_pool2d(x, kernel_size=2, stride=2)  # Downsample
        
        # Bottleneck forward pass
        x = self.bottleneck(x, t_embedding)
        
        # Decoder forward pass with skip connections
        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)  # Transpose Convolution
            
            # Get the corresponding encoder output
            enc_out = enc_outs[-(i // 2 + 1)]
            
            # Resize if dimensions don't match
            if x.shape[2:] != enc_out.shape[2:]:
                x = F.interpolate(x, size=enc_out.shape[2:])
            
            # Concatenate the encoder output with the decoder output
            x = torch.cat((x, enc_out), dim=1)
            x = self.decoder[i + 1](x, t_embedding)  # Pass time embedding into decoder conv block
        
        return self.final(x)

    
    def _conv_block(self, in_channels, out_channels, time_dim):
        return TimeConditionedConvBlock(in_channels, out_channels, time_dim)

# Time-conditioned Convolution Block
class TimeConditionedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super(TimeConditionedConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.act = nn.ReLU(inplace=True)
        
        # Linear layers to condition the conv block on time embeddings
        self.time_proj = nn.Sequential(
            nn.SiLU(),  # Swish activation
            nn.Linear(time_dim, out_channels)
        )
    
    def forward(self, x, t_embedding):
        # Pass the time embedding through the projection and reshape
        time_emb = self.time_proj(t_embedding).unsqueeze(-1).unsqueeze(-1)
        
        # Add time embedding to the feature maps
        x = self.conv1(x) + time_emb
        x = self.act(x)
        x = self.conv2(x) + time_emb
        return self.act(x)
