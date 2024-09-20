import torch
import torch.nn as nn
import torch.nn.functional as F


class Unet(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 3, features = [64, 128, 256, 512]):
        super(Unet, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
   
        for channels in features:
            self.encoder.append(self._conv_block(in_channels, channels))
            in_channels = channels
                    
        self.bottleneck = self._conv_block(features[-1], features[-1]*2)
        for channels in reversed(features):
            self.decoder.append(nn.ConvTranspose2d(channels*2, channels, kernel_size=2, stride=2))
            self.decoder.append(self._conv_block(channels*2, channels))
        
        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x, t=None):
        # Encoder part is unchanged
        enc_outs = []
        for layer in self.encoder:
            x = layer(x)
            enc_outs.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.bottleneck(x)
        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)
            enc_out = enc_outs[-(i//2 + 1)]
            if enc_out.dim() == 3:
                enc_out = enc_out.unsqueeze(2).unsqueeze(3)  # Add missing height and width dimensions
            if x.shape[2:] != enc_out.shape[2:]:
                x = F.interpolate(x, size=enc_out.shape[2:])
            x = torch.cat((x, enc_out), dim=1)
            x = self.decoder[i+1](x)
        return self.final(x)
            
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding =1 ),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                nn.ReLU(inplace=True)
                
        )
        
    
