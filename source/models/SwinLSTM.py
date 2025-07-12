import torch
import torch.nn as nn
import torch.nn.functional as F
from source.modules.SwinLSTM_modules import *     # STconvert
from scripts.registry import register_model
'''
20 Feb 2025 - Mert
https://github.com/SongTang-x/SwinLSTM/blob/main/SwinLSTM_B.py Original Source code
https://github.com/chengtan9907/OpenSTL/blob/OpenSTL-Lightning/openstl/models/swinlstm_model.py OpenSTL Spatiotemporal Benchmark
'''

@register_model("SwinLSTM")

class SwinLSTM_B(nn.Module):
    '''
    Args:
        inputMaps (Tensor): Input sequence of shape [B, C, T, H, W].
        truthMaps (Tensor): Ground truth frames of shape [B, (total_T - T), C, H, W]
        kwargs:
            return_loss (bool, optional): Whether to compute loss. Defaults to True.

    Returns:
        predictedFrames (Tensor): Predicted frames of shape [B, total_T - T, H, W].
        loss_allMaps (Tensor, optional): Computed loss, if return_loss=True.
    '''
    
    def __init__(self, configs):           
        super(SwinLSTM_B, self).__init__()
        self.configs = configs
        _, C, H, W = configs.input_shape
        assert H == W                          # Only support H = W for image input
        
        self.ST = STconvert(img_size=H, patch_size=configs.patch_size, in_chans=C,
                            embed_dim=configs.embed_dim, depths=configs.depths,
                            num_heads=configs.num_heads, window_size=configs.window_size)
        
        self.criterion = nn.MSELoss()
        
        # Add Conv layer to reduce channels from C to 1
        self.conv_out = nn.Conv3d(in_channels=C,                    # Channels
                                    out_channels=1,                 # Reduce to 1 channel
                                    kernel_size=(1, 1, 1),          # Keep spatial and temporal dimensions unchanged
                                    stride=1,
                                    padding=0)
        
    def forward(self, inputMaps, truthMaps, **kwargs):
        T, _, _, _ = self.configs.input_shape                       # T = Input sequence length
        total_T = self.configs.total_T                              # total_T = Input sequence length + Prediction horizon
        inputMaps = inputMaps.permute(0, 2, 1, 3, 4).contiguous()   # [B, C, T, H, W] -> [B, T, C, H, W]
        
        states = None                                               # Initialize hidden states
        nextFrames = []                                             # Initialize list to store predicted frames
        lastMap = inputMaps[:, -1]                                  # Last frame of the input sequence
        
        for i in range(T - 1):                                      # Predict the next T frames using the first T frames
            output, states = self.ST(inputMaps[:, i], states)       # [B, C, H, W]
            nextFrames.append(output)                               # Stack along dim=0 to get [T, B, C, H, W]
        for i in range(total_T - T):                                # Predict the next total_T - T frames using the predicted frames
            output, states = self.ST(lastMap, states)               # [B, C, H, W]
            nextFrames.append(output)                               # Stack along dim=0 to get [total_T, B, C, H, W]
            lastMap = output                                        # Update lastMap for the next iteration
            
        nextFrames = torch.stack(nextFrames, dim=0).permute(1, 2, 0, 3, 4).contiguous() # [T, B, C, H, W] -> [B, C, T, H, W]
        nextFrames = self.conv_out(nextFrames).permute(0, 2, 1, 3, 4).contiguous()      # [B, C, T, H, W] -> [B, T, C, H, W]
        
        allMaps = torch.cat((inputMaps[:,:,0,:,:], truthMaps), dim=1)  # [B, total_T, C, H, W] # Concatenate the input GIM (first channel) with the ground truth maps
        loss_allMaps = self.criterion(nextFrames.squeeze(2), allMaps[:, 1:])  # Calculate loss for all steps to asses the overall performance  
        predictedFrames = nextFrames[:, T - 1:].squeeze(2)  # [B, total_T-1, C, H, W] => [B, total_T - T, H, W]
        
        return predictedFrames, loss_allMaps