

import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from source.myTrainFuns import seedME
from scripts.registry import register_model
''' DNN221 Model from Boulch et al. (2018) 
Code adapted from https://github.com/aboulch/tec_prediction
This is the raw model.
Training does not use residual learning mentioned in the paper. (TO DO)
Last Update: 2 March 2025
'''
seedME(3)  # Seed everything for reproducibility
@register_model("DCNN121")
# Wrapper class for the DNN211 model to adapt IonoBench's input/output format
class DCNN121(nn.Module):
    def __init__(self, configs):
        super(DCNN121, self).__init__()
        self.configs = configs
        T_in, C, H, W = configs.input_shape
        self.base_model = UnetConvRecurrent(input_nbr=C, num_features=configs.num_features)  # Base model
        self.seq_input_length = configs.seq_len  # Input sequence length
        self.criterion = nn.MSELoss()  # Loss function
        self.conv_out = nn.Conv3d(in_channels=C,out_channels=1,kernel_size=(1, 1, 1),stride=1,padding=0)    # Extra conv3d to reduce the channel dimension (e.g. 18) to 1
        
    def forward(self, inputMaps, truthMaps):
        T, _, _, _ = self.configs.input_shape                       # T = Input sequence length
        total_T = self.configs.total_T                              # total_T = Input sequence length + Prediction horizon
        inputMaps = inputMaps.permute(0, 2, 1, 3, 4).contiguous()   # [B, C, T, H, W] -> [B, T, C, H, W]

        nextFrames = self.base_model.forward(inputMaps, self.configs.pred_horz, diff=False, predict_diff_data=None)
        nextFrames = nextFrames.permute(0, 2, 1, 3, 4).contiguous()      # [B, T, C, H, W] -> [B, C, T, H, W]  # Permute for Convolutions layer 
        nextFrames = self.conv_out(nextFrames).permute(0, 2, 1, 3, 4).contiguous()      # [B, C, T, H, W] -> [B, T, C, H, W]
        
        allMaps = torch.cat((inputMaps[:,:,0,:,:], truthMaps), dim=1)  # [B, total_T, C, H, W] # Concatenate the input GIM (first channel) with the ground truth maps
        loss_allMaps = self.criterion(nextFrames.squeeze(2), allMaps[:, 1:])  # Calculate loss for all steps to asses the overall performance
        predictedFrames = nextFrames[:, T - 1:].squeeze(2)  # [B, total_T-1, C, H, W] => [B, T_out, H, W]
        
        return predictedFrames, loss_allMaps


class UnetConvRecurrent(nn.Module):
    # Slight modification to output every steps prediction instead of just the `prediction_len` to calculate the loss for each step
    def __init__(self, input_nbr, num_features=8):
        super(UnetConvRecurrent, self).__init__()
        kernel_size = 3
        self.convRecurrentCell1 = CLSTM_cell(input_nbr, num_features, kernel_size, dilation=1, padding=1)
        self.convRecurrentCell2 = CLSTM_cell(num_features, num_features, kernel_size, dilation=2, padding=2)
        self.convRecurrentCell3 = CLSTM_cell(num_features, input_nbr, kernel_size, dilation=1, padding=1)

    def forward(self, z, prediction_len, diff=False, predict_diff_data=None):
        # z is assumed to be in sequence-first order: [B, T_in, C, H, W]  
        z = z.permute(1, 0, 2, 3, 4).contiguous()  # [T_in, B, C, H, W]
        output_inner = []
        seq_len = z.size(0)

        # Initialize hidden states for each recurrent cell
        hidden_state1, hidden_state2, hidden_state3 = None, None, None
        # --- Stage 1: Process the input sequence (all T_in frames) ---
        # Instead of only saving the last output, save all intermediate outputs.
        for t in range(seq_len):
            x = z[t]  # [B, C, H, W]
            hidden_state1 = self.convRecurrentCell1(x, hidden_state1)
            x1 = F.relu(hidden_state1[0])
            
            hidden_state2 = self.convRecurrentCell2(x1, hidden_state2)
            x2 = F.relu(hidden_state2[0])
            
            hidden_state3 = self.convRecurrentCell3(x2, hidden_state3)
            x3 = hidden_state3[0]
            y = x3  # output at time t
            
            output_inner.append(y)  # Save every output             (Modification)

        # --- Stage 2: Autoregressive prediction loop ---
        # Use the last output as a starting point and generate remaining frames.
        x = z[:, -1]  # Last frame of the input sequence
        for t in range(prediction_len - 1):
            if diff:
                x = y + predict_diff_data[t]
            else:
                x = y
                
            hidden_state1 = self.convRecurrentCell1(x, hidden_state1)
            x1 = F.relu(hidden_state1[0])
            
            hidden_state2 = self.convRecurrentCell2(x1, hidden_state2)
            x2 = F.relu(hidden_state2[0])
            
            hidden_state3 = self.convRecurrentCell3(x2, hidden_state3)
            x3 = hidden_state3[0]
            y = x3

            output_inner.append(y)
            x = y # Use the predicted frame as input for the next iteration

        # For comprehensive loss, you might want to compare against all frames. Thus modification makes sure all frames are returned.
        current_output = torch.stack(output_inner, dim=0)         # Stack along new time dimension.         # Expected shape: [T_total, B, C, H, W]
        return current_output.permute(1,0,2,3,4).contiguous()  # [T_total, B, C, H, W] => [B, T_total, C, H, W]
    
    

class CLSTM_cell(nn.Module):

    def __init__(self, input_size, hidden_size, kernel_size,dilation=1, padding=None):
        """Init."""
        super(CLSTM_cell, self).__init__()
        if padding is None:
            padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(self.input_size + self.hidden_size, 4*self.hidden_size, self.kernel_size, 1, padding=padding, dilation=dilation)

    def forward(self, input, prev_state=None):
        """Forward."""
        batch_size = input.data.size()[0]
        spatial_size = input.data.size()[2:]
        # Original:
        # if prev_state is None:
        #     state_size = [batch_size, self.hidden_size] + list(spatial_size)
        #     if(next(self.conv.parameters()).is_cuda):
        #         prev_state = [Variable(torch.zeros(state_size)).cuda(), Variable(torch.zeros(state_size)).cuda()]
        #     else:
        #         prev_state = [Variable(torch.zeros(state_size)), Variable(torch.zeros(state_size)).cuda()]
        if prev_state is None: # Little modification to make it device-agnostic
            # Get the device from the input tensor. This makes the cell device-agnostic.
            device = input.device 
            
            # Define the shape of the states
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            
            # Create the zero-filled tensors directly on the correct device
            h_0 = torch.zeros(state_size, device=device)
            c_0 = torch.zeros(state_size, device=device)

            prev_state = (Variable(h_0), Variable(c_0))
        hidden, c = prev_state  # hidden and c are images with several channels
        combined = torch.cat((input, hidden), 1)  # concatenate in the channels
        A = self.conv(combined)
        (ai, af, ao, ag) = torch.split(A, self.hidden_size, dim=1)  # it should return 4 tensors
        i = torch.sigmoid(ai)
        f = torch.sigmoid(af)
        o = torch.sigmoid(ao)
        g = torch.tanh(ag)

        next_c = f*c+i*g
        next_h = o*torch.tanh(next_c)
        return next_h, next_c
    