import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from .spatial_transformer import SpatialTransformer

 

def get_gradient_2d(start, stop, width, height, is_horizontal):
    if is_horizontal:
        return np.tile(np.linspace(start,stop,width),(height,1))
    else:
        return np.tile(np.linspace(start,stop,height),(width,1)).T

def get_gradient_3d(width,height,start_list,stop_list,is_horizontal_list):
    result=np.zeros((height,width,len(start_list)),dtype=np.float)
    for i, (start,stop,is_horizontal) in enumerate(zip(start_list,stop_list,is_horizontal_list)):
        result[:,:,i]=get_gradient_2d(start,stop,width,height,is_horizontal)
    return result


class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, _, y_pred):
        #dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :]) 
        #dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :]) 
        #dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1]) 
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]) 
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])         

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            #dz = dz * dz

        #d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        #grad = d / 3.0
        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad


class RegNet(nn.Module):

    def __init__(self,img_shape, RATIO=1, CHAN_S2=1, CHAN_S3=1, backbone_in_ch=2, backbone_out_ch=16):
   
        super(RegNet, self).__init__()

        self.ratio = RATIO
        self.cs2 = CHAN_S2
        self.cs3 = CHAN_S3
        self.chan_s2 = nn.MaxPool3d(kernel_size=(4,1,1), stride = (1,1,1), padding = 0)
        self.chan_s3 = nn.MaxPool3d(kernel_size=(4,1,1), stride = (1,1,1), padding = 0)        
        self.unchan = nn.Conv2d(1, self.cs3, 1, stride = 1, padding = 0)
        self.down_s2 = nn.Conv2d(1, 1, RATIO, stride=RATIO, padding=RATIO//2)
                      
        self.unet_model = U_Net2(
            img_ch=backbone_in_ch,
            output_ch=backbone_out_ch
        )
        
        # configure unet to flow field layer
        self.flow = nn.Conv2d(in_channels=backbone_out_ch, out_channels=len(img_shape), kernel_size=3, padding=1)
        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
        self.transformer = SpatialTransformer(img_shape)

    

    def forward(self, source, target, registration=False):
        # Source = S3, Target = S2 
        # Processing s2 data if required
        
        if self.cs3>1:
            source = self.chan_s3(source)
        
        if self.cs2>1:
            target = self.chan_s2(target)

        if self.ratio>1:
            target = self.down_s2(target)
        
        # Concatenate inputs (s3+s2)
        x = torch.cat([source, target], dim=1)
        
        # Propagate U-Net
        x = self.unet_model(x)

        # Transformation into flow field
        flow_field = self.flow(x)
        # Resize flow for integration
        #flow_field = self.resize(flow_field)
        # Integrate to produce diffeomorphic warp
        #flow_field = self.integrate(flow_field)
        # Resize to final resolution
        #flow_field = self.fullsize(flow_field)

        # Warp image with flow field
        y_source = self.transformer(source, flow_field)
        if self.cs3>1:
            y_source = self.unchan(y_source)
        
        return y_source, flow_field


class U_Net2(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(U_Net2,self).__init__()

        # encoder
        self.Conv1 = nn.Conv2d(in_channels=img_ch,out_channels=16,kernel_size=3,stride=2,padding=1)
        self.Conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=2,padding=1)
        self.Conv3 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=2,padding=1)
        self.Conv4 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=2,padding=1)
        
        # decoder
        self.Conv5 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.Conv6 = nn.Conv2d(in_channels=32+32,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.Conv7 = nn.Conv2d(in_channels=32+32,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.Conv8 = nn.Conv2d(in_channels=32+32,out_channels=32,kernel_size=3,stride=1,padding=1)
        
        # extra Conv
        self.Conv9 = nn.Conv2d(in_channels=32+16,out_channels=16,kernel_size=3,stride=1,padding=1)
        self.Conv10 = nn.Conv2d(in_channels=output_ch,out_channels=16,kernel_size=3,stride=1,padding=1)
    
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.activation = nn.LeakyReLU(0.2)

    def forward(self,x):
        
        # encoding path
        x1 = self.activation(self.Conv1(x))
        x2 = self.activation(self.Conv2(x1))
        x3 = self.activation(self.Conv3(x2))
        x4 = self.activation(self.Conv4(x3))

        # decoding + concat path
        d5 = self.activation(self.Conv5(x4))
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.upsample(d5)

        d6 = self.activation(self.Conv6(d5))
        d6 = torch.cat((x3,d6),dim=1)
        d6 = self.upsample(d6)

        d7 = self.activation(self.Conv7(d6))
        d7 = torch.cat((x2,d7),dim=1)
        d7 = self.upsample(d7)

        d8 = self.activation(self.Conv8(d7))
        d8 = torch.cat((x1,d8),dim=1)
        d8 = self.upsample(d8)

        e1 = self.activation(self.Conv9(d8))
        e2 = self.activation(self.Conv10(e1))

        return e2
