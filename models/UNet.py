import torch
from torch import nn

class UNet (nn.Module):
    # down sample * nbre à definir + upsample
    def __init__(self, nber_channels, nber_filters):
        super().__init__()
        
        self.nber_channels = nber_channels
        self.nber_filters = nber_filters
        
        # encoder
        in_c = nber_channels
        out_c = nber_filters
        self.fc1 = DownSample(in_c, out_c)
        
        in_c = out_c
        out_c = out_c*2
        self.fc2 = DownSample(in_c, out_c)
        
        in_c = out_c
        out_c =out_c*2
        self.fc3 = DownSample(in_c, out_c)
        
        in_c = out_c
        out_c =out_c*2
        self.fc4 = DownSample(in_c, out_c)
        
        # middle of network
        # convolutions
        in_c = out_c
        out_c =out_c*2
        self.layer_middle_down = Convolutions(in_c, out_c)
        # inversion of the values for in_c and out_c + deconvolution
        temp = out_c
        out_c = in_c
        in_c = temp
        
        # Which kernel_size ? 
        self.layer_middle_up = nn.ConvTranspose2d(in_c, out_c, kernel_size=3)
        
        # decoder
        self.fc5 = UpSample(in_c, out_c)
        in_c = out_c
        out_c = out_c//2
        self.fc6 = UpSample(in_c, out_c)
        in_c = out_c
        out_c =out_c//2
        self.fc7 = UpSample(in_c, out_c)
        
        # final layer = 1D convolution + sigmoid
        self.fc8 = nn.Sequential(
            # out channels = 2 because we want only white or black
            nn.Conv2d(in_channels = out_c, out_channels = 2, kernel_size = 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        conv1, pool1 = self.fc1(x)
        conv2, pool2 = self.fc2(pool1)
        conv3, pool3 = self.fc3(pool2)
        conv4, pool4 = self.fc4(pool3)
        
        middle_down = self.layer_middle_down(pool4)
        middle_up = self.layer_middle_up(middle_down)
        
        concat1 = torch.cat((pool3,middle_up),1)
        deconv1 = self.fc5(concat1)
        concat2 = torch.cat((pool2, deconv1),1)
        deconv2 = self.fc6(concat2)
        concat3 = torch.cat((pool1,deconv2),1)
        deconv3 = self.fc7(concat3)
        
        output = self.fc8(deconv3)
        return output
        
        
class DownSample (nn.Module):
    # conv + relu + pool
    def __init__(self,nber_channels, nber_filters):
        super().__init__()
        
        self.nber_channels = nber_channels
        self.nber_filters = nber_filters
        
        # layer = res block*2 + pool
        # res block = bn + relu + conv
        # pool = max pooling
        self.layer = nn.Sequential(
            # BN
            nn.BatchNorm2d(nber_channels),
            # relu
            nn.ReLU(inplace=True),
            # conv
            nn.Conv2d(in_channels = nber_channels, out_channels = nber_filters, kernel_size = 3, padding = 1),
            # BN
            nn.BatchNorm2d(nber_filters),
            # relu
            nn.ReLU(inplace=True),
            # conv
            nn.Conv2d(in_channels = nber_filters, out_channels = nber_filters, kernel_size = 3, padding = 1),
        )
        
        #pooling
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
    def forward(self, x):
        conv_out = self.layer(x)
        return conv_out, self.pool(conv_out)
    
class UpSample(nn.Module):
    def __init__(self, num_channels, num_filters):
        super().__init__()
        self.num_channels = num_channels
        self.num_filters = num_filters
        
        conv = Convolutions(num_channels, num_filters)
        trans_conv = nn.ConvTranspose2d(
            in_channels=num_filters,
            out_channels=num_filters // 2,
            kernel_size=3,
            stride=(2, 2),
            padding=1,
            output_padding=1
        )
        self.up_samp = nn.Sequential(
            conv,
            trans_conv
        )

    def forward(self, input):
        return self.up_samp(input)
    
class Convolutions (nn.Module):
    # convolutions in up sample part
    def __init__(self,nber_channels, nber_filters):
        super().__init__()
        
        self.nber_channels = nber_channels
        self.nber_filters = nber_filters
        
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels = nber_channels, out_channels = nber_filters, kernel_size=3, padding=1),
            nn.Conv2d(in_channels = nber_filters, out_channels = nber_filters, kernel_size=3, padding=1)
        )
        
    def foward(self, x):
        return self.layer(x)