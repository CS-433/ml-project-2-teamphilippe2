import torch
from torch import nn

class UNet (nn.Module):
    """
        Creates a U-net a variant of a fully convolutional networks (FCN)
    """
    
    def __init__(self, nber_channels, nber_filters):
        """
            Initialization of the U-net
            
            Parameters
            ----------
            nber_channels : int
                Number of channels of each object fed to the net
            nber_filters : int
                Number of output filters for the first layer of the net
        """
        super().__init__()
        
        self.nber_channels = nber_channels
        self.nber_filters = nber_filters
        
        # Define the encoder layers
        # Layer 1
        in_c = nber_channels
        out_c = nber_filters
        self.fc1 = DownSample(in_c, out_c)
        
        # Layer 2
        in_c = out_c
        out_c = out_c*2
        self.fc2 = DownSample(in_c, out_c)
        
        # Layer 3
        in_c = out_c
        out_c =out_c*2
        self.fc3 = DownSample(in_c, out_c)
        
        # Layer 4
        in_c = out_c
        out_c =out_c*2
        self.fc4 = DownSample(in_c, out_c)
        
        # Define the middle of network composed of 
        # - a last step of down sampling without pooling
        # - a first layer of deconvolution
        
        # Middle down Layer
        in_c = out_c
        out_c =out_c*2
        self.layer_middle_down = nn.Sequential(
            # BN
            nn.BatchNorm2d(in_c),
            # relu
            nn.ReLU(inplace=True),
            # conv
            nn.Conv2d(in_channels = in_c, out_channels = out_c, kernel_size = 3, padding = 1),
            # BN
            nn.BatchNorm2d(out_c),
            # relu
            nn.ReLU(inplace=True),
            # conv
            nn.Conv2d(in_channels = out_c, out_channels = out_c, kernel_size = 3, padding = 1)
        )
        
        # Middle Up Layer
        # Inversion of the values of the input channels and the output channels
        temp = out_c
        out_c = in_c
        in_c = temp
        self.layer_middle_up = nn.ConvTranspose2d(in_c, out_c, kernel_size = 3, stride=(2,2), output_padding = 1, padding = 1)
        
        # Define the decoder layers
        # Layer 5
        self.fc5 = UpSample(in_c, out_c)
        
        # Layer 6
        in_c = out_c
        out_c = out_c//2
        self.fc6 = UpSample(in_c, out_c)
        
        # Layer 7
        in_c = out_c
        out_c =out_c//2
        self.fc7 = UpSample(in_c, out_c)
        
        # Layer 8
        in_c = out_c
        out_c =out_c//2
        self.fc8 = UpSample(in_c, out_c)
        
        # The final layer is composed of 
        # - A 1D convolution with 1 output channel corresponding to the probability of patch being Road 
        # - A sigmoid activation function
        self.fc9 = nn.Sequential(
            nn.Conv2d(in_channels = out_c, out_channels = 1, kernel_size = 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
            Function that feed an input x to the neural net
            
            Parameter
            ---------
            x :
                Data point to feed to the network
                
            Returns
            -------
            tensor :
                Output Tensor of the network
        """
        
        # Decoder part
        conv1, pool1 = self.fc1(x)
        conv2, pool2 = self.fc2(pool1)
        conv3, pool3 = self.fc3(pool2)
        conv4, pool4 = self.fc4(pool3)
        
        # Middle of the network
        middle_down = self.layer_middle_down(pool4)
        middle_up = self.layer_middle_up(middle_down)
        
        # Decoder Part
        concat1 = torch.cat((conv4,middle_up),1)
        deconv1 = self.fc5(concat1)
        concat2 = torch.cat((conv3, deconv1),1)
        deconv2 = self.fc6(concat2)
        concat3 = torch.cat((conv2,deconv2),1)
        deconv3 = self.fc7(concat3)
        concat4 = torch.cat((conv1,deconv3),1)
        deconv4 = self.fc8(concat4)
        
        output = self.fc9(deconv3)
        return output
        
        
class DownSample (nn.Module):
    """
        Creates a Down Sampling block
    """
    
    def __init__(self,nber_channels, nber_filters):
        """
            Intialization of the Down Sampling Block
            
            Parameters
            ----------
            nber_channels : int
                Number of channels of each object fed to the net
            nber_filters : int
                Number of output filters for the first layer of the net
            
        """
        
        super().__init__()
        
        self.nber_channels = nber_channels
        self.nber_filters = nber_filters
        
        # A Down Sampling block is composed of 2 times the following layers :
        # - A Batch Norm layer
        # - a ReLU layer
        # - a 2D Convolution
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
            nn.Conv2d(in_channels = nber_filters, out_channels = nber_filters, kernel_size = 3, padding = 1)
        )
        
        # Last step of Block : a Pooling Layer
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
    def forward(self, x):
        """
            Function that feeds input x to block
            
            Parameter
            ---------
            x :
                Data point to feed to the network
                
            Returns
            -------
            tensor :
                Output Tensor of the block
            tensor :
                Pooled output Tensor of the block
       
        """
        
        conv_out = self.layer(x)
        return conv_out, self.pool(conv_out)
    
class UpSample (nn.Module):
    """
        Creates a Up Sampling block
    """
    
    def __init__(self, nber_channels, nber_filters):
         """
            Intialization of the Up Sampling Block
            
            Parameters
            ----------
            nber_channels : int
                Number of channels of each object fed to the net
            nber_filters : int
                Number of output filters for the first layer of the net
            
        """
        
        super().__init__()
        
        self.nber_channels = nber_channels
        self.nber_filters = nber_filters
        
        # A Up Sampling block is composed of the following layers :
        # - A convolution block
        # - a 2D Deconvolution layer
        self.layer = nn.Sequential(
            Convolutions(nber_channels, nber_filters),
            nn.ConvTranspose2d(nber_filters, nber_filters//2, kernel_size = 3, padding = 1, output_padding = 1, stride=(2,2))
        )
    def forward(self, x):
        """
            Function that feeds input x to block
            
            Parameter
            ---------
            x :
                Data point to feed to the network
                
            Returns
            -------
            tensor :
                Output Tensor of the block
       
        """
        
        return self.layer(x)
    
    
class Convolutions (nn.Module):
    """
        Creates a Convolution block
    """
    
    def __init__(self,nber_channels, nber_filters):
        """
            Intialization of the Convolution Block
            
            Parameters
            ----------
            nber_channels : int
                Number of channels of each object fed to the net
            nber_filters : int
                Number of output filters for the first layer of the net
            
        """
        
        super().__init__()
        
        self.nber_channels = nber_channels
        self.nber_filters = nber_filters
        
        # A Convolution block is composed of 2 2D Convolutions put into sequence
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels = nber_channels, out_channels = nber_filters, kernel_size=3, padding=1),
            nn.Conv2d(in_channels = nber_filters, out_channels = nber_filters, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        """
            Function that feeds input x to block
            
            Parameter
            ---------
            x :
                Data point to feed to the network
                
            Returns
            -------
            tensor :
                Output Tensor of the block
       
        """
        
        return self.layer(x)