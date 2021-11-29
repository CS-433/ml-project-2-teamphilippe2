import torch
from torch import nn

class NNET (nn.Module):
    # down sample * nbre à definir + upsample
    def __init__(self, in_channel=3, batch_norm=False):
        super().__init__()
        out_channel = 64
        self.conv_layer_1 = get_conv_relu(in_channel, out_channel, 2,batch_norm)
        in_channel = out_channel
       
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        in_channel = out_channel
        out_channel *=2
        self.conv_layer_2 = get_conv_relu(in_channel, out_channel, 2,batch_norm)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv_layer_8 = nn.Conv2d(in_channels=out_channel, out_channels = 3)
        
        in_channel = out_channel
        out_channel *=2
        self.conv_layer_3 = get_conv_relu(in_channel, out_channel, 3, batch_norm)
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_layer_9 = nn.Conv2d(in_channels=out_channel, out_channels = 3)
        
        in_channel = out_channel
        out_channel *=2
        self.conv_layer_4 = get_conv_relu(in_channel, out_channel, 3, batch_norm)
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_layer_9 = nn.Conv2d(in_channels=out_channel, out_channels = 3)
        
        in_channel = out_channel
        out_channel *=2
        self.conv_layer_5 = get_conv_relu(in_channel, out_channel, 3, batch_norm)
        self.pool_5 = nn.MaxPool2d(kernel_size=2, stride=2)
        in_channel = out_channel
        out_channel *=8
        self.conv_layer_6 = get_conv_relu(in_channel, out_channel, 1, batch_norm)
        self.drop_out_1 = nn.Dropout(p=0.5, inplace=True)
        in_channel = out_channel
        self.conv_layer_7 = get_conv_relu(in_channel, out_channel, 1, batch_norm)
        self.drop_out_1 = nn.Dropout(p=0.5, inplace=True)
        
        
        
    def forward(self, x):
        conv1 = self.conv_layer_1(x)
        pool_1 = self.pool_1(conv1)
        conv2 = self.conv_layer_2(pool_1)
        pool_2 = self.pool_2(conv2)
        conv3 = self.conv_layer_3(pool_2)
        pool_3 = self.pool_3(conv3)
        
        conv4 = self.conv_layer_4(pool_3)
        pool_4 = self.pool_4(conv4)
        conv5 = self.conv_layer_5(pool_4)
        pool_5 = self.pool_5(conv5)
        
        conv6 = self.conv_layer_5(pool_5)
        drop_1 = self.pool_5(conv6)
        conv7 = self.conv_layer_5(drop_1)
        drop_2 = self.pool_5(conv7)
        
        
        return output
    
    def get_conv_relu(in_chan, out_chan, nb, batch_norm):
        layers = []
        for i in range(nb):
            layers.append(nn.Conv2d(in_channels=in_chan, out_channels = out_chan))
            layers.append(nn.ReLU(inplace=True))
            
        return nn.Sequential(*layers)
    
        
        