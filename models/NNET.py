import torch
from torch import nn

class NNet(nn.Module):
    # down sample * nbre à definir + upsample
    def __init__(self, in_channel=400, batch_norm=False):
        super().__init__()
        
        channel = 64
        
        self.conv_layer_1 = get_conv_relu(in_channel, channel, 2,batch_norm)
       
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_layer_2 = get_conv_relu(channel, channel*2, 2, batch_norm)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv_layer_8 = nn.Conv2d(in_channels=channel*2, out_channels = 3,kernel_size=3)
        
        self.conv_layer_3 = get_conv_relu(channel*2, channel*4, 3, batch_norm)
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_layer_9 = nn.Conv2d(in_channels=channel*4, out_channels = 3,kernel_size=3)
        
        self.conv_layer_4 = get_conv_relu(channel*4, channel*8, 3, batch_norm)
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_layer_10 = nn.Conv2d(in_channels=channel*8, out_channels = 3,kernel_size=3)
        
        self.conv_layer_5 = get_conv_relu(channel*8, channel*16, 3, batch_norm)
        self.pool_5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_layer_6 = get_conv_relu(channel*16, channel*16, 1, batch_norm)
        self.drop_out_1 = nn.Dropout(p=0.5, inplace=True)
        
        self.conv_layer_11 = nn.Conv2d(in_channels=channel*16, out_channels = 3,kernel_size=3)
        
        self.conv_layer_merge_1 = nn.Conv2d(in_channels=6, out_channels=3,kernel_size=3)
        self.conv_layer_merge_2 = nn.Conv2d(in_channels=6, out_channels=3,kernel_size=3)
        self.conv_layer_merge_3 = nn.Conv2d(in_channels=6, out_channels=3,kernel_size=3)
        
        self.deconv_1 = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=3, stride=2)
        self.deconv_2 = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=3, stride=2)
        self.deconv_3 = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=3, stride=2)
        self.deconv_4 = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=3, stride=2)
        
        self.sigm = nn.Sigmoid()
        
    def forward(self, x):
        conv1 = self.conv_layer_1(x)
        pool_1 = self.pool_1(conv1)
        conv2 = self.conv_layer_2(pool_1)
        pool_2 = self.pool_2(conv2)
        conv3 = self.conv_layer_3(pool_2)
        pool_3 = self.pool_3(conv3)
        
        conv4 = self.conv_layer_4(pool_3)
        pool_4 = self.pool_4(conv4)
        
        conv_8 = self.conv_layer_8(pool_2)
        conv_9 = self.conv_layer_9(pool_3)
        conv_10 = self.conv_layer_10(pool_4)
        
        conv5 = self.conv_layer_5(pool_4)
        pool_5 = self.pool_5(conv5)
        
        conv6 = self.conv_layer_6(pool_5)
        drop_1 = self.drop_out_1(conv6)
        
        conv_11 = self.conv_layer_11(drop_1)
        deconv_1 = self.deconv_1(conv_11)
        
        first_merge = torch.cat((deconv_1, conv_10), 2)
        after_first_merge = self.conv_layer_merge_1(first_merge)
        deconv_2 = self.deconv_2(after_first_merge)
        
        second_merge = torch.cat((deconv_2, conv_10), 2)
        after_second_merge =self.conv_layer_merge_2(second_merge)
        deconv_3 = self.deconv_3(after_second_merge)
        
        third_merge = torch.cat((deconv_3, conv_9), 2)
        after_third_merge =self.conv_layer_merge_3(third_merge)
        deconv_4 = self.deconv_4(after_third_merge)
        
        fourth_marge = torch.cat((deconv_4, conv_8), 2)
        after_fourth_merge =self.conv_layer_merge_4(fourth_marge)
        
        return self.sigm(after_fourth_merge)
    
def get_conv_relu(in_chan, out_chan, nb, batch_norm):
    layers = []
    for i in range(nb):
        layers.append(nn.Conv2d(in_channels=in_chan, out_channels = out_chan, kernel_size=3))
        layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)
    
        
        