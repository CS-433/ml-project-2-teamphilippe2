import torch
from torch import nn


class FCNet(nn.Module):
    """
    Custom class representing our fully convolutional layers
    """
    def __init__(self, in_channel=3, channel=32, batch_norm=True):
        super().__init__()

        # Build all the different layers for the architecture explained in the report
        self.res_block1 = Block(in_channel, channel, 2, batch_norm)
        self.res_block2 = Block(channel, channel * 2, 2, batch_norm)
        self.res_block3 = Block(channel * 2, channel * 4, 3, batch_norm)
        self.res_block4 = Block(channel * 4, channel * 8, 3, batch_norm)
        self.res_block5 = Block(channel * 8, channel * 16, 1, batch_norm)

        self.conv_layer_8 = nn.Conv2d(in_channels=channel * 2, out_channels=3, kernel_size=3, padding=1)
        self.conv_layer_9 = nn.Conv2d(in_channels=channel * 4, out_channels=3, kernel_size=3, padding=1)
        self.conv_layer_10 = nn.Conv2d(in_channels=channel * 8, out_channels=3, kernel_size=3, padding=1)
        self.drop_out_1 = nn.Dropout(p=0.5, inplace=True)
        self.conv_layer_11 = nn.Conv2d(in_channels=channel * 16, out_channels=3, kernel_size=3, padding=1)

        self.drop_out_2 = nn.Dropout(p=0.5, inplace=True)

        self.conv_layer_merge_1 = get_merge_conv_layer()
        self.conv_layer_merge_2 = get_merge_conv_layer()
        self.conv_layer_merge_3 = get_merge_conv_layer()

        self.deconv_5 = get_deconv_layer()
        self.deconv_4 = get_deconv_layer()
        self.deconv_2 = get_deconv_layer()
        self.deconv_3 = get_deconv_layer()
        self.deconv_1 = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=4, stride=(2, 2),
                                           output_padding=1, padding=1)

        self.final_conv = nn.Conv2d(in_channels=in_channel, out_channels=1, kernel_size=3, padding=1)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        # Downsampling part
        block1 = self.res_block1(x)
        block2 = self.res_block2(block1)
        block3 = self.res_block3(block2)
        block4 = self.res_block4(block3)
        drop_1 = self.drop_out_1(block4)
        block5 = self.res_block5(drop_1)
        drop_2 = self.drop_out_1(block5)

        # Skip layers
        conv_8 = self.conv_layer_8(block2)
        conv_9 = self.conv_layer_9(block3)
        conv_10 = self.conv_layer_10(block4)
        conv_11 = self.conv_layer_11(drop_2)

        # Up sampling and concatenation
        deconv_1 = self.deconv_1(conv_11)
        concat1 = torch.cat((deconv_1, conv_10), 1)
        merge1 = self.conv_layer_merge_1(concat1)
        deconv_2 = self.deconv_2(merge1)

        concat2 = torch.cat((deconv_2, conv_9), 1)
        merge2 = self.conv_layer_merge_2(concat2)
        deconv_3 = self.deconv_3(merge2)

        concat3 = torch.cat((deconv_3, conv_8), 1)
        merge3 = self.conv_layer_merge_3(concat3)
        deconv_4 = self.deconv_4(merge3)
        deconv_5 = self.deconv_5(deconv_4)
        final_conv = self.final_conv(deconv_5)

        return final_conv


class Block(nn.Module):
    """
    This class represent a number o of successive Convolutional, Batch norm (if requested) and Relu layers. It ends
    by applying a Max pooling to reduce the size of the input image
    """
    def __init__(self, in_chan, out_chan, nb, batch_norm):
        """
        Build the requested block
        Parameters:
        -----------
            - in_chan: the number of input channels
            - out_chan: the number of output channels
            - nb: The number of time we put Convolutional, Batch norm and ReLu layers  in a row
            - batch_norm: whether we should use Batch Normalisation layers
        """
        super().__init__()
        layers = []
        for i in range(nb):
            layers.append(nn.Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=3, padding=1))

            if batch_norm:
                layers.append(nn.BatchNorm2d(out_chan))

            layers.append(nn.ReLU(inplace=True))
            in_chan = out_chan

        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.all_lays = nn.Sequential(*layers)

    def forward(self, x):
        return self.all_lays(x)


def get_deconv_layer():
    """
    Create and return a transpose convolution layer used during upsampling
    Returns:
    --------
        The requested layer
    """
    return nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=3, stride=(2, 2),
                              output_padding=1, padding=1)


def get_merge_conv_layer():
    """
    Return a convolutional layer used in merge (6 inputs channels and 3 in output)
    Returns:
    --------
        A convolutional layer used in merge
    """
    return nn.Conv2d(in_channels=6, out_channels=3, kernel_size=3, padding=1)
