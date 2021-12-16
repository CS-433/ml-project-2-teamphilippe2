import torch
from torch import nn


class NNet(nn.Module):
    # down sample * nbre à definir + upsample
    def __init__(self, in_channel=3, batch_norm=True):
        super().__init__()

        channel = 32
        self.res_block1 = ResBlock(in_channel, channel, 2, batch_norm)
        self.res_block2 = ResBlock(channel, channel * 2, 2, batch_norm)
        self.res_block3 = ResBlock(channel * 2, channel * 4, 3, batch_norm)
        self.res_block4 = ResBlock(channel * 4, channel * 8, 3, batch_norm)
        self.res_block5 = ResBlock(channel * 8, channel * 16, 1, batch_norm)

        self.conv_layer_8 = nn.Conv2d(in_channels=channel * 2, out_channels=3, kernel_size=3,padding=1)
        self.conv_layer_9 = nn.Conv2d(in_channels=channel * 4, out_channels=3, kernel_size=3,padding=1)
        self.conv_layer_10 = nn.Conv2d(in_channels=channel * 8, out_channels=3, kernel_size=3,padding=1)
        self.drop_out_1 = nn.Dropout(p=0.5, inplace=True)
        self.conv_layer_11 = nn.Conv2d(in_channels=channel * 16, out_channels=3, kernel_size=3,padding=1)

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
        res_block1 = self.res_block1(x)

        res_block2 = self.res_block2(res_block1)

        res_block3 = self.res_block3(res_block2)
        res_block4 = self.res_block4(res_block3)
        drop_1 = self.drop_out_1(res_block4)
        res_block5 = self.res_block5(drop_1)
        drop_2 = self.drop_out_1(res_block5)

        conv_8 = self.conv_layer_8(res_block2)
        conv_9 = self.conv_layer_9(res_block3)
        conv_10 = self.conv_layer_10(res_block4)
        conv_11 = self.conv_layer_11(drop_2)

        deconv_1 = self.deconv_1(conv_11)
        concat1 = torch.cat((deconv_1, conv_10), 1)
        merge1 = self.conv_layer_merge_1(concat1)
        deconv_2 = self.deconv_2(merge1)

        concat2 = torch.cat((deconv_2, conv_9), 1)
        merge2 = self.conv_layer_merge_2(concat2)
        deconv_3 = self.deconv_3(merge2)

        concat3 = torch.cat((deconv_3, conv_8),1)
        merge3 = self.conv_layer_merge_3(concat3)
        deconv_4 = self.deconv_4(merge3)
        deconv_5 = self.deconv_5(deconv_4)
        final_conv = self.final_conv(deconv_5)

        return final_conv

class ResBlock(nn.Module):
    def __init__(self, in_chan, out_chan, nb, batch_norm):
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
    return nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=3, stride=(2, 2),
                              output_padding=1, padding=1)

def get_merge_conv_layer():
    return nn.Conv2d(in_channels=6, out_channels=3, kernel_size=3, padding=1)
