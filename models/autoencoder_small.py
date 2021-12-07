from torch import nn
from torch.nn.functional import dropout, relu


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        # First level of convolutions
        # input 16x16x3
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)

        # Second level of convolutions
        self.conv2 = nn.Conv2d(32, 64, 3)

        # Third level of convolutions
        self.conv3 = nn.Conv2d(64, 128, 3)

        # Max pooling
        self.maxpool = nn.MaxPool2d(2)

        # Fully Connected layer
        self.fc = nn.Linear(128*5*5, 16)

    def forward(self, x):
        x = relu(self.conv1(x))

        x = relu(self.conv2(x))
        x = self.maxpool(x)

        x = relu(self.conv3(x))

        # Flatten the feature maps into a vector
        # torch.Size([B, 128, 5, 5])
        x = x.view(-1, 128*5*5)
        x = dropout(x, p=0.25, training=self.training)

        return self.fc(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        # First fully connected layer
        self.fc = nn.Linear(16, 128)

        # First deconvolution
        self.first_deconv_channels = 32
        self.deconv1 = nn.ConvTranspose2d(self.first_deconv_channels, 16, 3, stride=(2, 2), padding=1, output_padding=1)

        # Second deconvolution
        self.deconv2 = nn.ConvTranspose2d(16, 8, 3, stride=(2, 2), padding=1)

        # Third deconvolution
        self.deconv3 = nn.ConvTranspose2d(8, 3, 5, stride=(2, 2), padding=1, output_padding=1)

    def forward(self, x):
        x = relu(self.fc(x))
        x = dropout(x, p=0.25, training=self.training)

        # Reshape into 32 feature maps,
        # keeping the batch size
        x = x.view(x.shape[0], self.first_deconv_channels, 2, 2)
        x = relu(self.deconv1(x))

        x = relu(self.deconv2(x))
        x = self.deconv3(x)

        return x


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)
