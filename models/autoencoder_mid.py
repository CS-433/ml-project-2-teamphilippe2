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

        # Fourth level of convolutions
        self.conv4 = nn.Conv2d(128, 128, 3)

        # Fifth level of convolutions
        self.conv5 = nn.Conv2d(128, 128, 3)

        # Max pooling
        self.maxpool = nn.MaxPool2d(2)

        # Fully Connected layer
        self.fc = nn.Linear(128*8*8, 12)

    def forward(self, x):
        x = relu(self.conv1(x))
        x = relu(self.conv2(x))
        x = relu(self.conv3(x))
        x = relu(self.conv4(x))
        x = relu(self.conv5(x))

        # Flatten the feature maps into a vector
        # torch.Size([B, 128, 8, 8])
        x = x.view(-1, 128*8*8)
        x = dropout(x, p=0.25, training=self.training)

        return self.fc(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        # First fully connected layer
        self.fc = nn.Linear(12, 2048)

        # First deconvolution
        self.first_deconv_channels = 128
        self.deconv1 = nn.ConvTranspose2d(self.first_deconv_channels, 128, 3)

        # Second deconvolution
        self.deconv2 = nn.ConvTranspose2d(128, 64, 3)

        # Third deconvolution
        self.deconv3 = nn.ConvTranspose2d(64, 32, 3)

        # Fourth deconvolution
        self.deconv4 = nn.ConvTranspose2d(32, 16, 3)

        # Fifth deconvolution
        self.deconv5 = nn.ConvTranspose2d(16, 3, 5)

    def forward(self, x):
        x = relu(self.fc(x))
        x = dropout(x, p=0.25, training=self.training)

        # Reshape into 32 feature maps,
        # keeping the batch size
        x = x.view(x.shape[0], self.first_deconv_channels, 4, 4)
        x = relu(self.deconv1(x))
        x = relu(self.deconv2(x))
        x = relu(self.deconv3(x))
        x = relu(self.deconv4(x))
        x = self.deconv5(x)

        return x


class AutoEncoder(nn.Module):
    def __init__(self, encoder=Encoder(), decoder=Decoder()):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)
