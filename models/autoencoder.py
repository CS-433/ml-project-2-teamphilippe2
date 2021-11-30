from torch import nn
from torch.nn.functional import dropout, relu


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        # First level of convolutions
        # input 400x400x3
        self.conv1 = nn.Conv2d(3, 16, 5)

        # Second level of convolutions
        # input x400x16
        self.conv2 = nn.Conv2d(16, 32, 5)

        # Third level of convolutions
        self.conv3 = nn.Conv2d(32, 64, 3)

        # Max pooling
        self.maxpool = nn.MaxPool2d(2)

        # Fully Connected layer
        self.fc = nn.Linear(64*47*47, 64)

    def forward(self, x):
        x = relu(self.conv1(x))
        x = self.maxpool(x)

        x = relu(self.conv2(x))
        x = self.maxpool(x)

        x = relu(self.conv3(x))
        x = self.maxpool(x)

        # Flatten the feature maps into a vector
        # torch.Size([B, 64, 47, 47])
        x = x.view(-1, 64*47*47)
        x = dropout(x, p=0.25, training=self.training)

        return self.fc(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        # First fully connected layer
        self.fc = nn.Linear(64, 256)

        # First deconvolution
        self.deconv1 = nn.ConvTranspose2d(32, 16, 3)

        # Second deconvolution
        self.deconv2 = nn.ConvTranspose2d(16, 8, 5)

        # Third deconvolution
        self.deconv3 = nn.ConvTranspose2d(8, 3, 5)

    def forward(self, x):
        x = relu(self.fc(x))
        x = dropout(x, p=0.25, training=self.training)

        # Reshape into 32 feature maps
        x = x.view(-1, 32)
        x = relu(self.deconv1(x))
        x = relu(self.deconv2(x))
        x = self.deconv3(x)

        return x


class AutoEncoder(nn.Module):
    def __init__(self):
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)
