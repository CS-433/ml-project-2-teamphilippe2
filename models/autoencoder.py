from torch import nn
from torch.nn.functional import dropout, relu


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        # First level of convolutions
        # input 16x16x3
        self.conv1 = nn.Conv2d(3, 64, 5)

        # Second level of convolutions
        self.conv2 = nn.Conv2d(64, 128, 5)

        # Third level of convolutions
        self.conv3 = nn.Conv2d(128, 256, 5)

        # Fourth level of convolutions
        self.conv4 = nn.Conv2d(256, 256, 3)

        # Fifth level of convolutions
        self.conv5 = nn.Conv2d(256, 512, 3)

        # Sixth level of convolutions
        self.conv6 = nn.Conv2d(512, 512, 3)

        # Seventh level of convolutions
        self.conv7 = nn.Conv2d(512, 512, 3)

        # Fully Connected layer
        self.fc = nn.Linear(512*38*38, 512)

        self.max_pooling = nn.MaxPool2d(2)

    def forward(self, x):
        print('encoder')
        print(x.shape)
        x = relu(self.conv1(x))
        x = self.max_pooling(x)
        x = relu(self.conv2(x))
        x = self.max_pooling(x)
        x = relu(self.conv3(x))
        x = self.max_pooling(x)
        x = relu(self.conv4(x))
        x = relu(self.conv5(x))
        x = relu(self.conv6(x))
        x = relu(self.conv7(x))

        # Flatten the feature maps into a vector
        # torch.Size([B, 512, 4, 4])
        print(x.shape)
        x = x.view(-1, 512*38*38)
        #x = dropout(x, p=0.25, training=self.training)
        print('encoder')
        return self.fc(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        # First fully connected layer
        self.fc = nn.Linear(512, 8192)

        # First deconvolution
        self.first_deconv_channels = 512
        self.deconv1 = nn.ConvTranspose2d(self.first_deconv_channels, 256, 3, stride=2, padding=1)

        # Second deconvolution
        self.deconv2 = nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1)

        # Third deconvolution
        self.deconv3 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1)

        # Fourth deconvolution
        self.deconv4 = nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1)

        # Fifth deconvolution
        self.deconv5 = nn.ConvTranspose2d(128, 64, 5, stride=2, padding=1)

        # Sixth deconvolution
        self.deconv6 = nn.ConvTranspose2d(64, 64, 5, stride=2, padding=1)

        # Seventh deconvolution
        self.deconv7 = nn.ConvTranspose2d(64, 3, 5, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        print('decoder')
        x = relu(self.fc(x))
        x = dropout(x, p=0.25, training=self.training)
        print(x.shape)
        # Reshape into 512 feature maps,
        # keeping the batch size
        x = x.view(x.shape[0], self.first_deconv_channels, 4, 4)

        x = relu(self.deconv1(x))
        x = relu(self.deconv2(x))
        x = relu(self.deconv3(x))
        x = relu(self.deconv4(x))
        x = relu(self.deconv5(x))
        x = relu(self.deconv6(x))
        x = self.deconv7(x)
        print(x.shape)
        print('decoder')
        return x


class AutoEncoder(nn.Module):
    def __init__(self, encoder=Encoder(), decoder=Decoder()):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)
