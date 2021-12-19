from torch import nn
from torch.nn.functional import dropout, relu


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutional layers
        self.convs = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.Conv2d(64, 128, 3),
            nn.Conv2d(128, 256, 3),
            nn.Conv2d(256, 256, 3),
            nn.Conv2d(256, 512, 3),
            nn.Conv2d(512, 512, 3)
        )

        # Fully Connected layers
        self.fc = nn.Sequential(
            nn.Linear(512 * 38 * 38, 512),
            nn.Dropout(p=0.5),
            nn.Linear(512, 128),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        print(x.shape)
        x = self.convs(x)

        # Flatten the feature maps into a vector
        # torch.Size([B, 512, 4, 4])
        print(x.shape)
        x = x.view(-1, 512*38*38)
        x = dropout(x, p=0.5, training=self.training)
        print('encoder')
        return self.fc(x)
