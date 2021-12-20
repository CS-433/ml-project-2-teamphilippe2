import numpy as np
import torch
from torch.utils.data import DataLoader

from helper.datasets_patch import AutoencoderTrainingRoadPatches
from helper.loading import save_numpy
from helper.neural_net import load_model_weights
from helper.const import *


def extract_features_encoder(encoder, image_dir, weights_path):
    """
    Extract the patch features with the given trained encoder.
    Parameters:
    -----------
        - encoder:
            The trained encoder
        - image_dir:
            The directory containing the images
        - weights_path:
            The file containing the trained weights (without extension)
            for the encoder
        """
    # Load the trained encoder
    load_model_weights(encoder, weights_folder + "autoencoder/" + weights_path + ".pth")

    # Disable dropout etc
    encoder.eval()

    # Use GPU if available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Use GPU for features extraction.')
    else:
        device = torch.device('cpu')
        print('Use CPU for features extraction.')

    encoder = encoder.to(device)

    # Dataset to extract the features from
    dataset = AutoencoderTrainingRoadPatches(image_dir)
    dataloader = DataLoader(dataset, batch_size=1)

    # Matrix of extracted features of shape (nb samples, nb features)
    X = None
    i = 0

    print('Starting features extraction...')
    for img, _ in dataloader:
        img = img.to(device)

        # Batch size of 1, drop the 1 first batch dim
        features = encoder(img)[0].cpu().detach().numpy()

        if X is None:
            X = np.empty((len(dataloader), features.shape[0]))

        X[i, :] = features
        i += 1

    # Save features to disk
    file_path = features_folder + f'features_{weights_path}.npy'
    save_numpy(X, file_path)
    print(f'Features saved in {file_path}')

    return X

