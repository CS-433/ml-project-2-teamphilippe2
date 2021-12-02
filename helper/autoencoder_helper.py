import numpy as np
import torch
from torch.utils.data import DataLoader

from helper.data_augmentation import OriginalTrainingRoadPatches
from helper.loading import save_numpy
from helper.neural_net import load_model_weights


def extract_features_encoder(encoder, image_dir, weights_path, features_file):
    # Load the trained encoder
    load_model_weights(encoder, weights_path)

    # Disable dropout etc
    encoder.eval()

    # Use GPU if available
    if torch.cuda.is_available():
        device = torch.device('cpu')
        print('Use CPU for features extraction.')
    else:
        device = torch.device('cuda')
        print('Use GPU for features extraction.')

    encoder = encoder.to(device)

    # Dataset to extract the features from
    dataset = OriginalTrainingRoadPatches(image_dir)
    dataloader = DataLoader(dataset, batch_size=1)

    # Matrix of extracted features of shape (nb samples, nb features)
    X = None

    print('Starting features extraction...')
    for i, img in enumerate(dataloader):
        img = img.to(device)

        # Batch size of 1, drop the 1 first batch dim
        features = encoder(img)[0].numpy()

        if X is None:
            X = np.empty((len(dataloader), features.shape[0]))

        X[i, :] = features

    # Save features to disk
    save_numpy(features_file, X)

    return X
