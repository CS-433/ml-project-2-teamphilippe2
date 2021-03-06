from helper.image import *
import matplotlib.pyplot as plt
from PIL import Image


def display_imgs_side_by_side(img, gt_img):
    """
    Display two images side by side
    Parameters
    ----------
        - img: First image
        - gt_img: the second image.
    Return:
        - cimg: the concatenated image
    """
    cimg = concatenate_images(img, gt_img)
    fig1 = plt.figure(figsize=(10, 10))
    plt.imshow(cimg, cmap='Greys_r')
    return cimg


def make_img_overlay(img, predicted_img):
    """
    Display prediction on top of base image

    Parameters
    ----------
        img: The basic image
        predicted_img: The predicted road 
    Returns 
    -------
        new_img: the superposition of the two images
    """
    fig1 = plt.figure(figsize=(10, 10))

    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:, :, 0] = predicted_img * 255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    plt.imshow(new_img)
    return new_img


def plot_features_distributions(X, save_img=False, path='plots/baseline_features_dist.png'):
    """
    Plot the distributions of each of the
    features in X

    Parameters
    ----------
        X :
            Data to plot
        save_img:
            Whether to save the plot
        path:
            Where to save the plot

    Returns 
    -------
    """
    nb_features = X.shape[1]
    labels_features = {0: 'Mean', 1: 'Variance', 2: 'Max'}
    labels_channels = {0: 'Red', 1: 'Green', 2: 'Blue'}

    fig, axs = plt.subplots(3, 3, figsize=(10, 10))

    for i in range(nb_features):
        row = i // 3
        col = i % 3
        axs[row, col].hist(X[:, i], bins=100)
        axs[row, col].set_title(f'Histogram of {labels_features[row]} ({labels_channels[col]})')
        axs[row, col].set_ylabel('Count')

    fig.tight_layout()

    if save_img:
        fig.savefig(path)

    fig.show()


def plot_channels_img(img):
    """
    Plots the 3 channels of the given image

    Parameters
    ----------
        img :
            Image to plot

    Returns 
    -------
    """
    fig, axs = plt.subplots(1, 3, figsize=(16, 48))

    # Isolate the 3 channels and plot them
    imr = np.zeros((img.shape))
    imr[:, :, 0] = img[:, :, 0]
    axs[0].imshow(imr)

    imr = np.zeros((img.shape))
    imr[:, :, 1] = img[:, :, 1]
    axs[1].imshow(imr)

    imr = np.zeros((img.shape))
    imr[:, :, 2] = img[:, :, 2]
    axs[2].imshow(imr)

    plt.show()
