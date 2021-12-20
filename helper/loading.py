import matplotlib.image as mpimg
import os
import numpy as np
import torch
from torchvision.transforms.functional import resize
from helper.const import *

def load_image(image_file):
    """
    Load one image from the image file
    Parameters:
    -----------
        - image_file:
            The image file path
    Returns:
    -----------
        - The image
    """
    return mpimg.imread(image_file)


def load_all_images_in_folder(image_folder, nb=-1):
    """
    Load all images in the given folder
    Parameters:
    -----------
        - image_folder: 
        - nb: the number of images to load (or -1 if we need to load all of them)
    Returns:
    -----------
        A list with all images
    """
    files = os.listdir(image_folder)
    n = min(nb, len(files)) if nb != -1 else len(files)

    return [load_image(image_folder + files[i]) for i in range(n)]


def load_images_and_groundtruth(image_folder, groundtruth_folder, nb=-1):
    """
    Load all images in the two given folders
    Parameters:
    -----------
        - image_folder: The image folder path
        - groundtruth_folder: The groundtruth image folder path
        - nb: the number of images to load (or -1 if we need to load all of them)
    Returns:
    -----------
        Tuple: 
            - list of all images
            - list of all groundtruth images
    """
    imgs, gt_imgs = load_all_images_in_folder(image_folder, nb), load_all_images_in_folder(groundtruth_folder, nb)

    print(f"Loaded {len(imgs)} images")

    diff_array = np.diff([[img.shape[0], img.shape[1], img.shape[2]] for img in imgs], axis=0)
    b_same_size = np.all(diff_array == 0)

    print("All images have the same size !" if b_same_size else "Not a uniform size")
    if b_same_size:
        print(f'Image size = {imgs[0].shape[0]}, {imgs[0].shape[1]}')
    return imgs, gt_imgs


def split_data(x, y, ratio, seed=1):
    """
        Split the given dataset into 2 different datasets (local train/test)
        according to the given ratio
        
        Parameters
        ----------
            x :
                Data points
            y :
                Outputs of the data points
            ratio :
                Ratio of samples to keep in the training set
            seed :
                Seed to initialize the RNG
        Returns 
        -------
            Tuple:
                - x_train
                - y_train
                - x_test
                - y_test
    """
    # set seed
    np.random.seed(seed)
    nb_data = len(x)
    idx = np.random.choice(nb_data, int(ratio * nb_data), replace=False)
    y = np.array(y)
    x = np.array(x)

    return x[idx], y[idx], np.delete(x, idx, axis=0), np.delete(y, idx, axis=0)


def resize_image_test(tensor, resized_width, resized_height):
    """
    Resize the given tensor to the asked size
    
    Parameters: 
    -----------
        - tensor:
            The tensor to resize
        - resized_width: 
            The new width
        - resized_height:
            the new height
    Returns: 
    -----------
        - The transformed images
    """
    # Transform the tensor from (H,W,3) to (3,H,W)
    tensor = torch.permute(tensor, (2, 0, 1))

    # Resize the image to the given dimensions
    tensor = resize(tensor, [resized_height, resized_width])

    return tensor


def load_test_set(test_datapath, resized_width, resized_height):
    """
    Load all the test set images
    Parameters:
    -----------
        - test_datapath:
            The folder containing all the test images.
        - resized_width: 
            The new width
        - resized_height:
            the new height
    Returns:
    -----------
        - ids:
            Ids of the test images
        - test_imgs: 
            All the resized test images
        - orig_size:
            A list containing all the original size of the images
    """
    pref_length = len(test_image_prefix)
    suf_length = len(test_image_suffix)

    # List of all the subfolders
    files = os.listdir(test_datapath)

    ids = []
    test_imgs = []
    orig_size = []
    files = sorted(files, key=lambda x: int(x[pref_length:]))

    for file in files:
        # Each folder contains only one image
        im_in_dir = os.listdir(test_datapath + file)[0]

        # load image, resize it and transform it to tensor
        test_img = load_image(test_datapath + file + "/" + im_in_dir)
        tensor = resize_image_test(torch.from_numpy(test_img), resized_width, resized_height)

        # Save image on list
        test_imgs.append(tensor)
        ids.append(int(file[pref_length:]))
        orig_size.append((test_img.shape[0], test_img.shape[1]))

    return ids, test_imgs, orig_size


def save_numpy(X, file):
    """
    Save matrix X to disk
    Parameters:
    -----------
        - X:
            Numpy array to save
        - file:
            File to save to
    Returns:
    -----------
    """
    with open(file, 'wb') as f:
        np.save(f, X)


def load_numpy(file):
    """
    Load matrix X from disk
    Parameters:
    -----------
        - file:
            File to load from
    Returns:
    -----------
        - X:
            Numpy array loaded
    """
    with open(file, 'rb') as f:
        X = np.load(f)
    return X


def load_model_weights(model, weights_path):
    """
    Loads the weights from the given file
    Parameters:
    -----------
        - model:
            Model of the network
        - weights_path:
            Path of the file
    -----------
    """
    model.load_state_dict(torch.load(weights_path))