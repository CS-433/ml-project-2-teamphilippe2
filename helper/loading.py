import matplotlib.image as mpimg
import os 
import numpy as np
import torch
from torchvision.transforms.functional import resize

from helper.const import *
# Helper functions
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
    
    diff_array =np.diff([[img.shape[0], img.shape[1], img.shape[2]] for img in imgs], axis=0)
    b_same_size = np.all(diff_array==0)
    
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
    idx = np.random.choice(nb_data, int(ratio*nb_data), replace=False)
    y = np.array(y)
    x = np.array(x)

    return x[idx], y[idx], np.delete(x, idx, axis=0), np.delete(y, idx, axis=0)
    
def resize_image_test(tensor, resized_width, resized_height):
    tensor = torch.transpose(tensor, 0, 2)
    tensor = torch.transpose(tensor, 1, 2)
    tensor = resize(tensor, [resized_height, resized_width])
    tensor = torch.transpose(tensor, 1, 2)
    tensor = torch.transpose(tensor, 0, 2)
    
    return tensor

def load_test_set(test_datapath, resized_width, resized_height):

    pref_length = len(test_image_prefix)
    suf_length = len(test_image_suffix)
    
    files = os.listdir(test_datapath)
    ids = []
    test_imgs = []
    orig_size = []
    
    for file in files:
        # Each folder contains only one image
        im_in_dir =  os.listdir(test_datapath + file)[0]
        
        # load image, resize it and transform it to tensor
        test_img = load_image(test_datapath + file+"/"+im_in_dir)
        tensor = resize_image_test(torch.from_numpy(test_img), resized_width, resized_height)
        
        # Save image on list
        test_imgs.append(tensor)
        ids.append(file[pref_length:len(im_in_dir)-suf_length])
        orig_size.append((test_img.shape[0],test_img.shape[1]))
        
    return ids, test_imgs, orig_size