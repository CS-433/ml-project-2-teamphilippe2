import matplotlib.image as mpimg
import os 
import numpy as np

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
    