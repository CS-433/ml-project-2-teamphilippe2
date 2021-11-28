import numpy as np
from helper.image import *

def extract_features_from_patches(patches, extract_fct):
    """
        Extracts features from all the given patches
        according to the given extraction function.
        
        Parameters
        ----------
            patches :
                List of patches
            extract_fct :
                function taking a patch as argument
                and return a vector of features for
                this patch
            
        Returns 
        -------
            np.array of shape (nb patches, nb features)
    """
    return np.asarray([extract_fct(patch) for patch in patches])
    
def build_gt_from_patches(gt_patches, conversion_fct):
    """
        Builds the ground truths vector from the list of the
        ground truths patches, according to the given
        conversion function.
        
        Parameters
        ----------
            gt_patches :
                List of ground truth patches
            conversion_fct :
                function taking a gt patch as argument
                and return either class 1 or 0 for
                this gt patch
            
        Returns 
        -------
            np.array of shape (nb patches,)
    """
    return np.asarray([conversion_fct(gt_patch) for gt_patch in gt_patches])
    
def value_to_class(gt_patch, threshold=0.25):
    # Compute features for each image patch in groundtruth
    # percentage of pixels > 1 required to assign a foreground label to a patch
    df = np.mean(gt_patch)
    if df > threshold:
        return 1
    else:
        return 0

def extract_baseline_features(img):
    """
        Extracts features from a given image (patch)
        
        Parameters
        ----------
            img :
                Image/patch to extract the features from
            
        Returns 
        -------
            np.array : contains the features
            of this image/patch
    """
    # Extract the mean of each channel
    mean_r = np.mean(img[:, :, 0])
    mean_g = np.mean(img[:, :, 1])
    mean_b = np.mean(img[:, :, 2])
    
    # Extract the var of each channel
    var_r = np.var(img[:, :, 0])
    var_g = np.var(img[:, :, 1])
    var_b = np.var(img[:, :, 2])
    
    # Extract the max of each channel
    max_r = np.max(img[:, :, 0])
    max_g = np.max(img[:, :, 1])
    max_b = np.max(img[:, :, 2])
    
    return np.array([mean_r, mean_g, mean_b, var_r, var_g, var_b, max_r, max_g, max_b])

def standardize_features(X, means=None, stds=None):
    """
        Standardize the features.
        
        Parameters
        ----------
            X :
                Data to standardize
            
        Returns 
        -------
            np.array : same shape as input
            means : means of each features
            stds : stds of each features
    """
    if means is None or stds is None:
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0) 
        
    
    return (X - means) / stds, means, stds
