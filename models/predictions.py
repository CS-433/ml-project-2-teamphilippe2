from helper.image import *
import numpy as np
from models.features_extraction import *
    
def predict_test_set(test_set, baseline_model, patch_size):
    """
    Predict if patches in test set images corresponds to a road or not
    Parameters:
    -----------
        - test_set: 
            All test images
        - baseline_model:
            The fitted model to use to make the predictions 
        - patch_size:
            The patch size used to crop the test images
    Returns:
    -----------
        List of predictions for the test set
    """
    # Split and linearise patches 
    test_patches = [img_crop(test_set[i], patch_size, patch_size) for i in range(len(test_set))]
    test_patches = np.array([test_patches[i][j] for i in range(len(test_patches)) for j in range(len(test_patches[i]))])   
    #extract features from patches
    X = np.array([extract_features_2d(test_patches[i]) for i in range(len(test_patches))])

    # Predict using given model 
    return baseline_model.predict(X)
