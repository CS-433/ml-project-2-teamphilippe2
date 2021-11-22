from helper.image import *
import numpy as np
from models.features_extraction import *
    
def predict_test_set(test_set, baseline_model, patch_size):
    """
    Predict if patches in test set images corresponds to a road or not
    """
    # Split and linearise patches 
    test_patches = [img_crop(test_set[i], patch_size, patch_size) for i in range(len(test_set))]
    test_patches = np.array([test_patches[i][j] for i in range(len(test_patches)) for j in range(len(test_patches[i]))])   
    #extract features from patches
    X = np.array([extract_features_2d(test_patches[i]) for i in range(len(test_patches))])

    # Predict using given model 
    return baseline_model.predict(X)


def value_to_class(v, threshold=0.25):
    # Compute features for each image patch in groundtruth
    # percentage of pixels > 1 required to assign a foreground label to a patch
    df = np.sum(v)
    if df > threshold:
        return 1
    else:
        return 0

