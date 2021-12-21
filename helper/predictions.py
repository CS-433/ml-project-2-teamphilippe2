from helper.image import *
import numpy as np
from models.features_extraction import *
import torch
    
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
    X = np.array([extract_baseline_features(test_patches[i]) for i in range(len(test_patches))])

    # Predict using given model 
    return baseline_model.predict(X)

def pred_single_img_nnet(nn_model, img, device, threshold):
    dev_img = img.unsqueeze(0).to(device)
    pred = nn_model(dev_img).detach().cpu().numpy()
    
    if pred.shape[1]>1 :
        res = np.argmax(pred, axis=1)
    else :
        pred[pred<=threshold]=0
        pred[pred>threshold]=1
        res = pred
    return res

def predict_test_set_nn(img_tests, nn_model, threshold = 0.5):
    """
    Predit if the pixels of the image using the given neural network
    Parameters:
    -----------
        - img_test:
            All the test set images
        - nn_model:
            The model to use to predict
    Returns:
    -----------
        - All the predictions
    """
    
    device = torch.device("cuda")

    # If a GPU is available, use it
    if not torch.cuda.is_available():
        print("Things will go much quicker with a GPU")
        device = torch.device("cpu")
    
    return [pred_single_img_nnet(nn_model, img, device, threshold) for img in img_tests]
    