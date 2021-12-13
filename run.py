import torch
from datetime import datetime

from  torchvision.transforms.functional import resize

from helper.loading import *
from helper.submission import *
from helper.const import *
from models.predictions import *
from models.NNET import *
from models.UNet_orig import *

def load_model(weight_path, model):
    """
    Load the model from the given weights
    Parameters: 
    -----------
        - weight_path: The path to the weight we need to load
        - model : the model object we need to load the weights into
    """
    print("Used weights "+weight_path)
    # Restore the state of the model
    model.load_state_dict(torch.load(weight_path))
    
    device = torch.device("cuda")
    # If a GPU is available, use it
    if not torch.cuda.is_available():
        print("Things will go much quicker with a GPU")
        device = torch.device("cpu")
    
    # Transfer the model to the either the GPU or the CPU
    model.to(device)
    # Put model in evaluation mode 
    model.eval()

def main():    
    print("\n==> Loading test set...\n")
    ids, test_set, orig_size = load_test_set(test_dir, train_img_width, train_img_height)
    
    print('\n==> Loading model...\n')
    # Load the model
    #model = UNet(3,32)
    model = NNet()
    load_model(best_model_weight_path, model)
    
    print('\n==> Predicting labels for the test set...\n')
    # Make predictions for the test set
    y_test_pred = predict_test_set_nn(test_set, model)
    
    # Resize image to original size
    resize_pred = [resize(torch.from_numpy(pred), [height, width]).squeeze(0).squeeze(0) for pred, (height, width) in zip(y_test_pred, orig_size)]
    
    print('==> Creating submission file...\n')
    # Save the submission file
    now = datetime.now()
    predictions_to_submission(submission_folder+'submission_'+now.strftime("%Y-%m-%d_%H-%M-%S")+'.csv', ids, resize_pred)
    
    print('==> Submission files saved.')

if __name__ == '__main__':
    main()