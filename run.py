import torch
from datetime import datetime

from helper.loading import *
from helper.submission import *
from helper.const import *
from models.predictions import *
#from models.unet import *

def main():    
    print("\n==> Loading test set...\n")
    ids, test_set, orig_size = load_test_set(test_dir, train_img_width, train_img_height)
    
    print('\n==> Loading model...\n')
    # Load the model
    model = torch.load(best_model_weight_path)
    model.eval()
    
    print('\n==> Predicting labels for the test set...\n')
    # Make predictions for the test set
    y_test_pred = predict_test_set_nn(test_set, model)
    # Resize image to original size
    resize_pred = [resize_image_test(pred, width, height) for pred, (height, width) in zip(y_test_pred, orig_size)]
    
    print('==> Creating submission file...\n')
    # Save the submission file
    now = datetime.now()
    predictions_to_submission(submission_folder+'submission_'+now.strftime("%Y-%m-%d_%H-%M-%S")+'.csv', ids, resize_pred)
    
    print('==> Submission files saved.')

if __name__ == '__main__':
    main()