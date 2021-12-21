# Folder path
root_dir = "data/"
training_dir = root_dir +"training/"
image_dir = training_dir + "images/"
gt_dir = training_dir + "groundtruth/"
output_folder = "output/"
weights_folder = output_folder+ "weights/"
features_folder = output_folder + "features/"
submission_folder = output_folder + "submissions/"


# extension weight models
ext_weight_model = ".pth"

# Path to the best model weights
best_model_weight_path = weights_folder + "nnet/2021-12-21_05-49-07"+ext_weight_model

# Testing images related constants
test_dir = root_dir + "test_set_images/"
test_image_prefix = "test_"
test_image_suffix = ".jpg"

# Size of images used as input to the neural network
train_img_height = 400
train_img_width = 400
test_img_width = 608
test_img_height = 608