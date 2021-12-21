from torch.utils.data import Dataset
from helper.datasets_image import to_tensor_and_permute
from helper.image import get_img_patches, get_imgs_gt_patches
from helper.loading import *
from models.features_extraction import build_gt_from_patches, value_to_class


class AutoencoderTrainingRoadPatches(Dataset):
    """
    Original, non augmented training patches.
    This dataset returns the same patch as input and output.
    """
    def __init__(self, img_datapath, patch_size=16):
        # Load all training images in folder
        imgs = load_all_images_in_folder(img_datapath)

        # Get all the patches from the images
        # np array (nb patches, patch_size, patch_size, 3)
        patches = get_img_patches(imgs, patch_size=patch_size)

        # Convert to correct Tensor format
        self.patches = to_tensor_and_permute(patches)

        self.n_samples = len(self.patches)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, item):
        # For autoencoder, target is the original image
        return self.patches[item], self.patches[item]


class AutoencoderTestRoadPatches(Dataset):
    """
    Original, non augmented test patches. The test set is the
    set for the AICrowd submissions as supervision is not needed
    for the autoencoder.
    This dataset returns the same patch as input and output.
    """
    def __init__(self, img_datapath, patch_size=16):
        # Load all test images in folder
        # Keep original size
        ids, imgs, _ = load_test_set(img_datapath, test_img_width, test_img_height)

        # imgs is a list of (C, H, W), need numpy (W, H, C)
        # to extract patches
        imgs = [torch.permute(img, (2, 1, 0)).numpy() for img in imgs]

        # Get all the patches from the images
        # np array (nb patches, patch_size, patch_size, 3)
        patches = get_img_patches(imgs, patch_size=patch_size)

        # Convert to correct Tensor format
        self.patches = to_tensor_and_permute(patches)

        self.n_samples = len(self.patches)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, item):
        # For autoencoder, target is the original image
        return self.patches[item], self.patches[item]


class ConvNetTrainingRoadPatches(Dataset):
    """
    Original, non augmented training patches.
    This dataset returns patches as input and label as output
    """
    def __init__(self, image_dir, gt_dir, patch_size=16, seed=0, ratio_train=0.8):
        # Load all training images and gt in folder
        imgs, gt_imgs = load_images_and_groundtruth(image_dir, gt_dir)

        # Get all the patches from the images
        # np array (nb patches, patch_size, patch_size, 3)
        patches, gt_patches = get_imgs_gt_patches(imgs, gt_imgs)

        # Labels for each patch
        y = build_gt_from_patches(gt_patches,
                                  lambda gt_patch: value_to_class(gt_patch, threshold=0.5))

        # Split data into training and validation set
        self.patches_train, self.y_train, self.patches_test, self.y_test = split_data(patches, y, ratio_train,
                                                                                      seed=seed)

        # Convert to correct Tensor format
        self.patches_train = to_tensor_and_permute(self.patches_train)
        self.patches_test = to_tensor_and_permute(self.patches_test)

        self.y_train = torch.unsqueeze(torch.from_numpy(self.y_train).float(), 1)
        self.y_test = torch.unsqueeze(torch.from_numpy(self.y_test).float(), 1)

        self.n_samples = len(self.patches_train)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, item):
        return self.patches_train[item], self.y_train[item]

    def get_test_set(self):
        """
        Returns the local test corresponding to our
        training set.

        Return:
            - test images
            - test ground-truths
        """
        return self.patches_test, self.y_test

    def compute_pos_weights(self):
        """
        Compute the weights for the positive samples
        to use in the loss.
        Returns:
            Tensor with the positive weight
        """
        # Ratio of nb of negative samples / nb of positive samples
        nb_pos = self.y_train.sum()
        nb_neg = self.y_train.shape[0] - nb_pos
        return torch.Tensor([nb_neg / nb_pos])


class ConvNetTestRoadPatches(Dataset):
    """
    Original, non augmented local test patches.
    """
    def __init__(self, training_ds):
        self.test_data, self.test_ground_truth = training_ds.get_test_set()
        self.len = len(self.test_data)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.test_data[item], self.test_ground_truth[item]
