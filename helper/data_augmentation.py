import torch
from torch.utils.data import Dataset
from torchvision.transforms import *
from torchvision.transforms.functional import *

from helper.image import get_img_patches
from helper.loading import *


class RoadTestImages(Dataset):
    def __init__(self, augmented_ds):
        self.test_data, self.test_ground_truth = augmented_ds.get_test_set()

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, index):
        return self.test_data[index], self.test_ground_truth[index]


def to_tensor_and_permute(imgs):
    imgs_new = []
    for img in imgs:
        # Transform to tensor and transform (H,W,3) into (3,H,W)
        img = torch.from_numpy(img)
        img = torch.permute(img, (2, 0, 1))
        imgs_new.append(img)
    return imgs_new


class AugmentedRoadImages(Dataset):
    """
    Custom class to load our training dataset
    """

    def __init__(self, img_datapath, gt_datapath, ratio_train, seed):
        """
        Load and split the dataset
        
        Parameters:
        -----------
            - img_datapath:
                the folder containing all the training images
            - gt_datapath:
                The folder containing all the groundtruth train images
            - ratio_test: 
                The split ratio of train-test set 
            - seed: 
                Seed to split the dataset 
        """
        # Load train images
        imgs, gt_imgs = load_images_and_groundtruth(img_datapath, gt_datapath)
        # Split the train images into train and test set
        imgs_tr, gt_imgs_tr, imgs_te, gt_imgs_te = split_data(imgs, gt_imgs, ratio_train, seed=seed)

        # Set the test set and transform the images to tensor and permute ground_truth data.
        self.test_set = to_tensor_and_permute(imgs_te), [self.cap_ground_truth(torch.from_numpy(gt_te)) for gt_te in gt_imgs_te]

        # Transform the training images to tensor and permute the axes to obtain (3, H, W)
        imgs_tr = to_tensor_and_permute(imgs_tr)

        self.all_imgs = []
        self.gt_imgs = []

        # Perform data augmentation
        for img, gt in zip(imgs_tr, gt_imgs_tr):
            # Add the different transformations to the dataset
            img_trans, gt_trans = self.transform(img, gt)
            self.all_imgs.extend(img_trans)
            self.gt_imgs.extend(gt_trans)

        self.n_samples = len(self.all_imgs)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.all_imgs[index], self.gt_imgs[index]

    def cap_ground_truth(self, gt_img):
        gt_img[gt_img > 0] = 1
        gt_img[gt_img != 1] = 0
        return gt_img

    def get_test_set(self):
        """
        Return the test set corresponding to our dataset
        Returns:
            The test set
        """
        return self.test_set[0], self.test_set[1]

    def transform(self, img, gt):
        """
        Return multiple transformations of the same image and teh ground truth
        Parameters:
        -----------
            - img: the image we want to transform
            - gt : the corresponding ground truth image
        Return:
            - List of transformations (tensors) of the image
            - List of transformations (tensors) of the ground truth image
        """

        # Create a 3D tensor from the ground truth image
        gt = torch.unsqueeze(torch.from_numpy(gt), dim=0).expand(img.shape)

        imgs = [img]
        gt_imgs = [gt]

        # Generate 10 random crops
        for i in range(30):
            # Get random crop params of size 200x200 
            i, j, h, w = RandomCrop.get_params(
                img, output_size=(img.shape[1] // 2, img.shape[2] // 2))

            # Crop and store the different crops
            imgs.append(resized_crop(img, i, j, h, w, [img.shape[1], img.shape[2]]))
            gt_imgs.append(resized_crop(gt, i, j, h, w, [img.shape[1], img.shape[2]]))

        # generate different rotations of the same image
        rotation_angles = [90, 180, 270]
        for angle in rotation_angles:
            imgs.append(rotate(img, angle))
            gt_imgs.append(rotate(gt, angle))

        # Only select the first dimension to transform back the ground truth to a 2D image
        gt_imgs2 = []
        for gt_img in gt_imgs:
            gt = gt_img[0, :, :]
            gt = self.cap_ground_truth(gt)
            gt_imgs2.append(gt[None, :, :])

        return imgs, gt_imgs2


class OriginalTrainingRoadPatches(Dataset):
    """
    Original, non augmented training patches.
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


class OriginalTestRoadPatches(Dataset):
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
