import torch

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from torchvision import transforms
from torchvision import datasets

from torchvision.transforms import *
from torchvision.transforms.functional import * 

import matplotlib.pyplot as plt
import random

from helper.loading import *
from helper.visualisations import *

       
class AugmentedRoadImages(Dataset):
    """
    Custom class to load our training dataset
    """
    def __init__(self, img_datapath, gt_datapath, ratio_test, seed):
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
        imgs_tr, gt_imgs_tr, imgs_te, gt_imgs_te = split_data(imgs, gt_imgs, ratio_test,seed=seed)
        
        self.test_set = imgs_te, gt_imgs_te
        
        self.all_imgs = []
        self.gt_imgs = []
        
        # Perform data augmentation
        for img, gt in zip(imgs_tr, gt_imgs_tr):
            # Add the different transformations to the dataset
            img_trans, gt_trans = self.transform(img, gt)
            self.all_imgs.extend(img_trans)
            self.gt_imgs.extend(gt_trans)
            
        self.all_imgs.extend([torch.from_numpy(img) for img in imgs])
        self.gt_imgs.extend([torch.from_numpy(gt) for gt in gt_imgs])
        self.n_samples = len(self.all_imgs)
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        return self.all_imgs[index], self.gt_imgs[index]
        
    def get_test_set(self):
        return self.test_set
    
    def transform(self, img, gt):
        # Transform to tensor and transform (W,H,3) into (3,W,H)
        img = torch.from_numpy(img)
        img = torch.transpose(img, 0, 2)
        img = torch.transpose(img, 1, 2)
        # Create a 3D tesnsor from the groundtruth image
        gt = torch.unsqueeze(torch.from_numpy(gt), dim=0).expand(img.shape)
        
        imgs = []
        gt_imgs = []
        
        # Generate 10 random crops
        for i in range(10):
            # Get random crop params of size 200x200 
            i, j, h, w = RandomCrop.get_params(
                img, output_size=(img.shape[1]//2, img.shape[2]//2))
        
            # Crop and store the different crops
            imgs.append(resized_crop(img, i, j, h, w, [img.shape[1], img.shape[2]]))
            gt_imgs.append(resized_crop(gt, i, j, h, w, [img.shape[1], img.shape[2]]))

        # generate different rotations of the same image
        rotation_angles = [90, 180, 270]
        for angle in rotation_angles: 
            imgs.append(rotate(img, angle))
            gt_imgs.append(rotate(gt, angle))
        
        # Retransform the image from (3,W,H) to (W,H,3)
        imgs = [img.transpose(1,2).transpose(0,2) for img in imgs]
        # Only select the first dimension to transform back the groundtruth to a 2D image
        gt_imgs = [gt_img[0,:,:] for gt_img in gt_imgs]
        
        return imgs, gt_imgs