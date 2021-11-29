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
    def __init__(self, img_datapath, gt_datapath, ratio_test):
        imgs, gt_imgs = load_images_and_groundtruth(img_datapath, gt_datapath)
        imgs_tr, gt_imgs_tr, imgs_te, gt_imgs_te = split_data(imgs, gt_imgs, ratio_test)
        
        self.test_set = imgs_te, gt_imgs_te
        
        self.all_imgs = []
        self.gt_imgs = []
        
        for img, gt in zip(imgs_tr, gt_imgs_tr):
            img_trans, gt_trans = self.transform(img, gt)
            self.all_imgs.extend(img_trans)
            self.gt_imgs.extend(gt_trans)
            
        self.all_imgs.extend(imgs)
        self.gt_imgs.extend(gt_imgs)
        
        
        self.n_samples = len(self.all_imgs)
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        return self.all_imgs[index], self.gt_imgs[index]
        
    def get_test_set(self):
        return self.test_set
    
    def transform(self, img, gt):
        # Transform to tensor
        img = torch.from_numpy(img)
        img = torch.transpose(img, 0, 2)
        img = torch.transpose(img, 1, 2)
        
        gt = torch.unsqueeze(torch.from_numpy(gt), dim=0).expand(img.shape)
        
        imgs = []
        gt_imgs = []
        
        
        # Generate 10 random crops
        for i in range(10):
            # Random crop of size 200x200 
            i, j, h, w = RandomCrop.get_params(
                img, output_size=(img.shape[1]//2, img.shape[2]//2))
            
            imgs.append(resized_crop(img, i, j, h, w, [img.shape[1], img.shape[2]]))
            gt_imgs.append(resized_crop(gt, i, j, h, w, [img.shape[1], img.shape[2]]))

        # generate different rotations of the same image
        rotation_angles = [90, 180, 270]
        for angle in rotation_angles: 
            imgs.append(rotate(img, angle))
            gt_imgs.append(rotate(gt, angle))
            
        imgs = [img.transpose(1,2).transpose(0,2) for img in imgs]
        gt_imgs = [gt_img[0,:,:] for gt_img in gt_imgs]
        
        return imgs, gt_imgs