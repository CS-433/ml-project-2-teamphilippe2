import numpy as np
import matplotlib.pyplot as plt

def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

# Concatenate an image and its groundtruth
def concatenate_images(img, gt_img):
    """
    Concatenate two images side by side
    """
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def img_crop(im, w, h):
    """
    Divide the images into smaller patches
    """
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches


def get_img_patches(imgs, patch_size=16):
    """
    Get patches from all images
    """
    img_width, img_height = imgs[0].shape[0], imgs[0].shape[1]
    n = len(imgs)
    
    if(img_width % patch_size==0 and img_height % patch_size==0):
        img_patches = [img_crop(imgs[i], patch_size, patch_size) for i in range(n)]
        # Linearize list of patches
        img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
        return img_patches
    else: 
        print(f"Unable to get image patches as the required patch size is not a divider of {img_width}")
        return
    
def get_imgs_gt_patches(imgs, gt_imgs,patch_size = 16):
    """
    Get groundtruth and base images patches
    """
    return get_img_patches(imgs, patch_size), get_img_patches(gt_imgs, patch_size)

# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, w, h, labels):
    """
    Convert a list of labels to an image
    """
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            im[j:j+w, i:i+h] = labels[idx]
            idx = idx + 1
    return im