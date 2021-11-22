from helper.image import *
import matplotlib.pyplot as plt
from PIL import Image

def display_imgs_side_by_side(img, gt_img):
    """
    Display two images side by side 
    """
    cimg = concatenate_images(img, gt_img)
    fig1 = plt.figure(figsize=(10, 10))
    plt.imshow(cimg, cmap='Greys_r')
    return 

def make_img_overlay(img, predicted_img):
    """
    Display prediction on top of base image
    """
    fig1 = plt.figure(figsize=(10, 10))
    
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:,:,0] = predicted_img*255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    plt.imshow(new_img)
    return new_img