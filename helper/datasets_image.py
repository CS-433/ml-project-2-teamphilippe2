from torch.utils.data import Dataset
from torchvision.transforms import *
from torchvision.transforms.functional import *
from helper.loading import *


def to_tensor_and_permute(imgs):
    """
    Convert list of numpy arrays to tensors
    and permute the dimensions to match the
    format used in PyTorch.

    Parameters:
    -----------
        - imgs:
            List of ndarrays
    Returns:
        - imgs_new:
            list of torch tensors
    """
    imgs_new = []
    for img in imgs:
        # Transform to tensor and transform (H,W,3) into (3,H,W)
        img = torch.from_numpy(img)
        img = torch.permute(img, (2, 0, 1))
        imgs_new.append(img)
    return imgs_new


class RoadTestImages(Dataset):
    """
    Custom class for test set, based on a given training set.
    """

    def __init__(self, augmented_ds):
        # Get local test set from training set
        self.test_data, self.test_ground_truth = augmented_ds.get_test_set()

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, index):
        return self.test_data[index], self.test_ground_truth[index]


class AugmentedRoadImages(Dataset):
    """
    Custom class to load the images training set
    """

    def __init__(self, img_datapath, gt_datapath, ratio_train, seed, custom_rot = True):
        """
        Load and split the dataset
        
        Parameters:
        -----------
            - img_datapath:
                the folder containing all the training images
            - gt_datapath:
                The folder containing all the ground-truth train images
            - ratio_train:
                The split ratio of train-test set 
            - seed: 
                Seed to split the dataset
            - custom_rot: 
                If we want to do custom rotation (by an angle which is not a multiple of 90)
        """
        # Load train images
        imgs, gt_imgs = load_images_and_groundtruth(img_datapath, gt_datapath)
        # Split the train images into train and test set
        imgs_tr, gt_imgs_tr, imgs_te, gt_imgs_te = split_data(imgs, gt_imgs, ratio_train, seed=seed)

        # Set the test set and transform the images to tensor and permute ground_truth data.
        self.test_set = to_tensor_and_permute(imgs_te), [self.cap_ground_truth(torch.from_numpy(gt_te)) for gt_te in
                                                         gt_imgs_te]

        # Transform the training images to tensor and permute the axes to obtain (3, H, W)
        imgs_tr = to_tensor_and_permute(imgs_tr)

        self.all_imgs = []
        self.gt_imgs = []

        # Perform data augmentation
        for img, gt in zip(imgs_tr, gt_imgs_tr):
            # Add the different transformations to the dataset
            img_trans, gt_trans = self.transform(img, gt, custom_rot)
            self.all_imgs.extend(img_trans)
            self.gt_imgs.extend(gt_trans)

        self.n_samples = len(self.all_imgs)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.all_imgs[index], self.gt_imgs[index]

    def cap_ground_truth(self, gt_img):
        """
        Clip the values in the groundtruth mask to either 0 or 1.
        Parameters:
        -----------
            - gt_img: image to clip
        Return:
            - Clipped image
        """
        gt_img[gt_img > 0] = 1
        gt_img[gt_img != 1] = 0
        return gt_img

    def get_test_set(self):
        """
        Returns the local test corresponding to our
        training set.

        Return:
            - test images
            - test ground-truths
        """
        return self.test_set[0], self.test_set[1]

    def transform(self, img, gt, custom_rot):
        """
        Return multiple transformations of the same image and teh ground truth
        Parameters:
        -----------
            - img: the image we want to transform
            - gt : the corresponding ground truth image
            -  custom_rot: 
                If we want to do custom rotation (by an angle which is not a multiple of 90)
        Return:
            - List of transformations (tensors) of the image
            - List of transformations (tensors) of the ground truth image
        """

        # Create a 3D tensor from the ground truth image
        gt = torch.unsqueeze(torch.from_numpy(gt), dim=0).expand(img.shape)

        imgs = [img]
        gt_imgs = [gt]
        nb_crops = 10 if custom_rot else 30
        # Generate random crops
        for i in range(nb_crops):
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
        
        if custom_rot:
            # generate cropped rotation of the same image
            rotation_angles = [15, 45, 60, 105, 120, 165, 195, 210, 240, 255, 300, 315]
            crop_center = CenterCrop(200)
            resize = Resize([400,400])
            for angle in rotation_angles:
                imgs.append(resize(crop_center(rotate(img, angle))))
                gt_imgs.append(resize(crop_center(rotate(gt, angle))))

        # Only select the first dimension to transform back the ground truth to a 2D image
        gt_imgs2 = []
        for gt_img in gt_imgs:
            gt = gt_img[0, :, :]
            gt = self.cap_ground_truth(gt)
            gt_imgs2.append(gt[None, :, :])

        return imgs, gt_imgs2

    def compute_pos_weight(self):
        """
        Compute the reweighting factor for the positive class, i.e. return (nb negative samples)/(nb positive samples)
        Returns:
        --------
            The reweighting factors
        """

        cur_sum = 0
        width, height = 0, 0

        for gt in self.gt_imgs:
            cur_sum += gt.sum()
            width, height = gt.shape[1], gt.shape[2]

        total_size = width * height * len(self.gt_imgs) - cur_sum

        return total_size / cur_sum
