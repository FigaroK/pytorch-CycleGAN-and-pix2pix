"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
from data.base_dataset import BaseDataset, get_transform
from glob import glob
import numpy as np
import h5py
import cv2
import random
# from data.image_folder import make_dataset
# from PIL import Image

def torch2cv2(x):
    if x.ndim == 3:
        return x
    gray_list = []
    for i in x:
        if i.shape[2] == 3:
            rgb=i
        else:
            rgb = np.transpose(i, [1,2,0])
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray_list.append(gray[np.newaxis, np.newaxis, :])
    gray_list = np.vstack(gray_list)
    a = np.squeeze(gray_list, axis=1)
    return a

def eyediap_toGray(x):
    if x.ndim == 3 and x.shape[2]==3:
        x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    elif x.ndim ==3 and x.shape[0] == 3:
        x = np.transpose(x, [1,2,0])
        x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    x = x.astype(np.float32)
    x = (x / 255 - 0.5) / 0.5
    return x[np.newaxis, :]

class gazesingleDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--lambda_feature', type=float, default=0.1, help='# of gen filters in the last conv layer')
        parser.add_argument('--lambda_perceptual', type=float, default=0.1, help='# of gen filters in the last conv layer')
        parser.add_argument('--path_B', type=str, default="/4Tdisk/fjl/dataset/Unity_dis_from_MPIIGaze_H5/P00_partial.h5", help='# of gen filters in the last conv layer')
        parser.add_argument('--path_A', type=str, default="/4Tdisk/fjl/dataset/MPII_H5_single_evaluation/P*.h5", help='# of gen filters in the last conv layer')
        parser.add_argument('--path_extractor', type=str, default="/4Tdisk/fjl/checkpoint/gaze/cross_single_eye/all/resnet50_extractor_best.tar", help='# of gen filters in the last conv layer')
        parser.add_argument('--path_following', type=str, default="/4Tdisk/fjl/checkpoint/gaze/cross_single_eye/all/resnet50_following_best.tar", help='# of gen filters in the last conv layer')
        # parser.set_defaults(max_dataset_size=10, new_dataset_option=2.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        # get the image paths of your dataset;
        self.image_paths_A = glob(opt.path_A)  # You can call sorted(make_dataset(self.root, opt.max_dataset_size)) to get all the image paths under the directory self.root
        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        self.transform = eyediap_toGray
        self.A_files = [h5py.File(file_name, 'r') for file_name in self.image_paths_A]
        self.A = _eyediap_get_mat(self.A_files)
        self.A_size = self.A[-1]

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        data_A = self.A[0][index].astype(np.float32)
        data_A = self.transform(data_A)
        data_A_label = self.A[2][index].astype(np.float32)
        data_A_pose = self.A[4][index].astype(np.float32)
        return {'A': data_A, 'label_A':data_A_label, 'pose_A':data_A_pose, 'A_paths':f"{index}.png"}

    def __len__(self):
        """Return the total number of images."""
        a = self.A_size
        return a

def _eyediap_get_mat(files, single=True, flip_right=False):
    left_gazes = np.vstack([files[idx]['left_gaze'] for idx in range(len(files))])
    left_headposes = np.vstack([files[idx]['left_headpose'] for idx in range(len(files))])
    images_l = np.vstack([torch2cv2(files[idx]['left_eye_img']) for idx in range(len(files))])
    num_instances = images_l.shape[0]
    # dimension = images_l.shape[1]
    if 'right_gaze' in files[0] and flip_right:
        right_gazes = np.vstack([files[idx]['right_gaze'] for idx in range(len(files))])
        right_headposes = np.vstack([files[idx]['right_headpose'] for idx in range(len(files))])
        images_r = np.vstack([torch2cv2(files[idx]['right_eye_img']) for idx in range(len(files))])


    if single:
        # logger.debug("shape of 'images' is %s" % (images_l.shape,))
        if 'right_gaze' in files[0] and flip_right:
            num_instances *= 2
            # logger.debug("%d images loaded" % (num_instances))
            right_gazes[:,1] = -right_gazes[:,1]
            right_headposes[:,1] = -right_headposes[:,1]
            images_l = np.vstack([torch2cv2(images_l), torch2cv2(np.flip(images_r, 2 if images_r.ndim == 4 and images_r.shape[-1] == 3 else -1))]) 
            left_gazes= np.vstack([left_gazes, right_gazes])
            left_headposes= np.vstack([left_headposes, right_headposes])
        else:
            pass
        return images_l, None, left_gazes, None, left_headposes, None, num_instances

    # logger.debug("%s images loaded" % (num_instances))
    # logger.debug("shape of 'images' is %s" % (images_l.shape,))
    return images_l, images_r, left_gazes, right_gazes, left_headposes, right_headposes, num_instances
