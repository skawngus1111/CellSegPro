import os
import sys

from torch.utils.data import Dataset

import cv2
import numpy as np
import pandas as pd
from PIL import Image

class DataBowl2018Dataset(Dataset) :
    def __init__(self, dataset_dir, frame, transform=None, target_transform=None):
        super(DataBowl2018Dataset, self).__init__()
        self.dataset_dir = dataset_dir
        self.image_folder = 'stage1_train'
        self.transform = transform
        self.target_transform = target_transform
        self.frame = frame

    def __len__(self):
        return len(self.frame.ImageId.unique())

    def _get_image(self, idx):# iloc (int): The position in the dataframe
        # Collect image name from csv frame
        img_name = self.frame.ImageId.unique()[idx]
        img_path = os.path.join(
            self.dataset_dir, self.image_folder, img_name, "images", img_name + '.png')

        # Load image
        image = Image.open(img_path).convert('RGB')

        return image

    def _load_mask(self, idx): # iloc (int): index in the data frame
        # Collect image name from csv frame
        img_name = self.frame.ImageId.unique()[idx]
        mask_dir = os.path.join(self.dataset_dir, self.image_folder, img_name, "masks")
        mask_paths = [os.path.join(mask_dir, fp) for fp in os.listdir(mask_dir)]
        mask = None
        for fp in mask_paths:
            img = cv2.imread(fp, 0)
            if img is None:
                raise FileNotFoundError("Could not open %s" % fp)
            if mask is None:
                mask = img
            else:
                mask = np.maximum(mask, img)

        mask = Image.fromarray(mask)

        return mask

    def __getitem__(self, idx):
        image = self._get_image(idx)
        mask = self._load_mask(idx)

        # invert if too bright
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, a = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
        b = np.count_nonzero(a)
        ttl = np.prod(a.shape)
        if b > ttl / 2:
            image = Image.fromarray(cv2.bitwise_not(img))

        if self.transform:
            image = self.transform(image)
            mask = self.target_transform(mask)

        number = len(self.frame[self.frame.ImageId == self.frame.ImageId.unique()[idx]])

        return image, mask