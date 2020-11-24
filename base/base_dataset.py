# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -*- coding: utf-8 -*-
"""
Base class for datasets.

This code assumes that the dataset structure is the one
provided in the README.md file.

@author: Denis Tome'

"""
import os
import enum
from torchvision import transforms
from torch.utils.data import Dataset
from base import BaseTransform
from utils import ConsoleLogger, io


# import cv2
# import sys
# import os
# import torch
import numpy as np
# import torch.utils.data
# import utils.img

class GenerateHeatmap():  # Generate Probability Field -> 0~1
    def __init__(self, output_res, num_parts):
        self.output_res = output_res
        self.num_parts = num_parts
        sigma = (self.output_res-400)/64*2
        self.sigma = sigma
        size = 6*sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, keypoints):
        hms = np.zeros(shape = (self.num_parts, self.output_res, self.output_res), dtype = np.float32)
        sigma = self.sigma

        # for p in keypoints:
        for idx, pt in enumerate(keypoints):
            pt /= 16  # Heatmap Scaling
            pt += 200  # Heatmap Margin
            pt = [int(pt[0]), int(pt[1])]  # Pixel Location is Integer
            if pt[0] > 0: 
                x, y = int(pt[0]), int(pt[1])
                if x<0 or y<0 or x>=self.output_res or y>=self.output_res:
                    continue
                ul = int(x - 3*sigma - 1), int(y - 3*sigma - 1)
                br = int(x + 3*sigma + 2), int(y + 3*sigma + 2)

                c,d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                a,b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                cc,dd = max(0, ul[0]), min(br[0], self.output_res)
                aa,bb = max(0, ul[1]), min(br[1], self.output_res)
                hms[idx, aa:bb,cc:dd] = np.maximum(hms[idx, aa:bb,cc:dd], self.g[a:b,c:d])
        hms = hms[:, 0+200:48+200, 0+200:48+200]
        return hms


class SetType(enum.Enum):
    """Set types"""

    TRAIN = 'train_set'
    VAL = 'val_set'
    TEST = 'test_set'


class BaseDataset(Dataset):
    """
    Base class for all datasets
    """

    def __init__(self, db_path, set_type, transform=None):
        """Init class

        Arguments:
            db_path {str} -- path to set
            set_type {SetType} -- set

        Keyword Arguments:
            transform {BaseTransform} -- transformation to apply to data (default: {None})
        """

        assert isinstance(set_type, SetType)
        self.logger = ConsoleLogger(set_type.value)

        if io.exists(db_path):
            self.path = db_path
        else:
            self.logger.error('Dataset directory {} not found'.format(db_path))

        self.index = self._load_index()

        if transform:
            assert isinstance(transform, (BaseTransform, transforms.Compose))
        self.transform = transform

        #################### Image Size와 # of Keypoints 수동으로 그냥 설정해줌
        self.generateHeatmap = GenerateHeatmap(48+400, 15)  # Out of Range도 커버하기 위해서 상하좌우 200씩 크게 잡아서 만든 후에, Crop하는 방식으로 접근함.
        ####################

    def _load_index(self):
        """Get indexed set. If the set has already been
        indexed, load the file, otherwise index it and save cache.

        Returns:
            dict -- index set
        """

        idx_path = os.path.join(self.path, 'index.h5')
        if io.exists(idx_path):
            return io.read_h5(idx_path)

        index = self.index_db()
        io.write_h5(idx_path, index)
        return index

    def index_db(self):
        """Index data for faster execution"""

        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
