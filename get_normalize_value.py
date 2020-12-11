# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -*- coding: utf-8 -*-
"""
Demo code
@author: Denis Tome'
"""
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
from base import SetType
import dataset.transform as trsf
from dataset import Mocap
from utils import config_get_normalize_value, ConsoleLogger
from utils import evaluate, io
import math
from tqdm import tqdm


# class AverageMeter(object):
#     """
#     Computes and stores the average and current value
#     Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
#     """
#     def __init__(self):
#         self.reset()

#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0

#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count


LOGGER = ConsoleLogger("Main")


def main():
    """Main"""

    LOGGER.info('Starting demo...')

    # ------------------- Data loader -------------------

    data_transform = transforms.Compose([
        trsf.ImageTrsf(),
        trsf.Joints3DTrsf(),
        trsf.ToTensor()])

    # let's load data from validation set as example
    data = Mocap(
        config_get_normalize_value.dataset.train,
        SetType.TRAIN,
        transform=data_transform)
    data_loader = DataLoader(
        data,
        batch_size=config_get_normalize_value.data_loader.batch_size,
        shuffle=config_get_normalize_value.data_loader.shuffle)

    # ------------------- Evaluation -------------------

    # eval_body = evaluate.EvalBody()
    # eval_upper = evaluate.EvalUpperBody()
    # eval_lower = evaluate.EvalUpperBody()

    # ------------------- AverageMeter -------------------
    meanAverageMeter = [0]*66
    stdAverageMeter = [0]*66
    # for i in range(48):
    #     meanAverageMeter[i] = AverageMeter()
    #     stdAverageMeter[i] = AverageMeter()

    # ------------------- Read dataset frames -------------------
    for it, (img, p2d, p3d, action, heatmap) in tqdm(enumerate(data_loader), total=len(data_loader)):
        for i in range(66):
            meanAverageMeter[i] += p3d[0][int(i/3)][int(i%3)]
    for i in range(66):
        meanAverageMeter[i] /= len(data_loader)

    for it, (img, p2d, p3d, action, heatmap) in tqdm(enumerate(data_loader), total=len(data_loader)):
        for i in range(66):
            stdAverageMeter[i] += (p3d[0][int(i/3)][int(i%3)] - meanAverageMeter[i]) ** 2
    for i in range(66):
        stdAverageMeter[i] /= len(data_loader)
        stdAverageMeter[i] = math.sqrt(stdAverageMeter[i])
    
    print("meanAverageMeter: ")
    print(meanAverageMeter)
    print("stdAverageMeter: ")
    print(stdAverageMeter)


    # ------------------- Save results -------------------

    # LOGGER.info('Saving evaluation results...')
    # res = {'FullBody': eval_body.get_results(),
    #        'UpperBody': eval_upper.get_results(),
    #        'LowerBody': eval_lower.get_results()}

    # io.write_json(config_get_normalize_value.eval.output_file, res)

    # LOGGER.info('Done.')


if __name__ == "__main__":
    main()