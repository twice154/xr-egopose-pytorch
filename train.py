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
from utils import config, ConsoleLogger
from utils import evaluate, io

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
        config.dataset.train,
        SetType.TRAIN,
        transform=data_transform)
    data_loader = DataLoader(
        data,
        batch_size=config.data_loader.batch_size,
        shuffle=config.data_loader.shuffle)

    # ------------------- Evaluation -------------------

    eval_body = evaluate.EvalBody()
    eval_upper = evaluate.EvalUpperBody()
    eval_lower = evaluate.EvalUpperBody()

    # ------------------- Read dataset frames -------------------
    for it, (img, p2d, p3d, action) in enumerate(data_loader):
        #################### p2d는 각 Joint별 (x,y) 좌표를 나타낸듯. Image의 좌측상단이 (0,0)이다.
        #################### p3d는 Neck의 좌표를 (0,0,0)으로 생각했을 때의 각 Joint별 (^x,^y,^z) 좌표를 나타낸듯.
        #################### Joint 순서는 config.py에 있다.
        #################### 정확한 것은 나중에 Visualize 해야 확실하게 판단할 수 있을듯.

        LOGGER.info('Iteration: {}'.format(it))
        LOGGER.info('Images: {}'.format(img.shape))
        LOGGER.info('p2dShapes: {}'.format(p2d.shape))
        LOGGER.info('p2ds: {}'.format(p2d))
        LOGGER.info('p3dShapes: {}'.format(p3d.shape))
        LOGGER.info('p3ds: {}'.format(p3d))
        LOGGER.info('Actions: {}'.format(action))

        # -----------------------------------------------------------
        # ------------------- Run your model here -------------------
        # -----------------------------------------------------------

        # TODO: replace p3d_hat with model preditions
        p3d_hat = torch.ones_like(p3d)

        # Evaluate results using different evaluation metrices
        y_output = p3d_hat.data.cpu().numpy()
        y_target = p3d.data.cpu().numpy()

        eval_body.eval(y_output, y_target, action)
        eval_upper.eval(y_output, y_target, action)
        eval_lower.eval(y_output, y_target, action)

        # TODO: remove break
        break

    # ------------------- Save results -------------------

    LOGGER.info('Saving evaluation results...')
    res = {'FullBody': eval_body.get_results(),
           'UpperBody': eval_upper.get_results(),
           'LowerBody': eval_lower.get_results()}

    io.write_json(config.eval.output_file, res)

    LOGGER.info('Done.')


if __name__ == "__main__":
    main()