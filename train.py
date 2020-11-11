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

import torch.nn as nn
import torch.optim as optim

from models import resnet18
from models import resnet34
from models import resnet50
from models import resnet101
from models import resnet152

from models import HeatmapEncoder
from models import PoseDecoder
from models import HeatmapReconstructer

LOGGER = ConsoleLogger("Main")


def main():
    """Main"""

    LOGGER.info('Starting demo...')


    # -----------------------------------------------------------
    # -----------------------------------------------------------
    # --------------------- Training Phase ----------------------
    # -----------------------------------------------------------
    # -----------------------------------------------------------
    LOGGER.info('Training...')

    # ------------------- Data loader -------------------

    train_data_transform = transforms.Compose([
        trsf.ImageTrsf(),
        trsf.Joints3DTrsf(),
        trsf.ToTensor()])

    # let's load data from validation set as example
    train_data = Mocap(
        config.dataset.train,
        SetType.TRAIN,
        transform=train_data_transform)
    train_data_loader = DataLoader(
        train_data,
        batch_size=config.train_data_loader.batch_size,
        shuffle=config.train_data_loader.shuffle)
    
    # ------------------- Build Model -------------------

    backbone = resnet101()
    encoder = HeatmapEncoder()
    decoder = PoseDecoder()
    reconstructer = HeatmapReconstructer()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        LOGGER.info(("Let's use", torch.cuda.device_count(), "GPUs!"))
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        backbone = nn.DataParallel(backbone)
        encoder = nn.DataParallel(encoder)
        decoder = nn.DataParallel(decoder)
        reconstructer = nn.DataParallel(reconstructer)
    backbone = backbone.cuda()
    encoder = encoder.cuda()
    decoder = decoder.cuda()
    reconstructer = reconstructer.cuda()
    
    # ------------------- Build Loss & Optimizer -------------------

    # Build Loss
    heatmap_prediction_loss_func = nn.MSELoss()
    pose_prediction_cosine_similarity_loss_func = nn.CosineSimilarity()
    pose_prediction_l1_loss_func = nn.L1Loss()
    heatmap_reconstruction_loss_func = nn.MSELoss()

    # Build Optimizer
    optimizer = optim.Adam([
        {"params": backbone.parameters()},
        {"params": encoder.parameters()},
        {"params": decoder.parameters()},
        {"params": reconstructer.parameters()}
    ], lr=0.001)

    # ------------------- Read dataset frames -------------------
    for ep in range(config.train_setting.epoch):
        for it, (img, p2d, p3d, action, heatmap) in enumerate(train_data_loader):
            #################### p2d는 각 Joint별 (x,y) 좌표를 나타낸듯. Image의 좌측상단이 (0,0)이다.
            #################### p3d는 Neck의 좌표를 (0,0,0)으로 생각했을 때의 각 Joint별 (^x,^y,^z) 좌표를 나타낸듯.
            #################### Joint 순서는 config.py에 있다.

            LOGGER.info('Iteration: {}'.format(it))
            LOGGER.info('Images: {}'.format(img.shape))  # (Batch, Channel, Height(y), Width(x))
            LOGGER.info('p2dShapes: {}'.format(p2d.shape))  # (Width, Height)
            # LOGGER.info('p2ds: {}'.format(p2d))
            LOGGER.info('p3dShapes: {}'.format(p3d.shape))  # (^x, ^y, ^z)
            # LOGGER.info('p3ds: {}'.format(p3d))
            LOGGER.info('Actions: {}'.format(action))
            LOGGER.info('heatmapShapes: {}'.format(heatmap.shape))

            # -----------------------------------------------------------
            # ------------------- Run your model here -------------------
            # -----------------------------------------------------------

            optimizer.zero_grad()

            # Move Tensors to GPUs
            img = img.cuda()
            p3d = p3d.cuda()
            heatmap = heatmap.cuda()

            # Forward
            predicted_heatmap = backbone(img)
            latent = encoder(predicted_heatmap)
            predicted_pose = decoder(latent)
            reconstructed_heatmap = reconstructer(latent)

            # Loss Calculation
            heatmap_prediction_loss = heatmap_prediction_loss_func(predicted_heatmap, heatmap)
            p3d_for_loss = torch.cat((p3d[:, 4:6, :], p3d[:, 7:10, :], p3d[:, 11:, :]), dim=1)  # 13까지가 Upper Body
            p3d_for_loss = torch.reshape(p3d_for_loss, (-1, 48))
            pose_prediction_cosine_similarity_loss = pose_prediction_cosine_similarity_loss_func(predicted_pose, p3d_for_loss)
            pose_prediction_cosine_similarity_loss = torch.mean(pose_prediction_cosine_similarity_loss)
            pose_prediction_l1_loss = pose_prediction_l1_loss_func(predicted_pose, p3d_for_loss)
            pose_prediction_loss = -0.01*pose_prediction_cosine_similarity_loss + 0.5*pose_prediction_l1_loss
            heatmap_reconstruction_loss = heatmap_reconstruction_loss_func(reconstructed_heatmap, heatmap)
            # Backpropagating Loss with Weighting Factors
            backbone_loss = heatmap_prediction_loss
            lifting_loss = 0.1*pose_prediction_loss + 0.001*heatmap_reconstruction_loss
            loss = backbone_loss + lifting_loss

            # Backward & Update
            loss.backward()
            optimizer.step()


    # -----------------------------------------------------------
    # -----------------------------------------------------------
    # -------------------- Validation Phase ---------------------
    # -----------------------------------------------------------
    # -----------------------------------------------------------
    LOGGER.info('Validation...')

    # ------------------- Evaluation -------------------

    eval_body = evaluate.EvalBody()
    eval_upper = evaluate.EvalUpperBody()
    eval_lower = evaluate.EvalUpperBody()

    # ------------------- Evaluate -------------------

    # TODO: replace p3d_hat with model preditions
    p3d_hat = torch.ones_like(p3d)

    # Evaluate results using different evaluation metrices
    y_output = p3d_hat.data.cpu().numpy()
    y_target = p3d.data.cpu().numpy()

    eval_body.eval(y_output, y_target, action)
    eval_upper.eval(y_output, y_target, action)
    eval_lower.eval(y_output, y_target, action)


    # -----------------------------------------------------------
    # -----------------------------------------------------------
    # ----------------------- Save Phase ------------------------
    # -----------------------------------------------------------
    # -----------------------------------------------------------
    LOGGER.info('Save...')

    # ------------------- Save results -------------------

    LOGGER.info('Saving evaluation results...')
    res = {'FullBody': eval_body.get_results(),
           'UpperBody': eval_upper.get_results(),
           'LowerBody': eval_lower.get_results()}

    io.write_json(config.eval.output_file, res)

    LOGGER.info('Done.')


if __name__ == "__main__":
    main()
