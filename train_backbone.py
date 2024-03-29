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
from utils import config_backbone, ConsoleLogger
from utils import evaluate, io

import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from shutil import copyfile

from models import resnet18
from models import resnet34
from models import resnet50
from models import resnet101
from models import resnet152

from models import HeatmapEncoder
from models import PoseDecoder
from models import HeatmapReconstructer

LOGGER = ConsoleLogger("Main")


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is None:
            pass
        else:
            m.bias.data.fill_(0)  # m.bias.data.fill_(0.01)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is None:
            pass
        else:
            m.bias.data.fill_(0)  # m.bias.data.fill_(0.01)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is None:
            pass
        else:
            m.bias.data.fill_(0)  # m.bias.data.fill_(0.01)


def main():
    """Main"""

    LOGGER.info('Starting demo...')


    # -----------------------------------------------------------
    # -----------------------------------------------------------
    # --------------------- Training Phase ----------------------
    # -----------------------------------------------------------
    # -----------------------------------------------------------
    LOGGER.info('Training Backbone...')

    # ------------------- Data loader -------------------
    train_data_transform = transforms.Compose([
        trsf.ImageTrsf(),
        trsf.Joints3DTrsf(),
        trsf.ToTensor()])

    # let's load data from validation set as example
    train_data = Mocap(
        config_backbone.dataset.train,
        SetType.TRAIN,
        transform=train_data_transform)
    train_data_loader = DataLoader(
        train_data,
        batch_size=config_backbone.train_data_loader.batch_size,
        shuffle=config_backbone.train_data_loader.shuffle,
        num_workers=config_backbone.train_data_loader.workers)
    
    # ------------------- Build Model -------------------
    if config_backbone.train_setting.backbone_type == "resnet18":
        backbone = resnet18()
        LOGGER.info('Using ResNet18 Backbone!')
    elif config_backbone.train_setting.backbone_type == "resnet34":
        backbone = resnet34()
        LOGGER.info('Using ResNet34 Backbone!')
    elif config_backbone.train_setting.backbone_type == "resnet50":
        backbone = resnet50()
        LOGGER.info('Using ResNet50 Backbone!')
    elif config_backbone.train_setting.backbone_type == "resnet101":
        backbone = resnet101()
        LOGGER.info('Using ResNet101 Backbone!')
    elif config_backbone.train_setting.backbone_type == "resnet152":
        backbone = resnet152()
        LOGGER.info('Using ResNet152 Backbone!')
    # encoder = HeatmapEncoder()
    # decoder = PoseDecoder()
    # reconstructer = HeatmapReconstructer()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        LOGGER.info(str("Let's use " + str(torch.cuda.device_count()) + " GPUs!"))
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        backbone = nn.DataParallel(backbone)
        # encoder = nn.DataParallel(encoder)
        # decoder = nn.DataParallel(decoder)
        # reconstructer = nn.DataParallel(reconstructer)
    backbone = backbone.cuda()
    # encoder = encoder.cuda()
    # decoder = decoder.cuda()
    # reconstructer = reconstructer.cuda()

    # Load or Init Model Weights
    if config_backbone.train_setting.backbone_path:
        backbone.load_state_dict(torch.load(config_backbone.train_setting.backbone_path))
        # backbone = torch.load(config_backbone.train_setting.backbone_path)
        LOGGER.info('Backbone Weight Loaded!')
    else:
        backbone.apply(init_weights)
        LOGGER.info('Backbone Weight Initialized!')
    # if config_backbone.train_setting.encoder_path:
    #     encoder.load_state_dict(torch.load(config_backbone.train_setting.encoder_path))
    # else:
    #     encoder.apply(init_weights)
    # if config_backbone.train_setting.decoder_path:
    #     decoder.load_state_dict(torch.load(config_backbone.train_setting.decoder_path))
    # else:
    #     decoder.apply(init_weights)
    # if config_backbone.train_setting.reconstructer_path:
    #     reconstructer.load_state_dict(torch.load(config_backbone.train_setting.reconstructer_path))
    # else:
    #     reconstructer.apply(init_weights)
    
    # ------------------- Build Loss & Optimizer -------------------
    # Build Loss
    heatmap_prediction_loss_func = nn.MSELoss(reduction='mean')
    # pose_prediction_cosine_similarity_loss_func = nn.CosineSimilarity()
    # pose_prediction_l1_loss_func = nn.L1Loss()
    # heatmap_reconstruction_loss_func = nn.MSELoss()

    # Build Optimizer
    optimizer = optim.Adam([
        {"params": backbone.parameters()},
        # {"params": encoder.parameters()},
        # {"params": decoder.parameters()},
        # {"params": reconstructer.parameters()}
    ], lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Variable for Final Model Selection
    # errorMin = 100
    # errorMinIsUpdatedInThisEpoch = False
    # ------------------- Read dataset frames -------------------
    for ep in range(config_backbone.train_setting.epoch):
        backbone.train()
        # encoder.train()
        # decoder.train()
        # reconstructer.train()

        # Averagemeter for Epoch
        lossAverageMeter = AverageMeter()
        # fullBodyErrorAverageMeter = AverageMeter()
        # upperBodyErrorAverageMeter = AverageMeter()
        # lowerBodyErrorAverageMeter = AverageMeter()
        heatmapPredictionErrorAverageMeter = AverageMeter()
        # heatmapReconstructionErrorAverageMeter() = AverageMeter()
        for it, (img, p2d, p3d, action, heatmap) in tqdm(enumerate(train_data_loader), total=len(train_data_loader)):
            #################### p2d는 각 Joint별 (x,y) 좌표를 나타낸듯. Image의 좌측상단이 (0,0)이다.
            #################### p3d는 Neck의 좌표를 (0,0,0)으로 생각했을 때의 각 Joint별 (^x,^y,^z) 좌표를 나타낸듯.
            #################### Joint 순서는 config_backbone.py에 있다.

            # LOGGER.info('Iteration: {}'.format(it))
            # LOGGER.info('Images: {}'.format(img.shape))  # (Batch, Channel, Height(y), Width(x))
            # LOGGER.info('p2dShapes: {}'.format(p2d.shape))  # (Width, Height)
            # # LOGGER.info('p2ds: {}'.format(p2d))
            # LOGGER.info('p3dShapes: {}'.format(p3d.shape))  # (^x, ^y, ^z)
            # # LOGGER.info('p3ds: {}'.format(p3d))
            # LOGGER.info('Actions: {}'.format(action))
            # LOGGER.info('heatmapShapes: {}'.format(heatmap.shape))

            # -----------------------------------------------------------
            # ------------------- Run your model here -------------------
            # -----------------------------------------------------------
            optimizer.zero_grad()

            # Move Tensors to GPUs
            img = img.cuda()
            # p3d = p3d.cuda()
            heatmap = heatmap.cuda()

            # Forward
            predicted_heatmap = backbone(img)
            # latent = encoder(predicted_heatmap)
            # predicted_pose = decoder(latent)
            # reconstructed_heatmap = reconstructer(latent)

            # Loss Calculation
            heatmap_prediction_loss = heatmap_prediction_loss_func(predicted_heatmap, heatmap)
            # p3d_for_loss = torch.cat((p3d[:, 4:6, :], p3d[:, 7:10, :], p3d[:, 11:, :]), dim=1)  # 13까지가 Upper Body
            # p3d_for_loss = torch.reshape(p3d_for_loss, (-1, 48))
            # pose_prediction_cosine_similarity_loss = pose_prediction_cosine_similarity_loss_func(predicted_pose, p3d_for_loss)
            # pose_prediction_cosine_similarity_loss = torch.mean(pose_prediction_cosine_similarity_loss)
            # pose_prediction_l1_loss = pose_prediction_l1_loss_func(predicted_pose, p3d_for_loss)
            # pose_prediction_loss = -0.01*pose_prediction_cosine_similarity_loss + 0.5*pose_prediction_l1_loss
            # heatmap_reconstruction_loss = heatmap_reconstruction_loss_func(reconstructed_heatmap, heatmap)
            # Backpropagating Loss with Weighting Factors
            backbone_loss = heatmap_prediction_loss
            # lifting_loss = 0.1*pose_prediction_loss + 0.001*heatmap_reconstruction_loss
            loss = backbone_loss

            # Backward & Update
            loss.backward()
            optimizer.step()

            # AverageMeter Update
            lossAverageMeter.update(loss.data.cpu().numpy())
        LOGGER.info(str("Training Loss in Epoch " + str(ep) + " : " + str(lossAverageMeter.avg)))


        # if ep+1 == config_backbone.train_setting.epoch:  # Test only in Final Epoch because of Training Time Issue
        if True:
            # -----------------------------------------------------------
            # -----------------------------------------------------------
            # -------------------- Validation Phase ---------------------
            # -----------------------------------------------------------
            # -----------------------------------------------------------
            LOGGER.info('Validation...')

            # ------------------- Data loader -------------------
            test_data_transform = transforms.Compose([
                trsf.ImageTrsf(),
                trsf.Joints3DTrsf(),
                trsf.ToTensor()])

            # let's load data from validation set as example
            test_data = Mocap(
                config_backbone.dataset.test,
                SetType.TEST,
                transform=test_data_transform)
            test_data_loader = DataLoader(
                test_data,
                batch_size=config_backbone.test_data_loader.batch_size,
                shuffle=config_backbone.test_data_loader.shuffle,
                num_workers=config_backbone.test_data_loader.workers)

            # ------------------- Evaluation -------------------
            # eval_body = evaluate.EvalBody()
            # eval_upper = evaluate.EvalUpperBody()
            # eval_lower = evaluate.EvalUpperBody()

            # ------------------- Read dataset frames -------------------
            backbone.eval()
            # encoder.eval()
            # decoder.eval()
            # reconstructer.eval()
            for it, (img, p2d, p3d, action, heatmap) in tqdm(enumerate(test_data_loader), total=len(test_data_loader)):
                #################### p2d는 각 Joint별 (x,y) 좌표를 나타낸듯. Image의 좌측상단이 (0,0)이다.
                #################### p3d는 Neck의 좌표를 (0,0,0)으로 생각했을 때의 각 Joint별 (^x,^y,^z) 좌표를 나타낸듯.
                #################### Joint 순서는 config_backbone.py에 있다.

                # LOGGER.info('Iteration: {}'.format(it))
                # LOGGER.info('Images: {}'.format(img.shape))  # (Batch, Channel, Height(y), Width(x))
                # LOGGER.info('p2dShapes: {}'.format(p2d.shape))  # (Width, Height)
                # # LOGGER.info('p2ds: {}'.format(p2d))
                # LOGGER.info('p3dShapes: {}'.format(p3d.shape))  # (^x, ^y, ^z)
                # # LOGGER.info('p3ds: {}'.format(p3d))
                # LOGGER.info('Actions: {}'.format(action))
                # LOGGER.info('heatmapShapes: {}'.format(heatmap.shape))

                # ------------------- Evaluate -------------------
                # TODO: replace p3d_hat with model preditions
                # p3d_hat = torch.ones_like(p3d)

                # Move Tensors to GPUs
                img = img.cuda()
                # p3d = p3d.cuda()
                heatmap = heatmap.cuda()

                # Forward
                predicted_heatmap = backbone(img)
                # latent = encoder(predicted_heatmap)
                # predicted_pose = decoder(latent)

                # Evaluate results using different evaluation metrices
                heatmap_prediction_loss = heatmap_prediction_loss_func(predicted_heatmap, heatmap)
                # predicted_pose = torch.reshape(predicted_pose, (-1, 16, 3))
                # y_output = predicted_pose.data.cpu().numpy()
                # p3d_for_loss = torch.cat((p3d[:, 4:6, :], p3d[:, 7:10, :], p3d[:, 11:, :]), dim=1)  # 13까지가 Upper Body
                # p3d_for_loss = torch.reshape(p3d_for_loss, (-1, 16, 3))
                # y_target = p3d_for_loss.data.cpu().numpy()

                # eval_body.eval(y_output, y_target, action)
                # eval_upper.eval(y_output, y_target, action)
                # eval_lower.eval(y_output, y_target, action)

                # AverageMeter Update
                heatmapPredictionErrorAverageMeter.update(heatmap_prediction_loss.data.cpu().numpy())
            LOGGER.info(str("Validation heatmapPredictionErrorAverageMeter in Epoch " + str(ep) + " : " + str(heatmapPredictionErrorAverageMeter.avg)))


            # -----------------------------------------------------------
            # -----------------------------------------------------------
            # ----------------------- Save Phase ------------------------
            # -----------------------------------------------------------
            # -----------------------------------------------------------
            LOGGER.info('Save...')

            # mkdir for this experiment
            if not os.path.exists(os.path.join(os.getcwd(), config_backbone.eval.experiment_folder)):
                os.mkdir(os.path.join(os.getcwd(), config_backbone.eval.experiment_folder))

            # mkdir for this epoch
            if not os.path.exists(os.path.join(os.getcwd(), config_backbone.eval.experiment_folder, str("epoch_" + str(ep)))):
                os.mkdir(os.path.join(os.getcwd(), config_backbone.eval.experiment_folder, str("epoch_" + str(ep))))

            # Variable for Final Model Selection
            # if errorAverageMeter.avg <= errorMin:
            #     errorMin = ErrorAverageMeter.avg
            #     errorMinIsUpdatedInThisEpoch = True

            # ------------------- Save results -------------------
            LOGGER.info('Saving evaluation results...')

            # Evaluation Result Saving
            res_train = {'HeatmapPrediction': lossAverageMeter.avg}
            # res = {'FullBody': eval_body.get_results(),
            #     'UpperBody': eval_upper.get_results(),
            #     'LowerBody': eval_lower.get_results()}
            io.write_json(os.path.join(os.getcwd(), config_backbone.eval.experiment_folder, str("epoch_" + str(ep)), config_backbone.eval.training_result_file), res_train)

            res = {'HeatmapPrediction': heatmapPredictionErrorAverageMeter.avg}
            # res = {'FullBody': eval_body.get_results(),
            #     'UpperBody': eval_upper.get_results(),
            #     'LowerBody': eval_lower.get_results()}
            io.write_json(os.path.join(os.getcwd(), config_backbone.eval.experiment_folder, str("epoch_" + str(ep)), config_backbone.eval.evaluation_result_file), res)

            # Experiement config_backboneuration Saving
            copyfile("data/config_backbone.yml", os.path.join(os.getcwd(), config_backbone.eval.experiment_folder, str("epoch_" + str(ep)), config_backbone.eval.experiment_configuration_file))

            # Model Weights Saving
            torch.save(backbone.state_dict(), os.path.join(os.getcwd(), config_backbone.eval.experiment_folder, str("epoch_" + str(ep)), config_backbone.eval.backbone_weight_file))
            # torch.save(encoder, os.path.join(os.getcwd(), config_backbone.eval.experiment_folder, str("epoch_" + ep), config_backbone.eval.encoder_weight_file))
            # torch.save(decoder, os.path.join(os.getcwd(), config_backbone.eval.experiment_folder, str("epoch_" + ep), config_backbone.eval.decoder_weight_file))
            # torch.save(reconstructer, os.path.join(os.getcwd(), config_backbone.eval.experiment_folder, str("epoch_" + ep), config_backbone.eval.reconstructer_weight_file))

            # Variable for Final Model Selection
            # errorMinIsUpdatedInThisEpoch = False

        scheduler.step()
    LOGGER.info('Done.')


if __name__ == "__main__":
    main()
