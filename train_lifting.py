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
from utils import config_lifting, ConsoleLogger
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

from models import PosePredictionMSELoss
from models import PosePredictionCosineSimilarityPerJointLoss
from models import PosePredictionDistancePerJointLoss
from models import HeatmapReconstructionMSELoss


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
    LOGGER.info('Training Lifting...')

    # ------------------- Data loader -------------------
    train_data_transform = transforms.Compose([
        trsf.ImageTrsf(),
        trsf.Joints3DTrsf(),
        trsf.ToTensor()])

    # let's load data from validation set as example
    train_data = Mocap(
        config_lifting.dataset.train,
        SetType.TRAIN,
        transform=train_data_transform)
    train_data_loader = DataLoader(
        train_data,
        batch_size=config_lifting.train_data_loader.batch_size,
        shuffle=config_lifting.train_data_loader.shuffle,
        num_workers=config_lifting.train_data_loader.workers)
    
    # ------------------- Build Model -------------------
    # backbone = resnet101()
    encoder = HeatmapEncoder()
    decoder = PoseDecoder()
    reconstructer = HeatmapReconstructer()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        LOGGER.info(str("Let's use " + str(torch.cuda.device_count()) + " GPUs!"))
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        # backbone = nn.DataParallel(backbone)
        encoder = nn.DataParallel(encoder)
        decoder = nn.DataParallel(decoder)
        reconstructer = nn.DataParallel(reconstructer)
    # backbone = backbone.cuda()
    encoder = encoder.cuda()
    decoder = decoder.cuda()
    reconstructer = reconstructer.cuda()

    # Load or Init Model Weights
    # if config_lifting.train_setting.backbone_path:
    #     backbone.load_state_dict(torch.load(config_lifting.train_setting.backbone_path))
    # else:
    #     backbone.apply(init_weights)
    if config_lifting.train_setting.encoder_path:
        encoder.load_state_dict(torch.load(config_lifting.train_setting.encoder_path))
        # encoder = torch.load(config_lifting.train_setting.encoder_path)
        LOGGER.info('Encoder Weight Loaded!')
    else:
        encoder.apply(init_weights)
        LOGGER.info('Encoder Weight Initialized!')
    if config_lifting.train_setting.decoder_path:
        decoder.load_state_dict(torch.load(config_lifting.train_setting.decoder_path))
        # decoder = torch.load(config_lifting.train_setting.decoder_path)
        LOGGER.info('Decoder Weight Loaded!')
    else:
        decoder.apply(init_weights)
        LOGGER.info('Decoder Weight Initialized!')
    if config_lifting.train_setting.reconstructer_path:
        reconstructer.load_state_dict(torch.load(config_lifting.train_setting.reconstructer_path))
        # reconstructer = torch.load(config_lifting.train_setting.reconstructer_path)
        LOGGER.info('Reconstructer Weight Loaded!')
    else:
        reconstructer.apply(init_weights)
        LOGGER.info('Reconstructer Weight Initialized!')
    
    # ------------------- Build Loss & Optimizer -------------------
    # Build Loss
    pose_prediction_cosine_similarity_loss_func = PosePredictionCosineSimilarityPerJointLoss()
    pose_prediction_l1_loss_func = PosePredictionDistancePerJointLoss()
    pose_prediction_l2_loss_func = PosePredictionMSELoss()
    heatmap_reconstruction_loss_func = HeatmapReconstructionMSELoss()

    pose_prediction_cosine_similarity_loss_func = pose_prediction_cosine_similarity_loss_func.cuda()
    pose_prediction_l1_loss_func = pose_prediction_l1_loss_func.cuda()
    pose_prediction_l2_loss_func = pose_prediction_l2_loss_func.cuda()
    heatmap_reconstruction_loss_func = heatmap_reconstruction_loss_func.cuda()

    # Build Optimizer
    optimizer = optim.Adam([
        # {"params": backbone.parameters()},
        {"params": encoder.parameters()},
        {"params": decoder.parameters()},
        {"params": reconstructer.parameters()}
    ], lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Variable for Final Model Selection
    # errorMin = 100
    # errorMinIsUpdatedInThisEpoch = False
    # ------------------- Read dataset frames -------------------
    for ep in range(config_lifting.train_setting.epoch):

        # ------------------- Evaluation -------------------
        eval_body_train = evaluate.EvalBody()
        # eval_upper_train = evaluate.EvalUpperBody()
        # eval_lower_train = evaluate.EvalLowerBody()
        # eval_neck_train = evaluate.EvalNeck()
        # eval_head_train = evaluate.EvalHead()
        # eval_left_arm_train = evaluate.EvalLeftArm()
        # eval_left_elbow_train = evaluate.EvalLeftElbow()
        # eval_left_hand_train = evaluate.EvalLeftHand()
        # eval_right_arm_train = evaluate.EvalRightArm()
        # eval_right_elbow_train = evaluate.EvalRightElbow()
        # eval_right_hand_train = evaluate.EvalRightHand()
        # eval_left_leg_train = evaluate.EvalLeftLeg()
        # eval_left_knee_train = evaluate.EvalLeftKnee()
        # eval_left_foot_train = evaluate.EvalLeftFoot()
        # eval_left_toe_train = evaluate.EvalLeftToe()
        # eval_right_leg_train = evaluate.EvalRightLeg()
        # eval_right_knee_train = evaluate.EvalRightKnee()
        # eval_right_foot_train = evaluate.EvalRightFoot()
        # eval_right_toe_train = evaluate.EvalRightToe()

        # backbone.train()
        encoder.train()
        decoder.train()
        reconstructer.train()

        # Averagemeter for Epoch
        lossAverageMeter = AverageMeter()
        # fullBodyErrorAverageMeter = AverageMeter()
        # upperBodyErrorAverageMeter = AverageMeter()
        # lowerBodyErrorAverageMeter = AverageMeter()
        # heatmapPredictionErrorAverageMeter = AverageMeter()
        PosePredictionCosineSimilarityPerJointErrorAverageMeter = AverageMeter()
        PosePredictionDistancePerJointErrorAverageMeter = AverageMeter()
        PosePredictionMSEErrorAverageMeter = AverageMeter()
        heatmapReconstructionErrorAverageMeter = AverageMeter()
        # neckErrorAverageMeter = AverageMeter()
        # headErrorAverageMeter = AverageMeter()
        # leftArmErrorAverageMeter = AverageMeter()
        # leftElbowErrorAverageMeter = AverageMeter()
        # leftHandErrorAverageMeter = AverageMeter()
        # rightArmErrorAverageMeter = AverageMeter()
        # rightElbowErrorAverageMeter = AverageMeter()
        # rightHandErrorAverageMeter = AverageMeter()
        # leftLegErrorAverageMeter = AverageMeter()
        # leftKneeErrorAverageMeter = AverageMeter()
        # leftFootErrorAverageMeter = AverageMeter()
        # leftToeErrorAverageMeter = AverageMeter()
        # rightLegErrorAverageMeter = AverageMeter()
        # rightKneeErrorAverageMeter = AverageMeter()
        # rightFootErrorAverageMeter = AverageMeter()
        # rightToeErrorAverageMeter = AverageMeter()
        lossAverageMeterTrain = AverageMeter()
        # fullBodyErrorAverageMeterTrain = AverageMeter()
        # upperBodyErrorAverageMeterTrain = AverageMeter()
        # lowerBodyErrorAverageMeterTrain = AverageMeter()
        # heatmapPredictionErrorAverageMeterTrain = AverageMeter()
        PosePredictionCosineSimilarityPerJointErrorAverageMeterTrain = AverageMeter()
        PosePredictionDistancePerJointErrorAverageMeterTrain = AverageMeter()
        PosePredictionMSEErrorAverageMeterTrain = AverageMeter()
        heatmapReconstructionErrorAverageMeterTrain = AverageMeter()
        # neckErrorAverageMeterTrain = AverageMeter()
        # headErrorAverageMeterTrain = AverageMeter()
        # leftArmErrorAverageMeterTrain = AverageMeter()
        # leftElbowErrorAverageMeterTrain = AverageMeter()
        # leftHandErrorAverageMeterTrain = AverageMeter()
        # rightArmErrorAverageMeterTrain = AverageMeter()
        # rightElbowErrorAverageMeterTrain = AverageMeter()
        # rightHandErrorAverageMeterTrain = AverageMeter()
        # leftLegErrorAverageMeterTrain = AverageMeter()
        # leftKneeErrorAverageMeterTrain = AverageMeter()
        # leftFootErrorAverageMeterTrain = AverageMeter()
        # leftToeErrorAverageMeterTrain = AverageMeter()
        # rightLegErrorAverageMeterTrain = AverageMeter()
        # rightKneeErrorAverageMeterTrain = AverageMeter()
        # rightFootErrorAverageMeterTrain = AverageMeter()
        # rightToeErrorAverageMeterTrain = AverageMeter()
        for it, (img, p2d, p3d, action, heatmap) in tqdm(enumerate(train_data_loader), total=len(train_data_loader)):
            #################### p2d는 각 Joint별 (x,y) 좌표를 나타낸듯. Image의 좌측상단이 (0,0)이다.
            #################### p3d는 Neck의 좌표를 (0,0,0)으로 생각했을 때의 각 Joint별 (^x,^y,^z) 좌표를 나타낸듯.
            #################### Joint 순서는 config_lifting.py에 있다.

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
            # img = img.cuda()
            p3d = p3d.cuda()
            heatmap = heatmap.cuda()

            # Forward
            # predicted_heatmap = backbone(img)
            latent = encoder(heatmap)
            predicted_pose = decoder(latent)
            reconstructed_heatmap = reconstructer(latent)

            # Loss Calculation
            # heatmap_prediction_loss = heatmap_prediction_loss_func(predicted_heatmap, heatmap)
            p3d_for_loss = torch.cat((p3d[:, 4:6, :], p3d[:, 7:10, :], p3d[:, 11:, :]), dim=1)  # 13까지가 Upper Body
            p3d_for_loss = torch.reshape(p3d_for_loss, (-1, 48))
            pose_prediction_cosine_similarity_loss = pose_prediction_cosine_similarity_loss_func(predicted_pose, p3d_for_loss)
            pose_prediction_l1_loss = pose_prediction_l1_loss_func(predicted_pose, p3d_for_loss)
            pose_prediction_l2_loss = pose_prediction_l2_loss_func(predicted_pose, p3d_for_loss)
            pose_prediction_loss = pose_prediction_l2_loss - 0.01*pose_prediction_cosine_similarity_loss + 0.5*pose_prediction_l1_loss
            heatmap_reconstruction_loss = heatmap_reconstruction_loss_func(reconstructed_heatmap, heatmap)
            # Backpropagating Loss with Weighting Factors
            # backbone_loss = heatmap_prediction_loss
            lifting_loss = 0.1*pose_prediction_loss + 0.001*heatmap_reconstruction_loss
            # loss = backbone_loss + lifting_loss
            loss = lifting_loss
            # print(0.1*(-0.01)*pose_prediction_cosine_similarity_loss)
            # print(0.1*0.5*pose_prediction_l1_loss)
            # print(0.1*pose_prediction_l2_loss)
            # print(0.001*heatmap_reconstruction_loss)

            # Backward & Update
            loss.backward()
            optimizer.step()

            # Evaluate results using different evaluation metrices
            predicted_pose = torch.reshape(predicted_pose, (-1, 16, 3))
            y_output = predicted_pose.data.cpu().numpy()
            p3d_for_loss = torch.cat((p3d[:, 4:6, :], p3d[:, 7:10, :], p3d[:, 11:, :]), dim=1)  # 13까지가 Upper Body
            p3d_for_loss = torch.reshape(p3d_for_loss, (-1, 16, 3))
            y_target = p3d_for_loss.data.cpu().numpy()

            eval_body_train.eval(y_output, y_target, action)
            # eval_upper_train.eval(y_output, y_target, action)
            # eval_lower_train.eval(y_output, y_target, action)
            # eval_neck_train.eval(y_output, y_target, action)
            # eval_head_train.eval(y_output, y_target, action)
            # eval_left_arm_train.eval(y_output, y_target, action)
            # eval_left_elbow_train.eval(y_output, y_target, action)
            # eval_left_hand_train.eval(y_output, y_target, action)
            # eval_right_arm_train.eval(y_output, y_target, action)
            # eval_right_elbow_train.eval(y_output, y_target, action)
            # eval_right_hand_train.eval(y_output, y_target, action)
            # eval_left_leg_train.eval(y_output, y_target, action)
            # eval_left_knee_train.eval(y_output, y_target, action)
            # eval_left_foot_train.eval(y_output, y_target, action)
            # eval_left_toe_train.eval(y_output, y_target, action)
            # eval_right_leg_train.eval(y_output, y_target, action)
            # eval_right_knee_train.eval(y_output, y_target, action)
            # eval_right_foot_train.eval(y_output, y_target, action)
            # eval_right_toe_train.eval(y_output, y_target, action)

            # heatmap_prediction_loss = heatmap_prediction_loss_func(predicted_heatmap, heatmap)
            # heatmap_reconstruction_loss = heatmap_reconstruction_loss_func(reconstructed_heatmap, heatmap)

            # AverageMeter Update
            # fullBodyErrorAverageMeterTrain.update(eval_body_train.get_results()["All"])
            # upperBodyErrorAverageMeterTrain.update(eval_upper_train.get_results()["All"])
            # lowerBodyErrorAverageMeterTrain.update(eval_lower_train.get_results()["All"])
            # heatmapPredictionErrorAverageMeterTrain.update(heatmap_prediction_loss.data.cpu().numpy())
            PosePredictionCosineSimilarityPerJointErrorAverageMeterTrain.update(-0.001 * pose_prediction_cosine_similarity_loss.data.cpu().numpy())
            PosePredictionDistancePerJointErrorAverageMeterTrain.update(0.05 * pose_prediction_l1_loss.data.cpu().numpy())
            PosePredictionMSEErrorAverageMeterTrain.update(0.1 * pose_prediction_l2_loss.data.cpu().numpy())
            heatmapReconstructionErrorAverageMeterTrain.update(0.001 * heatmap_reconstruction_loss.data.cpu().numpy())
            # neckErrorAverageMeterTrain.update(eval_neck_train.get_results()["All"])
            # headErrorAverageMeterTrain.update(eval_head_train.get_results()["All"])
            # leftArmErrorAverageMeterTrain.update(eval_left_arm_train.get_results()["All"])
            # leftElbowErrorAverageMeterTrain.update(eval_left_elbow_train.get_results()["All"])
            # leftHandErrorAverageMeterTrain.update(eval_left_hand_train.get_results()["All"])
            # rightArmErrorAverageMeterTrain.update(eval_right_arm_train.get_results()["All"])
            # rightElbowErrorAverageMeterTrain.update(eval_right_elbow_train.get_results()["All"])
            # rightHandErrorAverageMeterTrain.update(eval_right_hand_train.get_results()["All"])
            # leftLegErrorAverageMeterTrain.update(eval_left_leg_train.get_results()["All"])
            # leftKneeErrorAverageMeterTrain.update(eval_left_knee_train.get_results()["All"])
            # leftFootErrorAverageMeterTrain.update(eval_left_foot_train.get_results()["All"])
            # leftToeErrorAverageMeterTrain.update(eval_left_toe_train.get_results()["All"])
            # rightLegErrorAverageMeterTrain.update(eval_right_leg_train.get_results()["All"])
            # rightKneeErrorAverageMeterTrain.update(eval_right_knee_train.get_results()["All"])
            # rightFootErrorAverageMeterTrain.update(eval_right_foot_train.get_results()["All"])
            # rightToeErrorAverageMeterTrain.update(eval_right_toe_train.get_results()["All"])

            # AverageMeter Update
            lossAverageMeterTrain.update(loss.data.cpu().numpy())
        LOGGER.info(str("Training Loss in Epoch " + str(ep) + " : " + str(lossAverageMeterTrain.avg)))
        LOGGER.info(str("Training PosePredictionCosineSimilarityPerJointErrorAverageMeter in Epoch " + str(ep) + " : " + str(PosePredictionCosineSimilarityPerJointErrorAverageMeterTrain.avg)))
        LOGGER.info(str("Training PosePredictionDistancePerJointErrorAverageMeter in Epoch " + str(ep) + " : " + str(PosePredictionDistancePerJointErrorAverageMeterTrain.avg)))
        LOGGER.info(str("Training PosePredictionMSEErrorAverageMeter in Epoch " + str(ep) + " : " + str(PosePredictionMSEErrorAverageMeterTrain.avg)))
        LOGGER.info(str("Training heatmapReconstructionErrorAverageMeter in Epoch " + str(ep) + " : " + str(heatmapReconstructionErrorAverageMeterTrain.avg)))
        LOGGER.info(str("Training fullBodyErrorAverageMeter in Epoch " + str(ep) + " : " + str(eval_body_train.get_results()["All"])))
        LOGGER.info(str("Training upperBodyErrorAverageMeter in Epoch " + str(ep) + " : " + str(eval_body_train.get_results()["UpperBody"])))
        LOGGER.info(str("Training lowerBodyErrorAverageMeter in Epoch " + str(ep) + " : " + str(eval_body_train.get_results()["LowerBody"])))
        # LOGGER.info(str("Training heatmapPredictionErrorAverageMeter in Epoch " + str(ep) + " : " + str(heatmapPredictionErrorAverageMeterTrain.avg)))

        # if ep+1 == config_lifting.train_setting.epoch:  # Test only in Final Epoch because of Training Time Issue
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
                config_lifting.dataset.test,
                SetType.TEST,
                transform=test_data_transform)
            test_data_loader = DataLoader(
                test_data,
                batch_size=config_lifting.test_data_loader.batch_size,
                shuffle=config_lifting.test_data_loader.shuffle,
                num_workers=config_lifting.test_data_loader.workers)

            # ------------------- Evaluation -------------------
            eval_body = evaluate.EvalBody()
            # eval_upper = evaluate.EvalUpperBody()
            # eval_lower = evaluate.EvalLowerBody()
            # eval_neck = evaluate.EvalNeck()
            # eval_head = evaluate.EvalHead()
            # eval_left_arm = evaluate.EvalLeftArm()
            # eval_left_elbow = evaluate.EvalLeftElbow()
            # eval_left_hand = evaluate.EvalLeftHand()
            # eval_right_arm = evaluate.EvalRightArm()
            # eval_right_elbow = evaluate.EvalRightElbow()
            # eval_right_hand = evaluate.EvalRightHand()
            # eval_left_leg = evaluate.EvalLeftLeg()
            # eval_left_knee = evaluate.EvalLeftKnee()
            # eval_left_foot = evaluate.EvalLeftFoot()
            # eval_left_toe = evaluate.EvalLeftToe()
            # eval_right_leg = evaluate.EvalRightLeg()
            # eval_right_knee = evaluate.EvalRightKnee()
            # eval_right_foot = evaluate.EvalRightFoot()
            # eval_right_toe = evaluate.EvalRightToe()

            # ------------------- Read dataset frames -------------------
            # backbone.eval()
            encoder.eval()
            decoder.eval()
            reconstructer.eval()
            for it, (img, p2d, p3d, action, heatmap) in tqdm(enumerate(test_data_loader), total=len(test_data_loader)):
                #################### p2d는 각 Joint별 (x,y) 좌표를 나타낸듯. Image의 좌측상단이 (0,0)이다.
                #################### p3d는 Neck의 좌표를 (0,0,0)으로 생각했을 때의 각 Joint별 (^x,^y,^z) 좌표를 나타낸듯.
                #################### Joint 순서는 config_lifting.py에 있다.

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
                # img = img.cuda()
                p3d = p3d.cuda()
                heatmap = heatmap.cuda()

                # Forward
                # predicted_heatmap = backbone(img)
                latent = encoder(heatmap)
                predicted_pose = decoder(latent)
                reconstructed_heatmap = reconstructer(latent)

                # Loss Calculation
                # heatmap_prediction_loss = heatmap_prediction_loss_func(predicted_heatmap, heatmap)
                p3d_for_loss = torch.cat((p3d[:, 4:6, :], p3d[:, 7:10, :], p3d[:, 11:, :]), dim=1)  # 13까지가 Upper Body
                p3d_for_loss = torch.reshape(p3d_for_loss, (-1, 48))
                pose_prediction_cosine_similarity_loss = pose_prediction_cosine_similarity_loss_func(predicted_pose, p3d_for_loss)
                pose_prediction_l1_loss = pose_prediction_l1_loss_func(predicted_pose, p3d_for_loss)
                pose_prediction_l2_loss = pose_prediction_l2_loss_func(predicted_pose, p3d_for_loss)
                pose_prediction_loss = pose_prediction_l2_loss - 0.01*pose_prediction_cosine_similarity_loss + 0.5*pose_prediction_l1_loss
                heatmap_reconstruction_loss = heatmap_reconstruction_loss_func(reconstructed_heatmap, heatmap)
                # Backpropagating Loss with Weighting Factors
                # backbone_loss = heatmap_prediction_loss
                lifting_loss = 0.1*pose_prediction_loss + 0.001*heatmap_reconstruction_loss
                # loss = backbone_loss + lifting_loss
                loss = lifting_loss
                # print(0.1*(-0.01)*pose_prediction_cosine_similarity_loss)
                # print(0.1*0.5*pose_prediction_l1_loss)
                # print(0.1*pose_prediction_l2_loss)
                # print(0.001*heatmap_reconstruction_loss)

                # Evaluate results using different evaluation metrices
                predicted_pose = torch.reshape(predicted_pose, (-1, 16, 3))
                y_output = predicted_pose.data.cpu().numpy()
                p3d_for_loss = torch.cat((p3d[:, 4:6, :], p3d[:, 7:10, :], p3d[:, 11:, :]), dim=1)  # 13까지가 Upper Body
                p3d_for_loss = torch.reshape(p3d_for_loss, (-1, 16, 3))
                y_target = p3d_for_loss.data.cpu().numpy()

                eval_body.eval(y_output, y_target, action)
                # eval_upper.eval(y_output, y_target, action)
                # eval_lower.eval(y_output, y_target, action)
                # eval_neck.eval(y_output, y_target, action)
                # eval_head.eval(y_output, y_target, action)
                # eval_left_arm.eval(y_output, y_target, action)
                # eval_left_elbow.eval(y_output, y_target, action)
                # eval_left_hand.eval(y_output, y_target, action)
                # eval_right_arm.eval(y_output, y_target, action)
                # eval_right_elbow.eval(y_output, y_target, action)
                # eval_right_hand.eval(y_output, y_target, action)
                # eval_left_leg.eval(y_output, y_target, action)
                # eval_left_knee.eval(y_output, y_target, action)
                # eval_left_foot.eval(y_output, y_target, action)
                # eval_left_toe.eval(y_output, y_target, action)
                # eval_right_leg.eval(y_output, y_target, action)
                # eval_right_knee.eval(y_output, y_target, action)
                # eval_right_foot.eval(y_output, y_target, action)
                # eval_right_toe.eval(y_output, y_target, action)

                # heatmap_reconstruction_loss = heatmap_reconstruction_loss_func(reconstructed_heatmap, heatmap)

                # AverageMeter Update
                # fullBodyErrorAverageMeter.update(eval_body.get_results()["All"])
                # upperBodyErrorAverageMeter.update(eval_upper.get_results()["All"])
                # lowerBodyErrorAverageMeter.update(eval_lower.get_results()["All"])
                # heatmapPredictionErrorAverageMeter.update(heatmap_prediction_loss.data.cpu().numpy())
                PosePredictionCosineSimilarityPerJointErrorAverageMeter.update(-0.001 * pose_prediction_cosine_similarity_loss.data.cpu().numpy())
                PosePredictionDistancePerJointErrorAverageMeter.update(0.05 * pose_prediction_l1_loss.data.cpu().numpy())
                PosePredictionMSEErrorAverageMeter.update(0.1 * pose_prediction_l2_loss.data.cpu().numpy())
                heatmapReconstructionErrorAverageMeter.update(0.001 * heatmap_reconstruction_loss.data.cpu().numpy())
                # neckErrorAverageMeter.update(eval_neck.get_results()["All"])
                # headErrorAverageMeter.update(eval_head.get_results()["All"])
                # leftArmErrorAverageMeter.update(eval_left_arm.get_results()["All"])
                # leftElbowErrorAverageMeter.update(eval_left_elbow.get_results()["All"])
                # leftHandErrorAverageMeter.update(eval_left_hand.get_results()["All"])
                # rightArmErrorAverageMeter.update(eval_right_arm.get_results()["All"])
                # rightElbowErrorAverageMeter.update(eval_right_elbow.get_results()["All"])
                # rightHandErrorAverageMeter.update(eval_right_hand.get_results()["All"])
                # leftLegErrorAverageMeter.update(eval_left_leg.get_results()["All"])
                # leftKneeErrorAverageMeter.update(eval_left_knee.get_results()["All"])
                # leftFootErrorAverageMeter.update(eval_left_foot.get_results()["All"])
                # leftToeErrorAverageMeter.update(eval_left_toe.get_results()["All"])
                # rightLegErrorAverageMeter.update(eval_right_leg.get_results()["All"])
                # rightKneeErrorAverageMeter.update(eval_right_knee.get_results()["All"])
                # rightFootErrorAverageMeter.update(eval_right_foot.get_results()["All"])
                # rightToeErrorAverageMeter.update(eval_right_toe.get_results()["All"])

                # AverageMeter Update
                lossAverageMeter.update(loss.data.cpu().numpy())
            LOGGER.info(str("Validation Loss in Epoch " + str(ep) + " : " + str(lossAverageMeter.avg)))
            LOGGER.info(str("Validation PosePredictionCosineSimilarityPerJointErrorAverageMeter in Epoch " + str(ep) + " : " + str(PosePredictionCosineSimilarityPerJointErrorAverageMeter.avg)))
            LOGGER.info(str("Validation PosePredictionDistancePerJointErrorAverageMeter in Epoch " + str(ep) + " : " + str(PosePredictionDistancePerJointErrorAverageMeter.avg)))
            LOGGER.info(str("Validation PosePredictionMSEErrorAverageMeter in Epoch " + str(ep) + " : " + str(PosePredictionMSEErrorAverageMeter.avg)))
            LOGGER.info(str("Validation heatmapReconstructionErrorAverageMeter in Epoch " + str(ep) + " : " + str(heatmapReconstructionErrorAverageMeter.avg)))
            LOGGER.info(str("Validation fullBodyErrorAverageMeter in Epoch " + str(ep) + " : " + str(eval_body.get_results()["All"])))
            LOGGER.info(str("Validation upperBodyErrorAverageMeter in Epoch " + str(ep) + " : " + str(eval_body.get_results()["UpperBody"])))
            LOGGER.info(str("Validation lowerBodyErrorAverageMeter in Epoch " + str(ep) + " : " + str(eval_body.get_results()["LowerBody"])))
            # LOGGER.info(str("Validation heatmapPredictionErrorAverageMeter in Epoch " + str(ep) + " : " + str(heatmapPredictionErrorAverageMeter.avg)))


            # -----------------------------------------------------------
            # -----------------------------------------------------------
            # ----------------------- Save Phase ------------------------
            # -----------------------------------------------------------
            # -----------------------------------------------------------
            LOGGER.info('Save...')

            # mkdir for this experiment
            if not os.path.exists(os.path.join(os.getcwd(), config_lifting.eval.experiment_folder)):
                os.mkdir(os.path.join(os.getcwd(), config_lifting.eval.experiment_folder))

            # mkdir for this epoch
            if not os.path.exists(os.path.join(os.getcwd(), config_lifting.eval.experiment_folder, str("epoch_" + str(ep)))):
                os.mkdir(os.path.join(os.getcwd(), config_lifting.eval.experiment_folder, str("epoch_" + str(ep))))

            # Variable for Final Model Selection
            # if errorAverageMeter.avg <= errorMin:
            #     errorMin = ErrorAverageMeter.avg
            #     errorMinIsUpdatedInThisEpoch = True

            # ------------------- Save results -------------------
            LOGGER.info('Saving evaluation results...')

            # Training Result Saving
            res_train = {'Loss': lossAverageMeterTrain.avg,
                # 'HeatmapPrediction': heatmapPredictionErrorAverageMeterTrain.avg,
                'PosePredictionCosineSimilarityPerJoint': PosePredictionCosineSimilarityPerJointErrorAverageMeterTrain.avg,
                'PosePredictionDistancePerJoint': PosePredictionDistancePerJointErrorAverageMeterTrain.avg,
                'PosePredictionMSE': PosePredictionMSEErrorAverageMeterTrain.avg,
                'HeatmapReconstruction': heatmapReconstructionErrorAverageMeterTrain.avg,
                'FullBody': eval_body_train.get_results()["All"],
                'UpperBody': eval_body_train.get_results()["UpperBody"],
                'LowerBody': eval_body_train.get_results()["LowerBody"],
                'Neck': eval_body_train.get_results()["Neck"],
                'Head': eval_body_train.get_results()["Head"],
                'LeftArm': eval_body_train.get_results()["LeftArm"],
                'LeftElbow': eval_body_train.get_results()["LeftElbow"],
                'LeftHand': eval_body_train.get_results()["LeftHand"],
                'RightArm': eval_body_train.get_results()["RightArm"],
                'RightElbow': eval_body_train.get_results()["RightElbow"],
                'RightHand': eval_body_train.get_results()["RightHand"],
                'LeftLeg': eval_body_train.get_results()["LeftLeg"],
                'LeftKnee': eval_body_train.get_results()["LeftKnee"],
                'LeftFoot': eval_body_train.get_results()["LeftFoot"],
                'LeftToe': eval_body_train.get_results()["LeftToe"],
                'RightLeg': eval_body_train.get_results()["RightLeg"],
                'RightKnee': eval_body_train.get_results()["RightKnee"],
                'RightFoot': eval_body_train.get_results()["RightFoot"],
                'RightToe': eval_body_train.get_results()["RightToe"]}
            io.write_json(os.path.join(os.getcwd(), config_lifting.eval.experiment_folder, str("epoch_" + str(ep)), config_lifting.eval.training_result_file), res_train)

            # Evaluation Result Saving
            res = {'Loss': lossAverageMeter.avg,
                # 'HeatmapPrediction': heatmapPredictionErrorAverageMeter.avg,
                'PosePredictionCosineSimilarityPerJoint': PosePredictionCosineSimilarityPerJointErrorAverageMeter.avg,
                'PosePredictionDistancePerJoint': PosePredictionDistancePerJointErrorAverageMeter.avg,
                'PosePredictionMSE': PosePredictionMSEErrorAverageMeter.avg,
                'HeatmapReconstruction': heatmapReconstructionErrorAverageMeter.avg,
                'FullBody': eval_body.get_results()["All"],
                'UpperBody': eval_body.get_results()["UpperBody"],
                'LowerBody': eval_body.get_results()["LowerBody"],
                'Neck': eval_body.get_results()["Neck"],
                'Head': eval_body.get_results()["Head"],
                'LeftArm': eval_body.get_results()["LeftArm"],
                'LeftElbow': eval_body.get_results()["LeftElbow"],
                'LeftHand': eval_body.get_results()["LeftHand"],
                'RightArm': eval_body.get_results()["RightArm"],
                'RightElbow': eval_body.get_results()["RightElbow"],
                'RightHand': eval_body.get_results()["RightHand"],
                'LeftLeg': eval_body.get_results()["LeftLeg"],
                'LeftKnee': eval_body.get_results()["LeftKnee"],
                'LeftFoot': eval_body.get_results()["LeftFoot"],
                'LeftToe': eval_body.get_results()["LeftToe"],
                'RightLeg': eval_body.get_results()["RightLeg"],
                'RightKnee': eval_body.get_results()["RightKnee"],
                'RightFoot': eval_body.get_results()["RightFoot"],
                'RightToe': eval_body.get_results()["RightToe"]}
            io.write_json(os.path.join(os.getcwd(), config_lifting.eval.experiment_folder, str("epoch_" + str(ep)), config_lifting.eval.evaluation_result_file), res)

            # Experiement config_liftinguration Saving
            copyfile("data/config_lifting.yml", os.path.join(os.getcwd(), config_lifting.eval.experiment_folder, str("epoch_" + str(ep)), config_lifting.eval.experiment_configuration_file))

            # Model Weights Saving
            # torch.save(backbone, os.path.join(os.getcwd(), config_lifting.eval.experiment_folder, str("epoch_" + ep), config_lifting.eval.backbone_weight_file))
            torch.save(encoder.state_dict(), os.path.join(os.getcwd(), config_lifting.eval.experiment_folder, str("epoch_" + str(ep)), config_lifting.eval.encoder_weight_file))
            torch.save(decoder.state_dict(), os.path.join(os.getcwd(), config_lifting.eval.experiment_folder, str("epoch_" + str(ep)), config_lifting.eval.decoder_weight_file))
            torch.save(reconstructer.state_dict(), os.path.join(os.getcwd(), config_lifting.eval.experiment_folder, str("epoch_" + str(ep)), config_lifting.eval.reconstructer_weight_file))

            # Variable for Final Model Selection
            # errorMinIsUpdatedInThisEpoch = False

        scheduler.step()
    LOGGER.info('Done.')


if __name__ == "__main__":
    main()
