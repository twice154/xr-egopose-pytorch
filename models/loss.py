import torch
import torch.nn as nn


class PosePredictionMSELoss(nn.Module):
    def __init__(self):
        super(PosePredictionMSELoss, self).__init__()
    def forward(self, pred, gt):
        l = ((pred - gt)**2)
        l = l.sum(dim=1).mean(dim=0)
        return l


class PosePredictionCosineSimilarityPerJointLoss(nn.Module):
    def __init__(self):
        super(PosePredictionCosineSimilarityPerJointLoss, self).__init__()
    def forward(self, pred, gt):
        n = pred * gt
        d1 = pred**2
        d2 = gt**2
        d2 = torch.sqrt(d2.sum(dim=1))

        nNeck = n[:, 0:3]
        nNeck = nNeck.sum(dim=1)
        d1Neck = d1[:, 0:3]
        d1Neck = torch.sqrt(d1Neck.sum(dim=1))
        lNeck = (nNeck / (d1Neck * d2))

        nHead = n[:, 3:6]
        nHead = nHead.sum(dim=1)
        d1Head = d1[:, 3:6]
        d1Head = torch.sqrt(d1Head.sum(dim=1))
        lHead = (nHead / (d1Head * d2))

        nLeftArm = n[:, 6:9]
        nLeftArm = nLeftArm.sum(dim=1)
        d1LeftArm = d1[:, 6:9]
        d1LeftArm = torch.sqrt(d1LeftArm.sum(dim=1))
        lLeftArm = (nLeftArm / (d1LeftArm * d2))

        nLeftElbow = n[:, 9:12]
        nLeftElbow = nLeftElbow.sum(dim=1)
        d1LeftElbow = d1[:, 9:12]
        d1LeftElbow = torch.sqrt(d1LeftElbow.sum(dim=1))
        lLeftElbow = (nLeftElbow / (d1LeftElbow * d2))

        nLeftHand = n[:, 12:15]
        nLeftHand = nLeftHand.sum(dim=1)
        d1LeftHand = d1[:, 12:15]
        d1LeftHand = torch.sqrt(d1LeftHand.sum(dim=1))
        lLeftHand = (nLeftHand / (d1LeftHand * d2))

        nRightArm = n[:, 15:18]
        nRightArm = nRightArm.sum(dim=1)
        d1RightArm = d1[:, 15:18]
        d1RightArm = torch.sqrt(d1RightArm.sum(dim=1))
        lRightArm = (nRightArm / (d1RightArm * d2))

        nRightElbow = n[:, 18:21]
        nRightElbow = nRightElbow.sum(dim=1)
        d1RightElbow = d1[:, 18:21]
        d1RightElbow = torch.sqrt(d1RightElbow.sum(dim=1))
        lRightElbow = (nRightElbow / (d1RightElbow * d2))

        nRightHand = n[:, 21:24]
        nRightHand = nRightHand.sum(dim=1)
        d1RightHand = d1[:, 21:24]
        d1RightHand = torch.sqrt(d1RightHand.sum(dim=1))
        lRightHand = (nRightHand / (d1RightHand * d2))

        nLeftLeg = n[:, 24:27]
        nLeftLeg = nLeftLeg.sum(dim=1)
        d1LeftLeg = d1[:, 24:27]
        d1LeftLeg = torch.sqrt(d1LeftLeg.sum(dim=1))
        lLeftLeg = (nLeftLeg / (d1LeftLeg * d2))

        nLeftKnee = n[:, 27:30]
        nLeftKnee = nLeftKnee.sum(dim=1)
        d1LeftKnee = d1[:, 27:30]
        d1LeftKnee = torch.sqrt(d1LeftKnee.sum(dim=1))
        lLeftKnee = (nLeftKnee / (d1LeftKnee * d2))

        nLeftFoot = n[:, 30:33]
        nLeftFoot = nLeftFoot.sum(dim=1)
        d1LeftFoot = d1[:, 30:33]
        d1LeftFoot = torch.sqrt(d1LeftFoot.sum(dim=1))
        lLeftFoot = (nLeftFoot / (d1LeftFoot * d2))

        nLeftToe = n[:, 33:36]
        nLeftToe = nLeftToe.sum(dim=1)
        d1LeftToe = d1[:, 33:36]
        d1LeftToe = torch.sqrt(d1LeftToe.sum(dim=1))
        lLeftToe = (nLeftToe / (d1LeftToe * d2))

        nRightLeg = n[:, 36:39]
        nRightLeg = nRightLeg.sum(dim=1)
        d1RightLeg = d1[:, 36:39]
        d1RightLeg = torch.sqrt(d1RightLeg.sum(dim=1))
        lRightLeg = (nRightLeg / (d1RightLeg * d2))

        nRightKnee = n[:, 39:42]
        nRightKnee = nRightKnee.sum(dim=1)
        d1RightKnee = d1[:, 39:42]
        d1RightKnee = torch.sqrt(d1RightKnee.sum(dim=1))
        lRightKnee = (nRightKnee / (d1RightKnee * d2))

        nRightFoot = n[:, 42:45]
        nRightFoot = nRightFoot.sum(dim=1)
        d1RightFoot = d1[:, 42:45]
        d1RightFoot = torch.sqrt(d1RightFoot.sum(dim=1))
        lRightFoot = (nRightFoot / (d1RightFoot * d2))

        nRightToe = n[:, 45:48]
        nRightToe = nRightToe.sum(dim=1)
        d1RightToe = d1[:, 45:48]
        d1RightToe = torch.sqrt(d1RightToe.sum(dim=1))
        lRightToe = (nRightToe / (d1RightToe * d2))

        return (lNeck+lHead+lLeftArm+lLeftElbow+lLeftHand+lRightArm+lRightElbow+lRightHand+lLeftLeg+lLeftKnee+lLeftFoot+lLeftToe+lRightLeg+lRightKnee+lRightFoot+lRightToe).mean(dim=0)


class PosePredictionDistancePerJointLoss(nn.Module):
    def __init__(self):
        super(PosePredictionDistancePerJointLoss, self).__init__()
    def forward(self, pred, gt):
        l = ((pred - gt)**2)

        lNeck = l[:, 0:3]
        lNeck = torch.sqrt(lNeck.sum(dim=1))

        lHead = l[:, 3:6]
        lHead = torch.sqrt(lHead.sum(dim=1))

        lLeftArm = l[:, 6:9]
        lLeftArm = torch.sqrt(lLeftArm.sum(dim=1))

        lLeftElbow = l[:, 9:12]
        lLeftElbow = torch.sqrt(lLeftElbow.sum(dim=1))

        lLeftHand = l[:, 12:15]
        lLeftHand = torch.sqrt(lLeftHand.sum(dim=1))

        lRightArm = l[:, 15:18]
        lRightArm = torch.sqrt(lRightArm.sum(dim=1))

        lRightElbow = l[:, 18:21]
        lRightElbow = torch.sqrt(lRightElbow.sum(dim=1))

        lRightHand = l[:, 21:24]
        lRightHand = torch.sqrt(lRightHand.sum(dim=1))

        lLeftLeg = l[:, 24:27]
        lLeftLeg = torch.sqrt(lLeftLeg.sum(dim=1))

        lLeftKnee = l[:, 27:30]
        lLeftKnee = torch.sqrt(lLeftKnee.sum(dim=1))

        lLeftFoot = l[:, 30:33]
        lLeftFoot = torch.sqrt(lLeftFoot.sum(dim=1))

        lLeftToe = l[:, 33:36]
        lLeftToe = torch.sqrt(lLeftToe.sum(dim=1))

        lRightLeg = l[:, 36:39]
        lRightLeg = torch.sqrt(lRightLeg.sum(dim=1))

        lRightKnee = l[:, 39:42]
        lRightKnee = torch.sqrt(lRightKnee.sum(dim=1))

        lRightFoot = l[:, 42:45]
        lRightFoot = torch.sqrt(lRightFoot.sum(dim=1))

        lRightToe = l[:, 45:48]
        lRightToe = torch.sqrt(lRightToe.sum(dim=1))

        return (lNeck+lHead+lLeftArm+lLeftElbow+lLeftHand+lRightArm+lRightElbow+lRightHand+lLeftLeg+lLeftKnee+lLeftFoot+lLeftToe+lRightLeg+lRightKnee+lRightFoot+lRightToe).mean(dim=0)


class HeatmapReconstructionMSELoss(nn.Module):
    def __init__(self):
        super(HeatmapReconstructionMSELoss, self).__init__()
    def forward(self, pred, gt):
        l = ((pred - gt)**2)
        l = l.sum(dim=3).sum(dim=2).sum(dim=1).mean(dim=0)
        return l


# import torch

# class HeatmapLoss(torch.nn.Module):
#     """
#     loss for detection heatmap
#     """
#     def __init__(self):
#         super(HeatmapLoss, self).__init__()

#     def forward(self, pred, gt):
#         l = ((pred - gt)**2)
#         l = l.mean(dim=3).mean(dim=2).mean(dim=1)
#         return l ## l of dim bsize