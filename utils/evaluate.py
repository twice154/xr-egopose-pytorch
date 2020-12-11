# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -*- coding: utf-8 -*-
"""
Class for model evaluation

@author: Denis Tome'

"""
import numpy as np
from base import BaseEval

__all__ = [
    'EvalBody',
    'EvalUpperBody',
    'EvalLowerBody'
]


def compute_error(pred, gt):
    """Compute error

    Arguments:
        pred {np.ndarray} -- format (N x 3)
        gt {np.ndarray} -- format (N x 3)

    Returns:
        float -- error
    """

    if pred.shape[1] != 3:
        pred = np.transpose(pred, [1, 0])

    if gt.shape[1] != 3:
        gt = np.transpose(gt, [1, 0])

    assert pred.shape == gt.shape
    error = np.sqrt(np.sum((pred - gt)**2, axis=1))

    return np.mean(error)


class EvalBody(BaseEval):
    """Eval entire body"""

    def __init__(self):
        super().__init__()
        self._SEL = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # [4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        self._UPPERBODY = [0, 1, 2, 3, 4, 5, 6, 7]
        self.error["UpperBody"] = []
        self._LOWERBODY = [8, 9, 10, 11, 12, 13, 14, 15]
        self.error["LowerBody"] = []
        self._NECK = [0]
        self.error["Neck"] = []  # 0
        self._HEAD = [1]
        self.error["Head"] = []  # 1
        self._LEFTARM = [2]
        self.error["LeftArm"] = []  # 2
        self._LEFTELBOW = [3]
        self.error["LeftElbow"] = []  # 3
        self._LEFTHAND = [4]
        self.error["LeftHand"] = []  # 4
        self._RIGHTARM = [5]
        self.error["RightArm"] = []  # 5
        self._RIGHTELBOW = [6]
        self.error["RightElbow"] = []  # 6
        self._RIGHTHAND = [7]
        self.error["RightHand"] = []  # 7, 여기까지가 UpperBody
        self._LEFTLEG = [8]
        self.error["LeftLeg"] = []  # 8
        self._LEFTKNEE = [9]
        self.error["LeftKnee"] = []  # 9
        self._LEFTFOOT = [10]
        self.error["LeftFoot"] = []  # 10
        self._LEFTTOE = [11]
        self.error["LeftToe"] = []  # 11
        self._RIGHTLEG = [12]
        self.error["RightLeg"] = []  # 12
        self._RIGHTKNEE = [13]
        self.error["RightKnee"] = []  # 13
        self._RIGHTFOOT = [14]
        self.error["RightFoot"] = []  # 14
        self._RIGHTTOE = [15]
        self.error["RightToe"] = []  # 15, 여기까지가 LowerBody

    def eval(self, pred, gt, actions=None):
        """Evaluate

        Arguments:
            pred {np.ndarray} -- predictions, format (N x 3)
            gt {np.ndarray} -- ground truth, format (N x 3)

        Keyword Arguments:
            action {str} -- action name (default: {None})
        """

        for pid, (pose_in, pose_target) in enumerate(zip(pred, gt)):
            err = compute_error(pose_in[self._SEL], pose_target[self._SEL])
            err_upperBody = compute_error(pose_in[self._UPPERBODY], pose_target[self._UPPERBODY])
            err_lowerBody = compute_error(pose_in[self._LOWERBODY], pose_target[self._LOWERBODY])
            err_neck = compute_error(pose_in[self._NECK], pose_target[self._NECK])
            err_head = compute_error(pose_in[self._HEAD], pose_target[self._HEAD])
            err_leftArm = compute_error(pose_in[self._LEFTARM], pose_target[self._LEFTARM])
            err_leftElbow = compute_error(pose_in[self._LEFTELBOW], pose_target[self._LEFTELBOW])
            err_leftHand = compute_error(pose_in[self._LEFTHAND], pose_target[self._LEFTHAND])
            err_rightArm = compute_error(pose_in[self._RIGHTARM], pose_target[self._RIGHTARM])
            err_rightElbow = compute_error(pose_in[self._RIGHTELBOW], pose_target[self._RIGHTELBOW])
            err_rightHand = compute_error(pose_in[self._RIGHTHAND], pose_target[self._RIGHTHAND])
            err_leftLeg = compute_error(pose_in[self._LEFTLEG], pose_target[self._LEFTLEG])
            err_leftKnee = compute_error(pose_in[self._LEFTKNEE], pose_target[self._LEFTKNEE])
            err_leftFoot = compute_error(pose_in[self._LEFTFOOT], pose_target[self._LEFTFOOT])
            err_leftToe = compute_error(pose_in[self._LEFTTOE], pose_target[self._LEFTTOE])
            err_rightLeg = compute_error(pose_in[self._RIGHTLEG], pose_target[self._RIGHTLEG])
            err_rightKnee = compute_error(pose_in[self._RIGHTKNEE], pose_target[self._RIGHTKNEE])
            err_rightFoot = compute_error(pose_in[self._RIGHTFOOT], pose_target[self._RIGHTFOOT])
            err_rightToe = compute_error(pose_in[self._RIGHTTOE], pose_target[self._RIGHTTOE])

            if actions:
                act_name = self._map_action_name(actions[pid])

                # add element to dictionary if not there yet
                if not self._is_action_stored(act_name):
                    self._init_action(act_name)
                self.error[act_name].append(err)

            # add to all
            act_name = 'All'
            self.error[act_name].append(err)

            # add body part error individually
            self.error["UpperBody"].append(err_upperBody)
            self.error["LowerBody"].append(err_lowerBody)
            self.error["Neck"].append(err_neck)
            self.error["Head"].append(err_head)
            self.error["LeftArm"].append(err_leftArm)
            self.error["LeftElbow"].append(err_leftElbow)
            self.error["LeftHand"].append(err_leftHand)
            self.error["RightArm"].append(err_rightArm)
            self.error["RightElbow"].append(err_rightElbow)
            self.error["RightHand"].append(err_rightHand)
            self.error["LeftLeg"].append(err_leftLeg)
            self.error["LeftKnee"].append(err_leftKnee)
            self.error["LeftFoot"].append(err_leftFoot)
            self.error["LeftToe"].append(err_leftToe)
            self.error["RightLeg"].append(err_rightLeg)
            self.error["RightKnee"].append(err_rightKnee)
            self.error["RightFoot"].append(err_rightFoot)
            self.error["RightToe"].append(err_rightToe)

    def desc(self):
        return 'Average3DError'


# class EvalUpperBody(BaseEval):
#     """Eval upper body"""

#     _SEL = [0, 1, 2, 3, 4, 5, 6, 7]  # [4, 5, 7, 8, 9, 11, 12, 13]  # [4, 5, 6, 7, 8, 9, 10, 11]

#     def eval(self, pred, gt, actions=None):
#         """Evaluate

#         Arguments:
#             pred {np.ndarray} -- predictions, format (N x 3)
#             gt {np.ndarray} -- ground truth, format (N x 3)

#         Keyword Arguments:
#             action {str} -- action name (default: {None})
#         """

#         for pid, (pose_in, pose_target) in enumerate(zip(pred, gt)):
#             err = compute_error(pose_in[self._SEL], pose_target[self._SEL])

#             if actions:
#                 act_name = self._map_action_name(actions[pid])

#                 # add element to dictionary if not there yet
#                 if not self._is_action_stored(act_name):
#                     self._init_action(act_name)
#                 self.error[act_name].append(err)

#             # add to all
#             act_name = 'All'
#             self.error[act_name].append(err)

#     def desc(self):
#         return 'UpperBody_Average3DError'


# class EvalLowerBody(BaseEval):
#     """Eval lower body"""

#     _SEL = [8, 9, 10, 11, 12, 13, 14, 15]  # [14, 15, 16, 17, 18, 19, 20, 21]  # [0, 1, 2, 3, 12, 13, 14, 15]

#     def eval(self, pred, gt, actions=None):
#         """Evaluate

#         Arguments:
#             pred {np.ndarray} -- predictions, format (N x 3)
#             gt {np.ndarray} -- ground truth, format (N x 3)

#         Keyword Arguments:
#             action {str} -- action name (default: {None})
#         """

#         for pid, (pose_in, pose_target) in enumerate(zip(pred, gt)):
#             err = compute_error(pose_in[self._SEL], pose_target[self._SEL])

#             if actions:
#                 act_name = self._map_action_name(actions[pid])

#                 # add element to dictionary if not there yet
#                 if not self._is_action_stored(act_name):
#                     self._init_action(act_name)
#                 self.error[act_name].append(err)

#             # add to all
#             act_name = 'All'
#             self.error[act_name].append(err)

#     def desc(self):
#         return 'LowerBody_Average3DError'


# class EvalNeck(BaseEval):

#     _SEL = [0]

#     def eval(self, pred, gt, actions=None):
#         """Evaluate

#         Arguments:
#             pred {np.ndarray} -- predictions, format (N x 3)
#             gt {np.ndarray} -- ground truth, format (N x 3)

#         Keyword Arguments:
#             action {str} -- action name (default: {None})
#         """

#         for pid, (pose_in, pose_target) in enumerate(zip(pred, gt)):
#             err = compute_error(pose_in[self._SEL], pose_target[self._SEL])

#             if actions:
#                 act_name = self._map_action_name(actions[pid])

#                 # add element to dictionary if not there yet
#                 if not self._is_action_stored(act_name):
#                     self._init_action(act_name)
#                 self.error[act_name].append(err)

#             # add to all
#             act_name = 'All'
#             self.error[act_name].append(err)

#     def desc(self):
#         return 'Neck_Average3DError'

# class EvalHead(BaseEval):

#     _SEL = [1]

#     def eval(self, pred, gt, actions=None):
#         """Evaluate

#         Arguments:
#             pred {np.ndarray} -- predictions, format (N x 3)
#             gt {np.ndarray} -- ground truth, format (N x 3)

#         Keyword Arguments:
#             action {str} -- action name (default: {None})
#         """

#         for pid, (pose_in, pose_target) in enumerate(zip(pred, gt)):
#             err = compute_error(pose_in[self._SEL], pose_target[self._SEL])

#             if actions:
#                 act_name = self._map_action_name(actions[pid])

#                 # add element to dictionary if not there yet
#                 if not self._is_action_stored(act_name):
#                     self._init_action(act_name)
#                 self.error[act_name].append(err)

#             # add to all
#             act_name = 'All'
#             self.error[act_name].append(err)

#     def desc(self):
#         return 'Head_Average3DError'

# class EvalLeftArm(BaseEval):

#     _SEL = [2]

#     def eval(self, pred, gt, actions=None):
#         """Evaluate

#         Arguments:
#             pred {np.ndarray} -- predictions, format (N x 3)
#             gt {np.ndarray} -- ground truth, format (N x 3)

#         Keyword Arguments:
#             action {str} -- action name (default: {None})
#         """

#         for pid, (pose_in, pose_target) in enumerate(zip(pred, gt)):
#             err = compute_error(pose_in[self._SEL], pose_target[self._SEL])

#             if actions:
#                 act_name = self._map_action_name(actions[pid])

#                 # add element to dictionary if not there yet
#                 if not self._is_action_stored(act_name):
#                     self._init_action(act_name)
#                 self.error[act_name].append(err)

#             # add to all
#             act_name = 'All'
#             self.error[act_name].append(err)

#     def desc(self):
#         return 'LeftArm_Average3DError'

# class EvalLeftElbow(BaseEval):

#     _SEL = [3]

#     def eval(self, pred, gt, actions=None):
#         """Evaluate

#         Arguments:
#             pred {np.ndarray} -- predictions, format (N x 3)
#             gt {np.ndarray} -- ground truth, format (N x 3)

#         Keyword Arguments:
#             action {str} -- action name (default: {None})
#         """

#         for pid, (pose_in, pose_target) in enumerate(zip(pred, gt)):
#             err = compute_error(pose_in[self._SEL], pose_target[self._SEL])

#             if actions:
#                 act_name = self._map_action_name(actions[pid])

#                 # add element to dictionary if not there yet
#                 if not self._is_action_stored(act_name):
#                     self._init_action(act_name)
#                 self.error[act_name].append(err)

#             # add to all
#             act_name = 'All'
#             self.error[act_name].append(err)

#     def desc(self):
#         return 'LeftElbow_Average3DError'

# class EvalLeftHand(BaseEval):

#     _SEL = [4]

#     def eval(self, pred, gt, actions=None):
#         """Evaluate

#         Arguments:
#             pred {np.ndarray} -- predictions, format (N x 3)
#             gt {np.ndarray} -- ground truth, format (N x 3)

#         Keyword Arguments:
#             action {str} -- action name (default: {None})
#         """

#         for pid, (pose_in, pose_target) in enumerate(zip(pred, gt)):
#             err = compute_error(pose_in[self._SEL], pose_target[self._SEL])

#             if actions:
#                 act_name = self._map_action_name(actions[pid])

#                 # add element to dictionary if not there yet
#                 if not self._is_action_stored(act_name):
#                     self._init_action(act_name)
#                 self.error[act_name].append(err)

#             # add to all
#             act_name = 'All'
#             self.error[act_name].append(err)

#     def desc(self):
#         return 'LeftHand_Average3DError'

# class EvalRightArm(BaseEval):

#     _SEL = [5]

#     def eval(self, pred, gt, actions=None):
#         """Evaluate

#         Arguments:
#             pred {np.ndarray} -- predictions, format (N x 3)
#             gt {np.ndarray} -- ground truth, format (N x 3)

#         Keyword Arguments:
#             action {str} -- action name (default: {None})
#         """

#         for pid, (pose_in, pose_target) in enumerate(zip(pred, gt)):
#             err = compute_error(pose_in[self._SEL], pose_target[self._SEL])

#             if actions:
#                 act_name = self._map_action_name(actions[pid])

#                 # add element to dictionary if not there yet
#                 if not self._is_action_stored(act_name):
#                     self._init_action(act_name)
#                 self.error[act_name].append(err)

#             # add to all
#             act_name = 'All'
#             self.error[act_name].append(err)

#     def desc(self):
#         return 'RightArm_Average3DError'

# class EvalRightElbow(BaseEval):

#     _SEL = [6]

#     def eval(self, pred, gt, actions=None):
#         """Evaluate

#         Arguments:
#             pred {np.ndarray} -- predictions, format (N x 3)
#             gt {np.ndarray} -- ground truth, format (N x 3)

#         Keyword Arguments:
#             action {str} -- action name (default: {None})
#         """

#         for pid, (pose_in, pose_target) in enumerate(zip(pred, gt)):
#             err = compute_error(pose_in[self._SEL], pose_target[self._SEL])

#             if actions:
#                 act_name = self._map_action_name(actions[pid])

#                 # add element to dictionary if not there yet
#                 if not self._is_action_stored(act_name):
#                     self._init_action(act_name)
#                 self.error[act_name].append(err)

#             # add to all
#             act_name = 'All'
#             self.error[act_name].append(err)

#     def desc(self):
#         return 'RightElbow_Average3DError'

# class EvalRightHand(BaseEval):

#     _SEL = [7]

#     def eval(self, pred, gt, actions=None):
#         """Evaluate

#         Arguments:
#             pred {np.ndarray} -- predictions, format (N x 3)
#             gt {np.ndarray} -- ground truth, format (N x 3)

#         Keyword Arguments:
#             action {str} -- action name (default: {None})
#         """

#         for pid, (pose_in, pose_target) in enumerate(zip(pred, gt)):
#             err = compute_error(pose_in[self._SEL], pose_target[self._SEL])

#             if actions:
#                 act_name = self._map_action_name(actions[pid])

#                 # add element to dictionary if not there yet
#                 if not self._is_action_stored(act_name):
#                     self._init_action(act_name)
#                 self.error[act_name].append(err)

#             # add to all
#             act_name = 'All'
#             self.error[act_name].append(err)

#     def desc(self):
#         return 'RightHand_Average3DError'

# class EvalLeftLeg(BaseEval):

#     _SEL = [8]

#     def eval(self, pred, gt, actions=None):
#         """Evaluate

#         Arguments:
#             pred {np.ndarray} -- predictions, format (N x 3)
#             gt {np.ndarray} -- ground truth, format (N x 3)

#         Keyword Arguments:
#             action {str} -- action name (default: {None})
#         """

#         for pid, (pose_in, pose_target) in enumerate(zip(pred, gt)):
#             err = compute_error(pose_in[self._SEL], pose_target[self._SEL])

#             if actions:
#                 act_name = self._map_action_name(actions[pid])

#                 # add element to dictionary if not there yet
#                 if not self._is_action_stored(act_name):
#                     self._init_action(act_name)
#                 self.error[act_name].append(err)

#             # add to all
#             act_name = 'All'
#             self.error[act_name].append(err)

#     def desc(self):
#         return 'LeftLeg_Average3DError'

# class EvalLeftKnee(BaseEval):

#     _SEL = [9]

#     def eval(self, pred, gt, actions=None):
#         """Evaluate

#         Arguments:
#             pred {np.ndarray} -- predictions, format (N x 3)
#             gt {np.ndarray} -- ground truth, format (N x 3)

#         Keyword Arguments:
#             action {str} -- action name (default: {None})
#         """

#         for pid, (pose_in, pose_target) in enumerate(zip(pred, gt)):
#             err = compute_error(pose_in[self._SEL], pose_target[self._SEL])

#             if actions:
#                 act_name = self._map_action_name(actions[pid])

#                 # add element to dictionary if not there yet
#                 if not self._is_action_stored(act_name):
#                     self._init_action(act_name)
#                 self.error[act_name].append(err)

#             # add to all
#             act_name = 'All'
#             self.error[act_name].append(err)

#     def desc(self):
#         return 'LeftKnee_Average3DError'

# class EvalLeftFoot(BaseEval):

#     _SEL = [10]

#     def eval(self, pred, gt, actions=None):
#         """Evaluate

#         Arguments:
#             pred {np.ndarray} -- predictions, format (N x 3)
#             gt {np.ndarray} -- ground truth, format (N x 3)

#         Keyword Arguments:
#             action {str} -- action name (default: {None})
#         """

#         for pid, (pose_in, pose_target) in enumerate(zip(pred, gt)):
#             err = compute_error(pose_in[self._SEL], pose_target[self._SEL])

#             if actions:
#                 act_name = self._map_action_name(actions[pid])

#                 # add element to dictionary if not there yet
#                 if not self._is_action_stored(act_name):
#                     self._init_action(act_name)
#                 self.error[act_name].append(err)

#             # add to all
#             act_name = 'All'
#             self.error[act_name].append(err)

#     def desc(self):
#         return 'LeftFoot_Average3DError'

# class EvalLeftToe(BaseEval):

#     _SEL = [11]

#     def eval(self, pred, gt, actions=None):
#         """Evaluate

#         Arguments:
#             pred {np.ndarray} -- predictions, format (N x 3)
#             gt {np.ndarray} -- ground truth, format (N x 3)

#         Keyword Arguments:
#             action {str} -- action name (default: {None})
#         """

#         for pid, (pose_in, pose_target) in enumerate(zip(pred, gt)):
#             err = compute_error(pose_in[self._SEL], pose_target[self._SEL])

#             if actions:
#                 act_name = self._map_action_name(actions[pid])

#                 # add element to dictionary if not there yet
#                 if not self._is_action_stored(act_name):
#                     self._init_action(act_name)
#                 self.error[act_name].append(err)

#             # add to all
#             act_name = 'All'
#             self.error[act_name].append(err)

#     def desc(self):
#         return 'LeftToe_Average3DError'

# class EvalRightLeg(BaseEval):

#     _SEL = [12]

#     def eval(self, pred, gt, actions=None):
#         """Evaluate

#         Arguments:
#             pred {np.ndarray} -- predictions, format (N x 3)
#             gt {np.ndarray} -- ground truth, format (N x 3)

#         Keyword Arguments:
#             action {str} -- action name (default: {None})
#         """

#         for pid, (pose_in, pose_target) in enumerate(zip(pred, gt)):
#             err = compute_error(pose_in[self._SEL], pose_target[self._SEL])

#             if actions:
#                 act_name = self._map_action_name(actions[pid])

#                 # add element to dictionary if not there yet
#                 if not self._is_action_stored(act_name):
#                     self._init_action(act_name)
#                 self.error[act_name].append(err)

#             # add to all
#             act_name = 'All'
#             self.error[act_name].append(err)

#     def desc(self):
#         return 'RightLeg_Average3DError'

# class EvalRightKnee(BaseEval):

#     _SEL = [13]

#     def eval(self, pred, gt, actions=None):
#         """Evaluate

#         Arguments:
#             pred {np.ndarray} -- predictions, format (N x 3)
#             gt {np.ndarray} -- ground truth, format (N x 3)

#         Keyword Arguments:
#             action {str} -- action name (default: {None})
#         """

#         for pid, (pose_in, pose_target) in enumerate(zip(pred, gt)):
#             err = compute_error(pose_in[self._SEL], pose_target[self._SEL])

#             if actions:
#                 act_name = self._map_action_name(actions[pid])

#                 # add element to dictionary if not there yet
#                 if not self._is_action_stored(act_name):
#                     self._init_action(act_name)
#                 self.error[act_name].append(err)

#             # add to all
#             act_name = 'All'
#             self.error[act_name].append(err)

#     def desc(self):
#         return 'RightKnee_Average3DError'

# class EvalRightFoot(BaseEval):

#     _SEL = [14]

#     def eval(self, pred, gt, actions=None):
#         """Evaluate

#         Arguments:
#             pred {np.ndarray} -- predictions, format (N x 3)
#             gt {np.ndarray} -- ground truth, format (N x 3)

#         Keyword Arguments:
#             action {str} -- action name (default: {None})
#         """

#         for pid, (pose_in, pose_target) in enumerate(zip(pred, gt)):
#             err = compute_error(pose_in[self._SEL], pose_target[self._SEL])

#             if actions:
#                 act_name = self._map_action_name(actions[pid])

#                 # add element to dictionary if not there yet
#                 if not self._is_action_stored(act_name):
#                     self._init_action(act_name)
#                 self.error[act_name].append(err)

#             # add to all
#             act_name = 'All'
#             self.error[act_name].append(err)

#     def desc(self):
#         return 'RightFoot_Average3DError'

# class EvalRightToe(BaseEval):

#     _SEL = [15]

#     def eval(self, pred, gt, actions=None):
#         """Evaluate

#         Arguments:
#             pred {np.ndarray} -- predictions, format (N x 3)
#             gt {np.ndarray} -- ground truth, format (N x 3)

#         Keyword Arguments:
#             action {str} -- action name (default: {None})
#         """

#         for pid, (pose_in, pose_target) in enumerate(zip(pred, gt)):
#             err = compute_error(pose_in[self._SEL], pose_target[self._SEL])

#             if actions:
#                 act_name = self._map_action_name(actions[pid])

#                 # add element to dictionary if not there yet
#                 if not self._is_action_stored(act_name):
#                     self._init_action(act_name)
#                 self.error[act_name].append(err)

#             # add to all
#             act_name = 'All'
#             self.error[act_name].append(err)

#     def desc(self):
#         return 'RightToe_Average3DError'