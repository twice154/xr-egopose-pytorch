import cv2
import numpy as np

Khmc = np.array([[352.59619801644876, 0.0, 0.0],
	              [0.0, 352.70276325061578, 0.0],
	              [654.6810228318458, 400.952228031277, 1.0]]).T  # camera intrinsic matrix
kd = np.array([-0.05631891929412012, -0.0038333424842925286,
                -0.00024681888617308917, -0.00012153386798050158])  # camera distortion coefficient

fisheye = cv2.imread("fisheye.png", cv2.IMREAD_COLOR)
perspective = cv2.fisheye.undistortImage(fisheye, Khmc, kd, Knew=Khmc)
cv2.imwrite("perspective_from_fisheye.png", perspective)

# https://darkpgmr.tistory.com/122?category=460965
# https://github.com/facebookresearch/xR-EgoPose/issues/7
# https://docs.opencv.org/3.4/db/d58/group__calib3d__fisheye.html#gab1ad1dc30c42ee1a50ce570019baf2c4