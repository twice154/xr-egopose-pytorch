# Official codebase for fisheye point transformation

import json
import cv2
import numpy as np
# https://pypi.org/project/transformations/
# http://docs.ros.org/en/jade/api/tf/html/python/transformations.html
import transformations

with open("json1.json") as json_file:
    data = json.load(json_file)

    # http://www.gisdeveloper.co.kr/?p=6868
    Khmc = np.array([[352.59619801644876, 0.0, 0.0],
    	              [0.0, 352.70276325061578, 0.0],
    	              [654.6810228318458, 400.952228031277, 1.0]]).T  # camera intrinsic matrix
    kd = np.array([-0.05631891929412012, -0.0038333424842925286,
                    -0.00024681888617308917, -0.00012153386798050158])  # camera distortion coefficient

    Mmaya = np.array([[1, 0, 0, 0],
                      [0, -1, 0, 0],
                      [0, 0, -1, 0],
                      [0, 0, 0, 1]])


    h_fov = np.array(data['camera']['cam_fov'])
    translation = np.array(data['camera']['trans'])
    rotation = np.array(data['camera']['rot']) * np.pi / 180.0

    Mf = transformations.euler_matrix(rotation[0],
                                      rotation[1],
                                      rotation[2],
                                      'sxyz')

    Mf[0:3, 3] = translation
    Mf = np.linalg.inv(Mf)
    M = Mmaya.T.dot(Mf)

    joints = np.vstack([j['trans'] for j in data['joints']]).T
    Xj = M[0:3, 0:3].dot(joints) + M[0:3, 3:4]

    # https://docs.opencv.org/3.4/db/d58/group__calib3d__fisheye.html
    pts2d, jac = cv2.fisheye.projectPoints(
        Xj.T.reshape((1, -1, 3)).astype(np.float32),
        (0, 0, 0),
        (0, 0, 0),
        Khmc,
        kd
    )
    pts2d = pts2d.T.reshape((2, -1))
