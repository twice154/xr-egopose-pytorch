# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -*- coding: utf-8 -*-
"""
Data processing where only Images and associated 3D
joint positions are loaded.

@author: Denis Tome'

"""
import os
from skimage import io as sio
import numpy as np
from base import BaseDataset
from utils import io, config

from skimage import transform as stransform

class Mocap(BaseDataset):
    """Mocap Dataset loader"""

    ROOT_DIRS = ['rgba', 'json']
    CM_TO_M = 100

    def index_db(self):

        return self._index_dir(self.path)

    def _index_dir(self, path):
        """Recursively add paths to the set of
        indexed files

        Arguments:
            path {str} -- folder path

        Returns:
            dict -- indexed files per root dir
        """

        indexed_paths = dict()
        sub_dirs, _ = io.get_subdirs(path)
        if set(self.ROOT_DIRS) <= set(sub_dirs):

            # get files from subdirs
            n_frames = -1

            # let's extract the rgba and json data per frame
            for sub_dir in self.ROOT_DIRS:
                d_path = os.path.join(path, sub_dir)
                _, paths = io.get_files(d_path)

                if n_frames < 0:
                    n_frames = len(paths)
                else:
                    if len(paths) != n_frames:
                        self.logger.error(
                            'Frames info in {} not matching other passes'.format(d_path))

                encoded = [p.encode('utf8') for p in paths]
                indexed_paths.update({sub_dir: encoded})

            return indexed_paths

        # initialize indexed_paths
        for sub_dir in self.ROOT_DIRS:
            indexed_paths.update({sub_dir: []})

        # check subdirs of path and merge info
        for sub_dir in sub_dirs:
            indexed = self._index_dir(os.path.join(path, sub_dir))

            for r_dir in self.ROOT_DIRS:
                indexed_paths[r_dir].extend(indexed[r_dir])

        return indexed_paths

    def _process_points(self, data):
        """Filter joints to select only a sub-set for
        training/evaluation

        Arguments:
            data {dict} -- data dictionary with frame info

        Returns:
            np.ndarray -- 2D joint positions, format (J x 2)
            np.ndarray -- 3D joint positions, format (J x 3)
        """

        p2d_orig = np.array(data['pts2d_fisheye']).T
        p3d_orig = np.array(data['pts3d_fisheye']).T
        joint_names = {j['name'].replace('mixamorig:', ''): jid
                       for jid, j in enumerate(data['joints'])}

        # ------------------- Filter joints -------------------

        p2d = np.empty([len(config.skel), 2], dtype=p2d_orig.dtype)
        p3d = np.empty([len(config.skel), 3], dtype=p2d_orig.dtype)

        for jid, j in enumerate(config.skel.keys()):
            p2d[jid] = p2d_orig[joint_names[j]]
            #################### Fisheye Camera라서 중간에 렌즈부분 맞춰서 Crop
            p2d[jid][0] -= (250+32)
            p2d[jid][1] -= (0+32)
            ####################

            p3d[jid] = p3d_orig[joint_names[j]]

        p3d /= self.CM_TO_M

        return p2d, p3d

    def __getitem__(self, index):

        # load image
        img_path = self.index['rgba'][index].decode('utf8')
        #################### DATASET PATH 수정해줘야함. H5 파일에 고정된 DIRECTORY로 저장되어 있는 것 같은데, 나는 고정된 DIRECTORY를 사용할 수가 없어서
        # img_path = "/SSD/xR-EgoPose/" + img_path
        ####################
        img = sio.imread(img_path).astype(np.float32)
        img /= 255.0
        #################### Fisheye Camera라서 중간에 렌즈부분 맞춰서 Crop
        img = img[0+32:800-32, 250+32:1050-32, :]  # (y, x)임
        ####################
        #################### Input Size 맞추기 위해서 1/2 Bicubic Downsampling
        img = stransform.downscale_local_mean(img, (2, 2, 1))
        ####################

        # read joint positions
        json_path = self.index['json'][index].decode('utf8')
        #################### DATASET PATH 수정해줘야함. H5 파일에 고정된 DIRECTORY로 저장되어 있는 것 같은데, 나는 고정된 DIRECTORY를 사용할 수가 없어서
        # json_path = "/SSD/xR-EgoPose/" + json_path
        ####################
        data = io.read_json(json_path)
        p2d, p3d = self._process_points(data)

        #################### 2D Image에 Labeled Keypoint 찍어보기 위한 Test Code
        # img *= 255.0
        # for (x, y) in p2d:
        #     for i in range(int(x-10), int(x+10)):
        #         for j in range(int(y-10), int(y+10)):
        #             img[j, i, :] = 0
        # sio.imsave("./test2DPose.png", img)
        ####################
        #################### 3D Space에 Labeled Keypoint 찍어보기 위한 Test Code
        # p3d *= 100
        # # This import registers the 3D projection, but is otherwise unused.
        # from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
        # import matplotlib.pyplot as plt
        # # for scattering point
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # for idx, point in enumerate(p3d):
        #     if idx < 14:
        #         ax.scatter(point[1], point[0], point[2], c='#FF0000')
        #     else:
        #         ax.scatter(point[1], point[0], point[2], c='#0000FF')
        # # for connecting point by line
        # ax.plot([p3d[5][1], p3d[4][1]], [p3d[5][0], p3d[4][0]], [p3d[5][2], p3d[4][2]], c='#FF0000')  # Head -> Neck
        # ax.plot([p3d[4][1], p3d[7][1]], [p3d[4][0], p3d[7][0]], [p3d[4][2], p3d[7][2]], c='#FF0000')  # Neck -> Left Arm
        # ax.plot([p3d[7][1], p3d[8][1]], [p3d[7][0], p3d[8][0]], [p3d[7][2], p3d[8][2]], c='#FF0000')  # Left Arm -> Left Elbow
        # ax.plot([p3d[8][1], p3d[9][1]], [p3d[8][0], p3d[9][0]], [p3d[8][2], p3d[9][2]], c='#FF0000')  # Left Elbow -> Left Hand
        # ax.plot([p3d[4][1], p3d[11][1]], [p3d[4][0], p3d[11][0]], [p3d[4][2], p3d[11][2]], c='#FF0000')  # Neck -> Right Arm
        # ax.plot([p3d[11][1], p3d[12][1]], [p3d[11][0], p3d[12][0]], [p3d[11][2], p3d[12][2]], c='#FF0000')  # Right Arm -> Right Elbow
        # ax.plot([p3d[12][1], p3d[13][1]], [p3d[12][0], p3d[13][0]], [p3d[12][2], p3d[13][2]], c='#FF0000')  # Right Elbow -> Right Hand
        # ax.plot([p3d[4][1], p3d[0][1]], [p3d[4][0], p3d[0][0]], [p3d[4][2], p3d[0][2]], c='#FF0000')  # Neck -> Hip
        # ax.plot([p3d[0][1], p3d[14][1]], [p3d[0][0], p3d[14][0]], [p3d[0][2], p3d[14][2]], c='#0000FF')  # Hip -> Left Leg
        # ax.plot([p3d[14][1], p3d[15][1]], [p3d[14][0], p3d[15][0]], [p3d[14][2], p3d[15][2]], c='#0000FF')  # Left Leg -> Left Knee
        # ax.plot([p3d[15][1], p3d[16][1]], [p3d[15][0], p3d[16][0]], [p3d[15][2], p3d[16][2]], c='#0000FF')  # Left Knee -> Left Foot
        # ax.plot([p3d[16][1], p3d[17][1]], [p3d[16][0], p3d[17][0]], [p3d[16][2], p3d[17][2]], c='#0000FF')  # Left Foot -> Left Toe
        # ax.plot([p3d[0][1], p3d[18][1]], [p3d[0][0], p3d[18][0]], [p3d[0][2], p3d[18][2]], c='#0000FF')  # Hip -> Right Leg
        # ax.plot([p3d[18][1], p3d[19][1]], [p3d[18][0], p3d[19][0]], [p3d[18][2], p3d[19][2]], c='#0000FF')  # Right Leg -> Right Knee
        # ax.plot([p3d[19][1], p3d[20][1]], [p3d[19][0], p3d[20][0]], [p3d[19][2], p3d[20][2]], c='#0000FF')  # Right Knee -> Right Foot
        # ax.plot([p3d[20][1], p3d[21][1]], [p3d[20][0], p3d[21][0]], [p3d[20][2], p3d[21][2]], c='#0000FF')  # Right Foot -> Right Toe
        # # set legend & save
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # plt.savefig("./test3DPose.png")
        ####################

        # get action name
        action = data['action']

        if self.transform:
            img = self.transform({'image': img})['image']
            p3d = self.transform({'joints3D': p3d})['joints3D']
            p2d = self.transform({'joints2D': p2d})['joints2D']
        
        #################### Keypoint 논문에 정의된 15개로 압축하고 p2d에서 manual하게 넣어준 후에 Heatmap 생성
        keypoints = np.zeros((15, 2))
        keypoints[0] = p2d[4]  # Neck
        keypoints[1] = p2d[7]  # Left Arm
        keypoints[2] = p2d[8]  # Left Elbow
        keypoints[3] = p2d[9]  # Left Hand
        keypoints[4] = p2d[11]  # Right Arm
        keypoints[5] = p2d[12]  # Right Elbow
        keypoints[6] = p2d[13]  # Right Hand
        keypoints[7] = p2d[14]  # Left Leg
        keypoints[8] = p2d[15]  # Left Knee
        keypoints[9] = p2d[16]  # Left Foot
        keypoints[10] = p2d[17]  # Left Toe
        keypoints[11] = p2d[18]  # Right Leg
        keypoints[12] = p2d[19]  # Right Knee
        keypoints[13] = p2d[20]  # Right Foot
        keypoints[14] = p2d[21]  # Right Toe
        heatmap = self.generateHeatmap(keypoints)
        ####################
        ################### 생성된 Heatmap Visualization을 위한 코드
        # heatmap = np.sum(heatmap, axis=0)
        # from PIL import Image
        # img = Image.fromarray(np.uint8(heatmap*255.0), 'L')
        # img.save("./test2DHeatmap.png")
        ###################

        return img, p2d, p3d, action, heatmap.astype(np.float32)

    def __len__(self):

        return len(self.index[self.ROOT_DIRS[0]])
