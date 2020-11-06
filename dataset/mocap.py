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
            # p2d[jid][0] -= (250+16)
            # p2d[jid][1] -= (0+16)

            p3d[jid] = p3d_orig[joint_names[j]]

        p3d /= self.CM_TO_M

        return p2d, p3d

    def __getitem__(self, index):

        # load image
        img_path = self.index['rgba'][index].decode('utf8')
        #################### DATASET PATH 수정해줘야함. H5 파일에 고정된 DIRECTORY로 저장되어 있는 것 같은데, 나는 고정된 DIRECTORY를 사용할 수가 없어서
        # img_path = "/SSD/xR-EgoPose/" + img_path
        img = sio.imread(img_path).astype(np.float32)
        img /= 255.0

        # read joint positions
        json_path = self.index['json'][index].decode('utf8')
        #################### DATASET PATH 수정해줘야함. H5 파일에 고정된 DIRECTORY로 저장되어 있는 것 같은데, 나는 고정된 DIRECTORY를 사용할 수가 없어서
        # json_path = "/SSD/xR-EgoPose/" + json_path
        data = io.read_json(json_path)
        p2d, p3d = self._process_points(data)

        #################### 2D Image에 Labeled Keypoint 찍어보기 위한 Test Code
        # img *= 255.0
        # for (x, y) in p2d:
        #     for i in range(int(x-10), int(x+10)):
        #         for j in range(int(y-10), int(y+10)):
        #             img[j, i, :] = 255
        # sio.imsave("./test.png", img)
        # exit()

        # get action name
        action = data['action']

        if self.transform:
            img = self.transform({'image': img})['image']
            p3d = self.transform({'joints3D': p3d})['joints3D']
            p2d = self.transform({'joints2D': p2d})['joints2D']
        
        #################### Keypoint 논문에 정의된 15개로 압축하고 p2d에서 manual하게 넣어준 후에 Heatmap 생성
        # keypoints = np.zeros((15, 2))
        # keypoints[0] = p2d[4]  # Neck
        # keypoints[1] = 
        # keypoints[2] = 
        # keypoints[3] = 
        # keypoints[4] = 
        # keypoints[5] = 
        # keypoints[6] = 
        # keypoints[7] = 
        # keypoints[8] = 
        # keypoints[9] = 
        # keypoints[10] = 
        # keypoints[11] = 
        # keypoints[12] = 
        # keypoints[13] = 
        # keypoints[14] = 
        # heatmaps = self.generateHeatmap(keypoints)

        return img, p2d, p3d, action, # heatmaps.astype(np.float32)

    def __len__(self):

        return len(self.index[self.ROOT_DIRS[0]])
