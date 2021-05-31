from skimage.io import imread
from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
from os.path import join
from util.pose_utils import quat_to_mat, calc_rel_rot_mat, calc_rel_trans
from util.utils import fetch_knn
from util.spectral_sync_utils import calc_relative_poses, decompose_poses_with_exp
import transforms3d as t3d


class KRPRDataset(Dataset):
    """
        A class used to represent an pytorch Dataset of absolute poses with k-nearest neighbors

    """

    def __init__(self, dataset_path, labels_file, embedding_path, k, data_transform):
        """
        :param dataset_path: (str) the path to the dataset
        :param labels_file: (str) a file with images and their path labels
        :param embedding_path (str) path to an embedding of the dataset
        :param k: (int) number of neighbors
        :param data_transform: (Transform object) a torchvision transform object
        :return: an instance of' the class
        """
        # compute k-nearest neighbors by pos
        super(KRPRDataset, self).__init__()
        self.img_paths, self.poses, _, _ = read_labels_file(labels_file, dataset_path)
        self.k = k
        n = len(self.img_paths)
        self.knn = [None] * n
        dataset_embedding = torch.load(embedding_path)
        for i in range(n):
            self.knn[i] = fetch_knn(dataset_embedding[i].reshape(1, dataset_embedding.shape[1]),
                                                           dataset_embedding, n)[1:(self.k+1)]
        self.transform = data_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load query and its pose
        img = imread(self.img_paths[idx])
        if self.transform:
            img = self.transform(img)
        pose = self.poses[idx]

        # Load image of k nearest neighbors and their poses
        knn_indices = self.knn[idx]
        knn_imgs = []

        # Compute K x 7 relative poses from Knn to query
        knn_query_rel_poses = np.zeros((self.k, 7))
        t1 = pose[:3]
        rot_mat1 = quat_to_mat(pose[3:])
        for i, knn_index in enumerate(knn_indices):
            knn_imgs.append(imread(self.img_paths[knn_index]))
            t2 = self.poses[knn_index][:3]
            rot_mat2 = quat_to_mat(self.poses[knn_index][3:])
            knn_query_rel_poses[i, :3] = calc_rel_trans(t1, t2) # relative transtio
            knn_query_rel_poses[i, 3:] = t3d.quaternions.mat2quat(calc_rel_rot_mat(rot_mat1, rot_mat2))

            # Apply the same transformation on
            if self.transform:
                knn_imgs[i] = self.transform(knn_imgs[i])

        # Compute the relative poses between KNN
        knn_poses = self.poses[knn_indices, :]
        exp_rel_knn_ts, rel_knn_rots = calc_relative_poses(knn_poses)

        # Decompose absolute KNN poses
        exp_abs_knn_ts, abs_knn_rots = decompose_poses_with_exp(self.poses)

        sample = {'img': img,
                  'pose': pose,
                  'knn_imgs': torch.stack(knn_imgs),
                  'exp_abs_knn_ts': exp_abs_knn_ts,
                  'abs_knn_rots': abs_knn_rots,
                  'knn_query_rel_poses': knn_query_rel_poses,
                  'exp_rel_knn_ts': exp_rel_knn_ts,
                  'rel_knn_rots': rel_knn_rots}
        return sample


def read_labels_file(labels_file, dataset_path):
    df = pd.read_csv(labels_file)
    imgs_paths = [join(dataset_path, path) for path in df['img_path'].values]
    scenes = df['scene'].values
    scene_unique_names = np.unique(scenes)
    scene_name_to_id = dict(zip(scene_unique_names, list(range(len(scene_unique_names)))))
    scenes_ids = [scene_name_to_id[s] for s in scenes]
    n = df.shape[0]
    poses = np.zeros((n, 7))
    poses[:, 0] = df['t1'].values
    poses[:, 1] = df['t2'].values
    poses[:, 2] = df['t3'].values
    poses[:, 3] = df['q1'].values
    poses[:, 4] = df['q2'].values
    poses[:, 5] = df['q3'].values
    poses[:, 6] = df['q4'].values
    return imgs_paths, poses, scenes, scenes_ids