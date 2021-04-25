import common_prep
import pandas as pd
import cufflinks as cf
from plotly.offline import init_notebook_mode
import argparse
import numpy as np
from util import pose_utils
import transforms3d as t3d

# init_notebook_mode(connected=False)
# cf.go_offline()


def calc_relative_poses(abs_poses):
    _rel_trans_mat = np.zeros((3 * abs_poses.shape[0], 3 * abs_poses.shape[0]))
    _rel_rot_mat = np.zeros((3 * abs_poses.shape[0], 3 * abs_poses.shape[0]))
    for i, pi in enumerate(abs_poses):
        for j, pj in enumerate(abs_poses):
            if i == j:
                continue
            rel_trans_ij = pose_utils.calc_rel_trans(pi[:3], pj[:3])
            _rel_trans_mat[(3 * i):(3 * (i + 1)), (3 * j):(3 * (j + 1))] = np.diag(np.exp(rel_trans_ij))

            rot_mat_i = t3d.quaternions.quat2mat(pi[3:] / np.linalg.norm(pi[3:]))
            rot_mat_j = t3d.quaternions.quat2mat(pj[3:] / np.linalg.norm(pj[3:]))
            rel_rot_mat_ij = pose_utils.calc_relative_rot_mat(rot_mat_i, rot_mat_j)
            _rel_rot_mat[(3 * i):(3 * (i + 1)), (3 * j):(3 * (j + 1))] = rel_rot_mat_ij

    np.fill_diagonal(_rel_trans_mat, 1)
    np.fill_diagonal(_rel_rot_mat, 1)
    return _rel_trans_mat, _rel_rot_mat


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('input_file', help='Path to the dataset file')
    arg_parser.add_argument('data_path', help='Path to the dataset location')
    arg_parser.add_argument('num_of_rel_imgs', help='The number of images to use in the Spectral Relative Pose estimator',
                            type=int, default=2)
    arg_parser.add_argument('--noise_level', help='The level of location uncertainty', type=float, default=0.0)
    args = arg_parser.parse_args()

    # Extracting the ground-truth poses
    scene_data = pd.read_csv(args.input_file)

    num_of_imgs = args.num_of_rel_imgs + 1
    start_idx = 0#np.random.randint(gt_pose.shape[0] - (args.num_of_rel_imgs + 1))
    gt_pose = scene_data[['t1', 't2', 't3', 'q1', 'q2', 'q3', 'q4']].to_numpy()[start_idx:(start_idx + num_of_imgs)]

    # Assemble the relative poses matrix
    rel_exp_trans_mat, rel_rot_mat = calc_relative_poses(gt_pose)

    # Calculate the absolute translation using spectral synchronization
    N, v = np.linalg.eig(rel_exp_trans_mat)
    _N = np.real(N)
    abs_trans_poses = np.zeros((3 * num_of_imgs, 3))
    count = 0
    for i, n in enumerate(_N):
        if np.round(n).astype(np.int) == num_of_imgs:
            abs_trans_poses[:, count] = np.real(v[:, i])
            count += 1

    trans_pos_est_err = np.sum(np.abs(abs_trans_poses - gt_pose[:, :3]))

    # Calculate the absolute orientation using spectral synchronization
    N, v = np.linalg.eig(rel_rot_mat)
    _N = np.real(N)
    abs_rot_poses = np.zeros((3 * num_of_imgs, 3))
    count = 0
    for i, n in enumerate(_N):
        if np.round(n).astype(np.int) == num_of_imgs:
            abs_rot_poses[:, count] = np.real(v[:, i])
            count += 1

    gt_abs_rot_poses = np.zeros((3 * num_of_imgs, 3))
    count = 0
    for i, n in enumerate(gt_pose):
        rot_mat = t3d.quaternions.quat2mat(n[3:] / np.linalg.norm(n[3:]))
        gt_abs_rot_poses[(3 * i):(3 * (i + 1)), :] = rot_mat
        count += 1

    rot_pos_est_err = np.sum(np.abs(abs_rot_poses - gt_abs_rot_poses))
    print('Rotation estimation err: {}'.format(rot_pos_est_err))
