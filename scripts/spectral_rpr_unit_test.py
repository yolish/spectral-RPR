import common_prep
import pandas as pd
import argparse
import numpy as np
from util.spectral_sync_utils import calc_relative_poses, decompose_poses_with_exp, spectral_sync_rot, spectral_sync_trans

# init_notebook_mode(connected=False)
# cf.go_offline()

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('input_file', help='Path to the dataset file')
    arg_parser.add_argument('num_of_rel_imgs', help='The number of images to use in the Spectral Relative Pose estimator',
                            type=int, default=2)
    arg_parser.add_argument('--noise_level', help='The level of location uncertainty', type=float, default=0.0)
    args = arg_parser.parse_args()

    # Extracting the ground-truth poses
    scene_data = pd.read_csv(args.input_file)

    num_of_imgs = args.num_of_rel_imgs + 1
    start_idx = np.random.randint(scene_data.shape[0] - (args.num_of_rel_imgs + 1))
    gt_poses = scene_data[['t1', 't2', 't3', 'q1', 'q2', 'q3', 'q4']].to_numpy()[start_idx:(start_idx + num_of_imgs)]

    exp_rel_trans_mat, rel_rot_mat = calc_relative_poses(gt_poses)
    exp_known_abs_trans_mat, known_abs_rot_mat = decompose_poses_with_exp(gt_poses)
    known_abs_trans_vec = gt_poses[:, :3].flatten()

    # Run spectral sync
    _, abs_trans_vec = spectral_sync_trans(exp_rel_trans_mat, exp_known_abs_trans_mat)
    _, abs_rot_mat = spectral_sync_rot(rel_rot_mat, known_abs_rot_mat)

    trans_pos_est_err = np.sum(np.abs(abs_trans_vec - known_abs_trans_vec)) / num_of_imgs
    print('Translation estimation err: {}'.format(trans_pos_est_err))
    rot_pos_est_err = np.sum(np.abs(abs_rot_mat - known_abs_rot_mat)) / num_of_imgs
    print('Rotation estimation err: {}'.format(rot_pos_est_err))
