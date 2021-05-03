import common_prep
import pandas as pd
import cufflinks as cf
from plotly.offline import init_notebook_mode
import argparse
import numpy as np
from util.spectral_sync import SpectralSync

# init_notebook_mode(connected=False)
# cf.go_offline()

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
    start_idx = np.random.randint(scene_data.shape[0] - (args.num_of_rel_imgs + 1))
    gt_pose = scene_data[['t1', 't2', 't3', 'q1', 'q2', 'q3', 'q4']].to_numpy()[start_idx:(start_idx + num_of_imgs)]

    spectral_sync = SpectralSync()
    est_trans_poses, est_rot_poses = spectral_sync.run_spectral_synchronization(gt_pose, verbose=True)
