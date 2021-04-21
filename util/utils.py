import logging
import logging.config
import PIL
import json
from os.path import join, exists, split, realpath
import time
from os import mkdir, getcwd
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
import transforms3d as t3d

# algo related utilities
#######################


def build_rel_matrix(camera_poses, indices, k):
    # K X K  X 7 matrix with relative poses a set of cameras
    knn_rel_poses = np.zeros((k, k, 7))
    for i, index0 in enumerate(indices):
        for j, index1 in enumerate(indices):
            knn_rel_poses[i, j, :] = rel_pose(camera_poses[index0], camera_poses[index1])
    return knn_rel_poses

def fetch_knn(query, db, k):
    """
    Fetch the indices of the k nearest neighbors in the given dataset's embedding based on the L2 distance
    :param query: (torch.tensor) an embedding of a query
    :param db: (torch.tensor) an embedding of a dataset
    :param k: (int) the number of requested nearest-neighbors
    :return: (np.array) indices of kNN
    """
    distances = torch.norm(db - query, dim=1)
    return np.argsort(distances.cpu().numpy())[:k]


def rel_pose(target_pose, query_pose):
    """
    Calculate the relative pose from target to query
    Returns the translation and rotation (quaternion representation)
    """
    # t_target + delta_t = t_query
    # delta_r = t_query - t_target
    delta_t = query_pose[:3] - target_pose[:3]

    # r_target*delta_r = r_query
    # delta_r = inv(r_target)*r_query
    target_quat = target_pose[3:]
    query_quat = query_pose[3:]
    target_rot_m = t3d.quaternions.quat2mat(target_quat / np.linalg.norm(target_quat))
    query_rot_m = t3d.quaternions.quat2mat(query_quat / np.linalg.norm(target_quat))
    delta_rot = np.dot(np.linalg.inv(target_rot_m), query_rot_m)
    delta_quat = t3d.quaternions.mat2quat(delta_rot)
    return delta_t, delta_quat

# Logging and output utils
##########################
def get_stamp_from_log():
    """
    Get the time stamp from the log file
    :return:
    """
    return split(logging.getLogger().handlers[0].baseFilename)[-1].replace(".log","")


def create_output_dir(name):
    """
    Create a new directory for outputs, if it does not already exist
    :param name: (str) the name of the directory
    :return: the path to the outpur directory
    """
    out_dir = join(getcwd(), name)
    if not exists(out_dir):
        mkdir(out_dir)
    return out_dir


def init_logger():
    """
    Initialize the logger and create a time stamp for the file
    """
    path = split(realpath(__file__))[0]

    with open(join(path, 'log_config.json')) as json_file:
        log_config_dict = json.load(json_file)
        filename = log_config_dict.get('handlers').get('file_handler').get('filename')
        filename = ''.join([filename, "_", time.strftime("%d_%m_%y_%H_%M", time.localtime()), ".log"])

        # Creating logs' folder is needed
        log_path = create_output_dir('out')

        log_config_dict.get('handlers').get('file_handler')['filename'] = join(log_path, filename)
        logging.config.dictConfig(log_config_dict)

        # disable external modules' loggers (level warning and below)
        logging.getLogger(PIL.__name__).setLevel(logging.WARNING)



# Evaluation utils
##########################
def pose_err(est_pose, gt_pose):
    """
    Calculate the position and orientation error given the estimated and ground truth pose(s
    :param est_pose: (torch.Tensor) a batch of estimated poses (Nx7, N is the batch size)
    :param gt_pose: (torch.Tensor) a batch of ground-truth poses (Nx7, N is the batch size)
    :return: position error(s) and orientation errors(s)
    """
    posit_err = torch.norm(est_pose[:, 0:3] - gt_pose[:, 0:3], dim=1)
    est_pose_q = F.normalize(est_pose[:, 3:], p=2, dim=1)
    gt_pose_q = F.normalize(gt_pose[:, 3:], p=2, dim=1)
    inner_prod = torch.bmm(est_pose_q.view(est_pose_q.shape[0], 1, est_pose_q.shape[1]),
                           gt_pose_q.view(gt_pose_q.shape[0], gt_pose_q.shape[1], 1))
    orient_err = 2 * torch.acos(torch.abs(inner_prod)) * 180 / np.pi
    return posit_err, orient_err

# Plotting utils
##########################
def plot_loss_func(sample_count, loss_vals, loss_fig_path):
    plt.figure()
    plt.plot(sample_count, loss_vals)
    plt.grid()
    plt.title('Camera Pose Loss')
    plt.xlabel('Number of samples')
    plt.ylabel('Loss')
    plt.savefig(loss_fig_path)

# Augmentations
train_transforms = {
    'baseline': transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize(256),
                                    transforms.RandomCrop(224),
                                    transforms.ColorJitter(0.5, 0.5, 0.5, 0.2),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])

}
test_transforms = {
    'baseline': transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])
        ])
}



