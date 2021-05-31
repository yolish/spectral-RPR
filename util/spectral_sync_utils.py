from .pose_utils import quat_to_mat, calc_rel_trans, calc_rel_rot_mat
import numpy as np


def calc_relative_poses(abs_poses):
    exp_rel_trans_mat = np.zeros((3 * abs_poses.shape[0], 3 * abs_poses.shape[0]))
    rel_rot_mat = np.zeros((3 * abs_poses.shape[0], 3 * abs_poses.shape[0]))
    for i, pi in enumerate(abs_poses):
        for j, pj in enumerate(abs_poses):
            if i == j:
                continue
            rel_trans_ij = calc_rel_trans(pi[:3], pj[:3])
            exp_rel_trans_mat[(3 * i):(3 * (i + 1)), (3 * j):(3 * (j + 1))] = np.diag(np.exp(rel_trans_ij))
            rot_mat_i = quat_to_mat(pi[3:])
            rot_mat_j = quat_to_mat(pj[3:])
            rel_rot_mat_ij = calc_rel_rot_mat(rot_mat_i, rot_mat_j)
            rel_rot_mat[(3 * i):(3 * (i + 1)), (3 * j):(3 * (j + 1))] = rel_rot_mat_ij

    np.fill_diagonal(exp_rel_trans_mat, 1)
    np.fill_diagonal(rel_rot_mat, 1)
    return exp_rel_trans_mat, rel_rot_mat


def decompose_poses_with_exp(poses):
    k = poses.shape[0]
    knn_exp_abs_ts = np.zeros((k * 3, 3))
    knn_abs_rots = np.zeros((k * 3, 3))
    for i in range(k):
        knn_exp_abs_ts[(3 * i):(3 * (i + 1)), :] = np.diag(np.exp(poses[i, :3]))
        knn_abs_rots[(3 * i):(3 * (i + 1)), :] = quat_to_mat(poses[i, 3:])
    return knn_exp_abs_ts, knn_abs_rots


def compose_exp_rel_trans_mat(query_ts, exp_rel_knn_trans):
    n_imgs = query_ts.shape[0]  + 1
    rel_mat = np.zeros((n_imgs * 3, n_imgs * 3))
    rel_mat[3:, 3:] = exp_rel_knn_trans
    rel_mat[:3, :3] = np.eye(3)
    for i in range(1, n_imgs):
        rel_mat[:3, (3 * i):(3 * (i + 1))] = np.diag(np.exp(query_ts[i-1]))
        rel_mat[(3 * i):(3 * (i + 1)), :3] = np.diag(np.exp(-query_ts[i-1]))
    return rel_mat


def compose_rel_rot_mat(query_rel_quats, rel_knn_rots):
    n_imgs = query_rel_quats.shape[0]  + 1
    rel_mat = np.zeros((n_imgs * 3, n_imgs * 3))
    rel_mat[3:, 3:] = rel_knn_rots
    rel_mat[:3, :3] = np.eye(3)
    for i in range(1, n_imgs):
        my_rot_mat = quat_to_mat(query_rel_quats[i-1])
        rel_mat[:3, (3 * i):(3 * (i + 1))] = my_rot_mat
        rel_mat[(3 * i):(3 * (i + 1)), :3] = np.linalg.inv(my_rot_mat)
    return rel_mat

def spectral_sync_trans(exp_rel_trans_mat, exp_known_abs_trans_mat):
    # ===================================================================
    # Calculate the absolute translation using spectral synchronization
    # ===================================================================
    # (1) Extract the eigen-vectors and eigen values of the relative translation matrix
    num_of_imgs = exp_rel_trans_mat.shape[0] // 3
    N, v = np.linalg.eig(exp_rel_trans_mat)
    _N = np.real(N)
    ev_trans_poses = np.zeros((3 * num_of_imgs, 3))
    count = 0
    for i, n in enumerate(_N):
        if np.round(n).astype(np.int32) == num_of_imgs:
            ev_trans_poses[:, count] = np.real(v[:, i])
            count += 1

    # (2) Find the linear combination of the calculated ev using the known ground-truth
    exp_abs_trans = np.zeros(ev_trans_poses.shape)
    for i in range(3):
        # Solve Ax = B using known absolute translations
        x = np.linalg.solve(ev_trans_poses[-3:, :], exp_known_abs_trans_mat[-3:, i])
        exp_abs_trans[:, i] = np.dot(ev_trans_poses, x)

    # (3) Take log
    abs_trans = np.zeros(3 * num_of_imgs)
    for i in range(num_of_imgs):
        abs_trans[(3 * i):(3 * (i + 1))] = np.log(np.diagonal(exp_abs_trans[(3 * i):(3 * (i + 1)), :]))

    query_abs_trans = abs_trans[:3]
    return query_abs_trans, abs_trans


def spectral_sync_rot(rel_rot_mat, abs_known_rot_mat):
    # ===================================================================
    # Calculate the absolute orientation using spectral synchronization
    # ===================================================================
    # (1) Extract the eigen-vectors and eigen values of the relative rotation matrix
    num_of_imgs = rel_rot_mat.shape[0] // 3
    N, v = np.linalg.eig(rel_rot_mat)
    _N = np.real(N)
    ev_rot_mats = np.zeros((3 * num_of_imgs, 3))
    count = 0
    for i, n in enumerate(_N):
        if np.round(n, 0) == num_of_imgs:
            ev_rot_mats[:, count] = np.real(v[:, i])
            count += 1

    # (2) Finding the linear combination of the calculated ev using the known ground-truth
    abs_rot_mats = np.zeros(ev_rot_mats.shape)
    for i in range(3):
        x = np.linalg.solve(ev_rot_mats[-3:, :], abs_known_rot_mat[-3:, i])
        abs_rot_mats[:, i] = np.dot(ev_rot_mats, x)

    query_abs_rot = abs_rot_mats[:3, :3]
    return query_abs_rot, abs_rot_mats