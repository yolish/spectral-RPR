import numpy as np
import transforms3d as t3d
from .pose_utils import *


class SpectralSync():
    def __init__(self):
        super(SpectralSync).__init__()
        self._num_of_imgs = None

    @staticmethod
    def _calc_relative_poses(abs_poses):
        _rel_trans_mat = np.zeros((3 * abs_poses.shape[0], 3 * abs_poses.shape[0]))
        _rel_rot_mat = np.zeros((3 * abs_poses.shape[0], 3 * abs_poses.shape[0]))
        for i, pi in enumerate(abs_poses):
            for j, pj in enumerate(abs_poses):
                if i == j:
                    continue
                rel_trans_ij = calc_rel_trans(pi[:3], pj[:3])
                _rel_trans_mat[(3 * i):(3 * (i + 1)), (3 * j):(3 * (j + 1))] = np.diag(np.exp(rel_trans_ij))

                rot_mat_i = t3d.quaternions.quat2mat(pi[3:] / np.linalg.norm(pi[3:]))
                rot_mat_j = t3d.quaternions.quat2mat(pj[3:] / np.linalg.norm(pj[3:]))
                rel_rot_mat_ij = calc_relative_rot_mat(rot_mat_i, rot_mat_j)
                _rel_rot_mat[(3 * i):(3 * (i + 1)), (3 * j):(3 * (j + 1))] = rel_rot_mat_ij

        np.fill_diagonal(_rel_trans_mat, 1)
        np.fill_diagonal(_rel_rot_mat, 1)
        return _rel_trans_mat, _rel_rot_mat

    @staticmethod
    def _find_linear_comb(v, g):
        x = np.linalg.solve(v, g)
        return x

    def run_spectral_synchronization(self, gt_pose, verbose=False):
        '''
            Suppose that A is a NxN matrix and v1,v2,...,vk are eigen-vectors of A corresponding to the same eigen-value lambda.
            If u is a linear combination of v1,...,vk, then u is an eigen-vector of A and A*u=Lambda*u.
            In our case, we need to find the linear combination of the eigen-vectors corresponding to the eigen-value N.
            Although the pose P1 is unknown, since the P2,...,PN poses are known, we can solve the equation of type Aw=b
            using a subset of the known poses
        '''
        self._num_of_imgs = gt_pose.shape[0]

        # Assemble the relative poses matrix
        rel_exp_trans_mat, rel_rot_mat = self._calc_relative_poses(gt_pose)

        # ===================================================================
        # Calculate the absolute translation using spectral synchronization
        # ===================================================================
        # (1) Extract the eigen-vectors and eigen values of the relative rotation matrix
        N, v = np.linalg.eig(rel_exp_trans_mat)
        _N = np.real(N)
        ev_trans_poses = np.zeros((3 * self._num_of_imgs, 3))
        count = 0
        for i, n in enumerate(_N):
            if np.round(n).astype(np.int32) == self._num_of_imgs:
                ev_trans_poses[:, count] = np.real(v[:, i])
                count += 1

        # (2) Assembling the absolute rotation rotation matrices (R1, R2,..., RN)
        gt_exp_abs_trans_poses = np.zeros((3 * self._num_of_imgs, 3))
        count = 0
        for i, n in enumerate(gt_pose):
            gt_exp_abs_trans_poses[(3 * i):(3 * (i + 1)), :] = np.diag(np.exp(n[:3]))
            count += 1

        # (3) Finding the linear combination of the calculated ev using the known ground-truth
        abs_exp_trans_poses = np.zeros(ev_trans_poses.shape)
        for i in range(3):
            x = self._find_linear_comb(ev_trans_poses[-3:, :], gt_exp_abs_trans_poses[-3:, i])
            abs_exp_trans_poses[:, i] = np.dot(ev_trans_poses, x)

        # (4) Calculating the absolute translation values
        gt_abs_trans_poses = np.zeros(3 * self._num_of_imgs)
        abs_trans_poses = np.zeros(3 * self._num_of_imgs)
        for i in range(3):
            gt_abs_trans_poses[(3 * i):(3 * (i + 1))] = np.log(np.diagonal(gt_exp_abs_trans_poses[(3 * i):(3 * (i + 1)), :]))
            abs_trans_poses[(3 * i):(3 * (i + 1))] = np.log(np.diagonal(abs_exp_trans_poses[(3 * i):(3 * (i + 1)), :]))

        if verbose:
            trans_pos_est_err = np.sum(np.abs(abs_trans_poses - gt_abs_trans_poses)) / self._num_of_imgs
            print('Translation estimation err: {}'.format(trans_pos_est_err))

        # ===================================================================
        # Calculate the absolute orientation using spectral synchronization
        # ===================================================================
        # (1) Extract the eigen-vectors and eigen values of the relative rotation matrix
        N, v = np.linalg.eig(rel_rot_mat)
        _N = np.real(N)
        ev_rot_poses = np.zeros((3 * self._num_of_imgs, 3))
        count = 0
        for i, n in enumerate(_N):
            if np.round(n, 0) == self._num_of_imgs:
                ev_rot_poses[:, count] = np.real(v[:, i])
                count += 1

        # (2) Assembling the absolute rotation rotation matrices (R1, R2,..., RN)
        gt_abs_rot_poses = np.zeros((3 * self._num_of_imgs, 3))
        count = 0
        for i, n in enumerate(gt_pose):
            rot_mat = t3d.quaternions.quat2mat(n[3:] / np.linalg.norm(n[3:]))
            gt_abs_rot_poses[(3 * i):(3 * (i + 1)), :] = rot_mat
            count += 1

        # (3) Finding the linear combination of the calculated ev using the known ground-truth
        abs_rot_poses = np.zeros(ev_rot_poses.shape)
        for i in range(3):
            x = self._find_linear_comb(ev_rot_poses[-3:, :], gt_abs_rot_poses[-3:, i])
            abs_rot_poses[:, i] = np.dot(ev_rot_poses, x)

        if verbose:
            rot_pos_est_err = np.sum(np.abs(abs_rot_poses - gt_abs_rot_poses)) / self._num_of_imgs
            print('Rotation estimation err: {}'.format(rot_pos_est_err))

        return abs_trans_poses, abs_rot_poses