import numpy as np


def calc_relative_rot_mat(rot_mat1, rot_mat2):
    """
    Calculates the relative rotation matrix
    """
    rot_mat2_inv = rot_mat2.T#np.linalg.inv(rot_mat2)
    rel_rot_mat_12 = np.dot(rot_mat1, rot_mat2_inv)
    return rel_rot_mat_12


def calc_rel_trans(trans1, trans2):
    """
    Calculate the relative translation
    """
    rel_trans_12 = trans1 - trans2
    return rel_trans_12
