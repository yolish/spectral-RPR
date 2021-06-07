import torch
from datasets.CameraPoseDataset import CameraPoseDataset
from util import utils
import numpy as np
import logging
import time
from util.spectral_sync_utils import calc_relative_poses, decompose_poses_with_exp
import transforms3d as t3d
from util.pose_utils import quat_to_mat, calc_rel_rot_mat, calc_rel_trans

def test(model, config, device, dataset_path, train_labels_file, test_labels_file,
         rpr_train_dataset_embedding_path, ir_train_dataset_embedding_path=None, res_output_file=None):
    """
    model: (nn.Module) RPR spectral model
    config: config object for set up
    device: (torch.device)
    dataset_path: (str) path to train dataset (where images are located)
    train_labels_file (str) path to a file associating images with their absolute poses - for the train set
    test_labels_file (str) path to a file associating images with their absolute poses - for the test set
    train_dataset_embedding_path (str) path to embedding of the train dataset for knn retrieval
    rpr_train_dataset_embedding_path (str) path to embedding of the train dataset for rp
    ir_train_dataset_embedding_path (str) path to embedding of the train dataset for knn retrieval, if NONE taken from RPR
    return test stats (mean and median pose error and mean inference time)
    """

    # Set to eval mode
    model.eval()

    # Set the dataset and data loader
    transform = utils.test_transforms.get('baseline')
    train_dataset = CameraPoseDataset(dataset_path, train_labels_file, transform)
    test_dataset = CameraPoseDataset(dataset_path, test_labels_file, transform)
    loader_params = {'batch_size': 1,
                     'shuffle': False,
                     'num_workers': config.get('n_workers')}
    dataloader = torch.utils.data.DataLoader(test_dataset, **loader_params)
    rpr_train_dataset_embedding = torch.load(rpr_train_dataset_embedding_path)
    if ir_train_dataset_embedding_path is None:
        ir_train_dataset_embedding = rpr_train_dataset_embedding
    else:
        ir_train_dataset_embedding = torch.load(ir_train_dataset_embedding_path)
    train_poses = train_dataset.poses
    k = config.get("k")

    rel_to_abs_method = config.get("rel_to_abs_method")
    abs_stats = np.zeros((len(dataloader.dataset), 3))
    rel_stats = np.zeros((len(dataloader.dataset), 2))


    relative_poses = np.zeros((k * len(test_dataset), 7))
    query_paths = []
    nn_paths = []
    query_to_knn_indices = {}
    save_res = res_output_file is not None


    with torch.no_grad():
        for i, minibatch in enumerate(dataloader, 0):
            for key, value in minibatch.items():
                minibatch[key] = value.to(device)

            # TODO consider using NetVLAD for IR purposes
            # Forward pass to predict the pose
            tic = time.time()

            # Embed the image
            latent_query = model.forward_backbone(minibatch.get('img'))

            # Fetch latent of knns
            knn_indices = utils.fetch_knn(latent_query, ir_train_dataset_embedding, k)

            knn_poses = train_poses[knn_indices, :] # GT abs poses
            query_to_knn_indices[i] = knn_indices
            latent_knns = rpr_train_dataset_embedding[knn_indices, :].unsqueeze(0)

            # Get relative poses from knns to query
            rel_query_ts, rel_query_quats = model.forward_regressor_heads(latent_query.repeat((latent_knns.shape[1], 1)),
                                                                        latent_knns.squeeze(0))

            device = rel_query_quats.device
            # Save paths and poses
            if save_res:
                for j in knn_indices:
                    query_paths.append(test_dataset.img_paths[i].replace(test_dataset.dataset_path, ""))
                    nn_paths.append(train_dataset.img_paths[j].replace(train_dataset.dataset_path, ""))
                relative_poses[i*k:i*k+k, 0:3] = rel_query_ts.cpu().numpy()
                relative_poses[i * k:i * k + k, 3:] = rel_query_quats.cpu().numpy()

            gt_query_pose = test_dataset.poses[i]
            for j in range(k):
                est_rel_pose = torch.cat([rel_query_ts[j,:], rel_query_quats[j,:]]).unsqueeze(0)
                gt_knn_pose = knn_poses[j]
                gt_rel_t = calc_rel_trans(gt_query_pose[:3], gt_knn_pose[:3])
                gt_rel_quat = t3d.quaternions.mat2quat(calc_rel_rot_mat(t3d.quaternions.quat2mat(gt_query_pose[3:]),
                                                                        t3d.quaternions.quat2mat(gt_knn_pose[3:])))
                gt_rel_pose = torch.cat([torch.Tensor(gt_rel_t).unsqueeze(0),
                                         torch.Tensor(gt_rel_quat).unsqueeze(0)], dim=1).to(est_rel_pose.device)
                rel_posit_err, rel_orient_err = utils.pose_err(est_rel_pose, gt_rel_pose)
                rel_stats[i,0] += rel_posit_err
                rel_stats[i, 1] += rel_orient_err
            rel_stats[i,0] /= k
            rel_stats[i, 1] /= k
            logging.info("Mean Relative Pose error over {} neighbors: {:.3f}[m], {:.3f}[deg]".format(
                k, rel_stats[i, 0], rel_stats[i, 1]))

            if rel_to_abs_method == 'spectral':
                # Get GT relative poses
                knn_rel_exp_ts, knn_rel_rots = calc_relative_poses(knn_poses)
                # Prepare GT abs t and R for finding the coefficients of the linear combination of the eigen vectors
                exp_abs_knn_ts, abs_knn_rots = decompose_poses_with_exp(knn_poses)
                exp_abs_knn_ts = torch.Tensor(exp_abs_knn_ts).to(device).unsqueeze(0)
                abs_knn_rots = torch.Tensor(abs_knn_rots).to(device).unsqueeze(0)
                knn_rel_exp_ts = torch.Tensor(knn_rel_exp_ts).to(device).unsqueeze(0)
                knn_rel_rots = torch.Tensor(knn_rel_rots).to(device).unsqueeze(0)
                abs_t, abs_rot = model.forward_spectral(knn_rel_exp_ts, knn_rel_rots,
                                                    exp_abs_knn_ts, abs_knn_rots,
                                                    rel_query_ts, rel_query_quats, k)
            elif rel_to_abs_method == "1nn":
                abs_1st_nn = knn_poses[0, :]
                rel_to_1st = relative_poses[i*k, :]
                abs_t = torch.Tensor(abs_1st_nn[:3] + rel_to_1st[:3]).to(device).unsqueeze(0)
                abs_rot = np.dot(t3d.quaternions.quat2mat(rel_to_1st[3:]),t3d.quaternions.quat2mat(abs_1st_nn[3:]))
                abs_rot = torch.Tensor(t3d.quaternions.mat2quat(abs_rot)).to(device).unsqueeze(0)


            elif rel_query_quats == "mean":
                pass  # TODO add here: absolute from nearest neighbor

            # get absolute pose using spectral analysis
            est_pose = torch.cat((abs_t, abs_rot), dim=1)
            toc = time.time()
            gt_pose = minibatch.get('pose').to(dtype=torch.float32)

            # Evaluate error
            posit_err, orient_err = utils.pose_err(est_pose, gt_pose)

            # Collect statistics
            abs_stats[i, 0] = posit_err.item()
            abs_stats[i, 1] = orient_err.item()
            abs_stats[i, 2] = (toc - tic) * 1000

            # Record
            logging.info("Pose error: ,{:.3f}[m], {:.3f}[deg], inferred in {:.2f}[ms]".format(
                abs_stats[i, 0], abs_stats[i, 1], abs_stats[i, 2]))

    # Write to file
    if save_res:
        f = open(res_output_file, 'w')
        f.write('query_path,nn_path,rel-t1,rel-t2,rel-t3,rel-q1,rel-q2,rel-q3,rel-q4,'
                'abs-t1,abs-t2,abs-t3,abs-q1,abs-q2,abs-q3,abs-q4,'
                'abs-nn-t1,abs-nn-t2,abs-nn-t3,abs-nn-q1,abs-nn-q2,abs-nn-q3,abs-nn-q4\n')
        for i in range(len(test_dataset)):
            indices = query_to_knn_indices[i]
            for j in indices:
                f.write('{},{},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},'
                        '{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f}'
                        '{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f}\n'.format(query_paths[i],nn_paths[i],
                                                              relative_poses[i,0], relative_poses[i,1],
                                                              relative_poses[i,2], relative_poses[i,3],
                                                              relative_poses[i,4],relative_poses[i,5],
                                                              relative_poses[i,6],
                                                              test_dataset.poses[i,0], test_dataset.poses[i,1],
                                                              test_dataset.poses[i,2], test_dataset.poses[i,3],
                                                              test_dataset.poses[i,4], test_dataset.poses[i,5],
                                                              test_dataset.poses[i,6],
                                                              train_dataset.poses[j, 0],
                                                              train_dataset.poses[j, 1],
                                                              train_dataset.poses[j, 2],
                                                              train_dataset.poses[j, 3],
                                                              train_dataset.poses[i, 4],
                                                              train_dataset.poses[j, 5],
                                                              train_dataset.poses[j, 6]
                                                              ))
        f.close()

    return abs_stats, rel_stats

