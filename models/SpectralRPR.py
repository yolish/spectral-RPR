import torch.nn as nn
import torch.nn.functional as F
import torch
import transforms3d as t3d
import numpy as np
from util import spectral_sync_utils, utils
import torchvision


class Backbone(nn.Module):

    def __init__(self, config):
        super(Backbone, self).__init__()
        self.encoder = torch.load(config["backbone"])
        self.output_dim = 1280 # EfficientNet
        self.avg_pooling_2d = nn.AdaptiveAvgPool2d(1)

    def forward(self, img):
        """
        Forward pass
        :param img: input image (N X C X H X W)
        :return: (torch.Tensor) (N x
        """
        x = self.encoder.extract_features(img)
        x = self.avg_pooling_2d(x)
        x = x.flatten(start_dim=1)
        return x


class SpectralRPR(nn.Module):
    """
    A Relative Pose Regressor with Spectral Synchornization for Absolute Pose Estimation
    """
    def __init__(self, config):
        super(SpectralRPR, self).__init__()
        output_dim = 1280 # EfficientNet
        self.fc_h = nn.Linear(output_dim*2, output_dim)
        self.regressor_t = nn.Linear(output_dim, 3)
        self.regressor_rot = nn.Linear(output_dim, 4)
        self._reset_parameters()
        self.dropout = nn.Dropout(config.get("dropout"))
        # Create the backbone
        self.backbone = Backbone(config)

    def get_encoding_dim(self):
        return self.backbone.output_dim

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward_backbone(self, img):
        return self.backbone(img)

    def forward_regressor_heads(self, latent_query, latent_knns):
        latent_pairs = torch.cat((latent_query, latent_knns), 1)

        latent_pairs = self.dropout(F.relu(self.fc_h(latent_pairs)))
        rel_ts = self.regressor_t(latent_pairs)
        rel_quats = self.regressor_rot(latent_pairs)
        return rel_ts, rel_quats


    @torch.no_grad()
    def forward_spectral(self, exp_rel_knn_ts, rel_knn_rots,
                                                exp_abs_knn_ts, abs_knn_rots,
                                                rel_query_ts, rel_query_quats, k):
        # rel_knn_ts = N X 3K X 3K (exponent of relative translation between K neighbors)
        # rel_knn_rots = N x 3K x 3K (relative rotation between K neighbors)
        # rel_query_ts / quats - N x 3 / 4 (relative k to query)
        device = exp_abs_knn_ts.device
        my_dtype = exp_abs_knn_ts.dtype
        batch_size = exp_abs_knn_ts.shape[0]

        # Move everything to cpu and numpy
        exp_rel_knn_ts = exp_rel_knn_ts.cpu().numpy()
        rel_knn_rots = rel_knn_rots.cpu().numpy()
        exp_abs_knn_ts = exp_abs_knn_ts.cpu().numpy()
        abs_knn_rots = abs_knn_rots.cpu().numpy()
        rel_query_ts = rel_query_ts.cpu().numpy()
        rel_query_quats = rel_query_quats.cpu().numpy()

        # Prepare data structures
        abs_ts = np.zeros((batch_size, 3))
        abs_quats = np.zeros((batch_size, 4))
        # Loop and do:
        for i in range(batch_size):
            rel_trans_mat = spectral_sync_utils.compose_exp_rel_trans_mat(rel_query_ts[i*k:(i+1)*k, :],exp_rel_knn_ts[i, :, :])
            abs_t, _ = spectral_sync_utils.spectral_sync_trans(rel_trans_mat, exp_abs_knn_ts[i, :, :])
            abs_ts[i, :] = abs_t

            rel_rot_mat = spectral_sync_utils.compose_rel_rot_mat(rel_query_quats[i * k:(i + 1) * k, :],
                                                                  rel_knn_rots[i, :, :])
            abs_rot, _ = spectral_sync_utils.spectral_sync_rot(rel_rot_mat, abs_knn_rots[i])
            abs_quats[i, :] = t3d.quaternions.mat2quat(abs_rot)

        return torch.tensor(abs_ts).to(device).to(my_dtype), torch.tensor(abs_quats).to(device).to(my_dtype)

    def forward(self, data, apply_spectral=False):
        query = data['img'] # N x C X H X W
        knns = data['knn_imgs'] # N x K X C X H X W
        batch_size = query.shape[0] # N
        k = knns.shape[1] # K

        # N x K X C X H X W ==> N*K X C X H W
        knns = knns.view(batch_size*k, *knns.shape[2:])

        #### This can be replaced later with a Transformer-based architecture
        # Compute latent representations of query and KNNs
        latent_query = self.forward_backbone(query) # N X H (H = latent backbone dimension)
        latent_knns = self.forward_backbone(knns) # N*K x H

        # Regress relative poses between query and knns
        # N*K X H (repeat each query for K times)
        latent_query = latent_query.repeat(1, k).reshape(batch_size*k, latent_query.shape[1])
        rel_query_ts, rel_query_quats = self.forward_regressor_heads(latent_query, latent_knns)

        abs_poses = None
        if apply_spectral:
            # Apply spectral synchornization
            exp_rel_knn_ts = data["exp_rel_knn_ts"]
            rel_knn_rots = data["rel_knn_rots"]
            exp_abs_knn_ts = data["exp_abs_knn_ts"]
            abs_knn_rots = data["abs_knn_rots"]

            abs_t, abs_quat = self.forward_spectral(exp_rel_knn_ts, rel_knn_rots,
                                                    exp_abs_knn_ts, abs_knn_rots,
                                                    rel_query_ts, rel_query_quats)
            abs_poses = torch.cat((abs_t, abs_quat), dim=1)

        res = {"abs_poses": abs_poses, "rel_poses": torch.cat((rel_query_ts, rel_query_quats), dim=1)}
        return res







