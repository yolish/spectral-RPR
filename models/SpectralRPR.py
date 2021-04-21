import torch.nn as nn
import torch.nn.functional as F
import torch


class Backbone(nn.Module):

    def __init__(self, config):
        super(Backbone, self).__init__()
        self.output_dim = 1280 # EfficientNet
        self.backbone = torch.load(config.pretrained_path)
        self.avg_pooling_2d = nn.AdaptiveAvgPool2d(1)

    def forward(self, img):
        """
        Forward pass
        :param img: input image (N X C X H X W)
        :return: (torch.Tensor) (N x
        """
        x = self.backbone.extract_features(img)
        x = self.avg_pooling_2d(x)
        x = x.flatten(start_dim=1)
        return x


class PoseRegressor(nn.Module):
    """ A simple MLP to regress a pose component"""

    def __init__(self, input_dim, output_dim):
        """
        input_dim: (int) the input dimension
        output_dim: (int) the outpur dimension
        use_prior: (bool) whether to use prior information
        """
        super().__init__()
        ch = 1024
        self.fc_h = nn.Linear(input_dim, ch)
        self.fc_o = nn.Linear(ch, output_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        Forward pass
        """
        x = F.gelu(self.fc_h(x))
        return self.fc_o(x)


class SpectralRPR(nn.Module):
    """
    A Relative Pose Regressor with Spectral Synchornization for Absolute Pose Estimation
    """
    def __init__(self, config):
        self.backbone = Backbone(config)
        self.regressor_t = PoseRegressor(self.backbone.output_dim, 3)
        self.regressor_rot = PoseRegressor(self.backbone.output_dim, 4)

    def forward_backbone(self, img):
        return self.backbone(img)

    def forward_regressor_heads(self, latent_query, latent_knns):
        latent_pairs = torch.cat((latent_query, latent_knns), 0)
        rel_ts = self.regressor_t(latent_pairs)
        rel_rots = self.regressor_rot(latent_pairs)
        return rel_ts, rel_rots

    def forward_spectral(self, rel_knn_ts, rel_knn_rots, rel_query_ts, rel_query_rots, batch_size):
        # Place holder until we add here the actual code
        abs_t = torch.ones((batch_size, 3)).to(rel_knn_ts.device)
        abs_rot = torch.ones((batch_size, 4)).to(rel_knn_rots.device)
        return abs_t, abs_rot

    def forward(self, data):
        query = data['img']
        knns = data['knn_imgs']
        batch_size = query.shape[0]
        k = knns.shape[1]

        knns = knns.view(batch_size*k, knns.shape[2])

        # Compute latent representations of query and KNNs
        latent_query = self.forward_backbone(query)
        latent_knns = self.forward_backbone(knns)

        # Regress relative poses between query and knns
        rel_query_ts, rel_query_rots = self.forward_regressor_heads(latent_query.repeat((latent_knns.shape[0], 1)), latent_knns)

        # Apply spectral synchornization
        rel_knn_poses = data['rel_knn_poses']
        rel_knn_ts = rel_knn_poses[:, :, :, :3]
        rel_knn_rots = rel_knn_poses[:, :, :, 3:]
        abs_t, abs_rot = self.foward_spectral(rel_knn_ts, rel_knn_rots, rel_query_ts, rel_query_rots)

        res = {"abs_poses": torch.cat((abs_t, abs_rot), dim=1),
               "rel_poses": torch.cat((rel_query_ts, rel_query_rots), dim=2)}
        return res




