import torch.nn as nn
import torch.nn.functional as F
import torch


class Backbone(nn.Module):

    def __init__(self, config):
        super(Backbone, self).__init__()
        self.output_dim = 1280 # EfficientNet
        self.backbone = torch.load(config["backbone"])
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
        super(PoseRegressor, self).__init__()
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
        super(SpectralRPR, self).__init__()
        self.backbone = Backbone(config)
        self.regressor_t = PoseRegressor(self.backbone.output_dim*2, 3)
        self.regressor_rot = PoseRegressor(self.backbone.output_dim*2, 4)

    def forward_backbone(self, img):
        return self.backbone(img)

    def forward_regressor_heads(self, latent_query, latent_knns):
        latent_pairs = torch.cat((latent_query, latent_knns), 1)
        rel_ts = self.regressor_t(latent_pairs)
        rel_rots = self.regressor_rot(latent_pairs)
        return rel_ts, rel_rots

    def forward_spectral(self, rel_knn_ts, rel_knn_rots, rel_query_ts, rel_query_rots, batch_size):
        # Place holder until we add here the actual code
        abs_t = torch.ones((batch_size, 3)).to(rel_knn_ts.device)
        abs_rot = torch.ones((batch_size, 4)).to(rel_knn_rots.device)
        return abs_t, abs_rot

    def forward(self, data):
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
        rel_query_ts, rel_query_rots = self.forward_regressor_heads(latent_query, latent_knns)
        ###########################3###################

        # Apply spectral synchornization
        rel_knn_poses = data['knn_rel_poses']
        rel_knn_ts = rel_knn_poses[:, :, :, :3]
        rel_knn_rots = rel_knn_poses[:, :, :, 3:]
        abs_t, abs_rot = self.forward_spectral(rel_knn_ts, rel_knn_rots, rel_query_ts, rel_query_rots, batch_size)

        res = {"abs_poses": torch.cat((abs_t, abs_rot), dim=1),
               "rel_poses": torch.cat((rel_query_ts, rel_query_rots), dim=1)}
        return res




