# Cloned from: https://github.com/yolish/transposenet and modified to support smoother regression
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import copy


class PoseNet(nn.Module):
    """
    A class to represent a classic pose regressor (PoseNet) with a resnet-101 backbone
    PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization,
    Kendall et al., 2015
    """
    def __init__(self, config):
        """
        Constructor
        :param backbone_path: backbone path to a resnet backbone
        """
        super(PoseNet, self).__init__()

        # Load resnet 101
        backbone = torchvision.models.resnet101(pretrained=False)
        backbone.load_state_dict(torch.load(config.get("backbone")))

        # Remove the classifier heads and pooling
        self.backbone = copy_modules(backbone, 0, -2)

        # Regressor layers
        self.fc_loc = nn.Linear(2048, 1024)
        self.smooth_regression = config.get("smooth_regression")
        if self.smooth_regression:
            output_dim = config.get("output_dim")
            self.fc1 = nn.Linear(1024, output_dim)
            self.fc2 = nn.Linear(1024, output_dim)
            self.fc3 = nn.Linear(1024, output_dim)
            self.fc4 = nn.Linear(1024, output_dim)
            self.fc5 = nn.Linear(1024, output_dim)
            self.fc6 = nn.Linear(1024, output_dim)
            self.fc7 = nn.Linear(1024, output_dim)
        else:
            self.fc1 = nn.Linear(1024, 3)
            self.fc2 = nn.Linear(1024, 4)

        self.dropout = nn.Dropout(p=config.get("dropout"))
        self.avg_pooling_2d = nn.AdaptiveAvgPool2d(1)

        # Initialize FC layers
        for m in list(self.modules()):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)

    def forward_regressor_heads(self, x):
        if self.smooth_regression:
            p1 = self.fc1(x)
            p2 = self.fc2(x)
            p3 = self.fc3(x)
            p4 = self.fc4(x)
            p5 = self.fc5(x)
            p6 = self.fc6(x)
            p7 = self.fc7(x)
            pose = torch.cat([p1, p2, p3, p4, p5, p6, p7], dim=1)
        else:
            p_x = self.fc1(x)
            p_q = self.fc2(x)
            pose = torch.cat((p_x, p_q), dim=1)
        return pose

    def forward(self, data):
        """
        Forward pass
        :param data: (torch.Tensor) dictionary with key-value 'img' -- input image (N X C X H X W)
        :return: (torch.Tensor) dictionary with key-value 'pose' -- 7-dimensional absolute pose for (N X 7)
                                                                 or a smooth representation N X 7 X M
        """
        x = data.get('img')
        x = self.avg_pooling_2d(self.backbone(x))  # N X 3 X 224 X 224 -> Nx2048x7x7 -> Nx2048x1
        x = x.view(x.size(0), -1)  # output shape Nx2048
        x = self.dropout(F.relu(self.fc_loc(x)))
        pose = self.forward_regressor_heads(x)
        return {'pose': pose}


def copy_modules(model, start_idx, end_idx):
    """
    Copy modules from a model
    :param net: (nn.Module) the network to copy
    :param start_idx: (int) index of the module where the copy should start
    :param end_idx: (int) index of the module where the copy should end (exclusive)
    :return: deep copy of submodel
    """
    modules = list(model.children())[start_idx:end_idx]

    # Copy the modules
    sub_model = nn.Sequential(*modules)
    params_orig = model.state_dict()
    params_truncated = sub_model.state_dict()

    # Copy the parameters
    for name, param in params_orig.items():
        if name in params_truncated:
            params_truncated[name].data.copy_(param.data)

    # Load parameters into the architecture
    sub_model.load_state_dict(params_truncated)
    return copy.deepcopy(sub_model)

