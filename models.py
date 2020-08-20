import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import count_params


class CNN(nn.Module):
    def __init__(self, img_size, in_channels=1,
                 conv1_kernel_size_size=9, conv1_out_channels=32,
                 conv2_kernel_size_size=7, conv2_out_channels=64,
                 fc_units=10):
        super().__init__()
        
        self.img_size = img_size
        self.in_channels = in_channels
        self.conv1_kernel_size_size = conv1_kernel_size_size
        self.conv2_kernel_size_size = conv2_kernel_size_size
        self.conv1_out_channels = conv1_out_channels
        self.conv2_out_channels = conv2_out_channels

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, conv1_out_channels, conv1_kernel_size_size),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(conv1_out_channels, conv2_out_channels, conv2_kernel_size_size),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(conv2_out_channels * self.compute_conv_output_size(img_size), fc_units)
        )
    
    def compute_conv_output_size(self, img_size):
        conv1_output_size = (img_size - self.conv1_kernel_size_size + 1)
        pool1_output_size = conv1_output_size // 2
        conv2_output_size = (pool1_output_size - self.conv2_kernel_size_size + 1)
        pool2_output_size = conv2_output_size // 2

        flatten_size = pool2_output_size**2

        return flatten_size

    def forward(self, x):
        return self.layers(x)

    @property
    def num_params(self):
        return count_params(self)


class FCN(nn.Module):
    def __init__(self, img_size, in_channels=1,
                 fc1_units=128, fc2_units=128, fc3_units=10):
        super().__init__()

        self.img_size = img_size
        self.in_channels = in_channels
        self.fc1_units = fc1_units
        self.fc2_units = fc2_units
        self.fc3_units = fc3_units
        
        self.layers = nn.Sequential(
            nn.Linear(img_size**2 * in_channels, fc1_units),
            nn.ReLU(True),
            nn.Linear(fc1_units, fc2_units),
            nn.ReLU(True),
            nn.Linear(fc2_units, fc3_units),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)

        return self.layers(x)

    @property
    def num_params(self):
        return count_params(self)


class SpatialTransformerModule(nn.Module):
    def __init__(self, model_name,
                 img_size, in_channels,
                 conv1_kernel_size=5, conv1_out_channels=20,
                 conv2_kernel_size=5, conv2_out_channels=20,
                 fc_units=6,
                 fc1_units=32, fc2_units=32, fc3_units=6,
                 transform_type='Aff'):
        super().__init__()

        assert model_name in ['ST-CNN', 'ST-FCN'], 'model name must be either ST-CNN or ST-FCN'
        assert transform_type in ['Aff', 'Proj', 'TPS'], 'type of transformation must be one of Aff, Proj, TPS'
        
        self.model_name = model_name
        self.transform_type = transform_type

        self.img_size = img_size
        self.in_channels = in_channels
        self.conv1_kernel_size = conv1_kernel_size
        self.conv2_kernel_size = conv2_kernel_size
        self.conv1_out_channels = conv1_out_channels
        self.conv2_out_channels = conv2_out_channels

        self.conv_out_dim = self.conv2_out_channels * (
            (((self.img_size - self.conv1_kernel_size) + 1) // 2 - self.conv2_kernel_size) + 1
        )**2

        self.fc_units = fc_units

        self.fc1_units = fc1_units
        self.fc2_units = fc2_units
        self.fc3_units = fc3_units

        # --localisation networks --
        if model_name == 'ST-CNN':
            self.loc = nn.Sequential(
                nn.Conv2d(self.in_channels, self.conv1_out_channels, self.conv1_kernel_size),  # (20, 24, 24)
                nn.MaxPool2d(2),  # (20, 12, 12)
                nn.ReLU(True),
                nn.Conv2d(self.conv1_out_channels, self.conv2_out_channels, self.conv2_kernel_size),  # (20, 8, 8)
                nn.ReLU(True),
                nn.Flatten(),
                nn.Linear(self.conv_out_dim, self.fc_units)  # (6)
            )
        elif model_name == 'ST_FCN':
            self.loc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.in_channels * self.img_size**2, self.fc1_units),  # (32)
                nn.ReLU(True),
                nn.Linear(self.fc1_units, self.fc2_units),  # (32)
                nn.ReLU(True),
                nn.Linear(self.fc2_units, self.fc3_units)  # (6)
            )

        self.reset_params()

    def reset_params(self):
        # set bias of the last layer
        old_bias = self.loc[-1].bias.data  # (6,)
        new_bias = torch.randn_like(old_bias) * 0.1
        new_bias = new_bias + torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        self.loc[-1].bias.data = new_bias

    def generate_theta(self, x):
        theta = self.loc(x)
        theta = theta.view((-1, 2, 3))  # (N, 2, 3)

        return theta

    def sample_grid(self, x, theta):
        if self.transform_type == 'Aff':
            grid = F.affine_grid(theta, x.size(), align_corners=False)
            grid_sample = F.grid_sample(x, grid, align_corners=False, padding_mode='border', mode='bilinear')
        elif self.transform_type == 'Proj':
            raise NotImplementedError  # TODO:
        else:
            raise NotImplementedError  # TODO:

        return grid_sample

    def forward(self, x):
        theta = self.generate_theta(x)
        grid_sample = self.sample_grid(x, theta)

        return grid_sample

    @property
    def num_params(self):
        return count_params(self)


class SpatialTransformerNetwork(nn.Module):
    def __init__(self, spatial_transformer, backbone):
        super().__init__()

        self.spatial_transformer = spatial_transformer
        self.backbone = backbone

        self.norm = None
        self.spatial_transformer.register_backward_hook(self.hook)
    
    def transform(self, x):
        return self.spatial_transformer(x)

    def forward(self, x):
        x = self.transform(x)
        y = self.backbone(x)

        return y

    @property
    def num_params(self):
        return self.spatial_transformer.num_params + self.backbone.num_params

    def hook(self, module, grad_in, grad_out):
        # grad_in: (None, torch.Size([256, 28, 28, 2]))
        # grad_out: (torch.Size([256, 1, 28, 28]),)

        norm = torch.sqrt(torch.sum(grad_in[1]**2))
        self.norm = norm.detach().cpu().numpy()
