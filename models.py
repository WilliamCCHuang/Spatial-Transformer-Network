import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import count_params


class BaseCNNModel(nn.Module):
    def __init__(self, img_size, in_planes=1, hidden_planes=32, out_planes=64):
        # the number of filters shold be around 32~64
        assert 32 <= hidden_planes <= 64, 'number of filters in conv1 must in the range of 32~64'
        assert 32 <= out_planes <= 64, 'number of filters in conv1 must in the range of 32~64'
        
        self.hidden_planes = hidden_planes
        self.out_planes = out_planes

        self.conv1 = nn.Conv2d(in_planes, self.hidden_planes, 9)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(self.hidden_planes, self.out_planes, 7)
        self.pool2 = nn.MaxPool2d(2)
        self.cls = nn.Linear(self.out_planes * self.cal_conv_output_size(img_size), 10)

        self.act = nn.ReLU(True)
    
    def cal_conv_output_size(self, img_size):
        conv1_output_size = (img_size - 9 + 1)
        pool1_output_size = conv1_output_size // 2
        conv2_output_size = (pool1_output_size - 7 + 1)
        pool2_output_size = conv2_output_size // 2

        flatten_size = pool2_output_size**2

        return flatten_size

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        y = self.cls(x)

        return y

    def num_params(self):
        return count_params(self)


class BaseFCNModel(nn.Module):
    def __init__(self, img_size, in_planes=1, fc1_unit=128, fc2_unit=256):
        super().__init__()

        # the number of unit should be around 128~256
        assert 128 <= fc1_unit <= 256, 'number of unit in fc1 should around 128~256'
        assert 128 <= fc2_unit <= 256, 'number of unit in fc2 should around 128~256'

        self.fc1_unit = fc1_unit
        self.fc2_unit = fc2_unit
        
        self.fc1 = nn.Linear(img_size**2 * in_planes, self.fc1_unit)
        self.fc2 = nn.Linear(self.fc1_unit, self.fc2_unit)
        self.cls = nn.Linear(self.fc2_unit, 10)
        self.act = nn.ReLU(True)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        y = self.cls(x)

        return y

    def num_params(self):
        return count_params(self)


class BaseSTN(nn.Module):
    def __init__(self, model_name,
                 img_size, in_planes,
                 conv1_kernel=5, conv2_kernel=5,
                 conv1_outdim=20, conv2_outdim=20,
                 theta_row=2, theta_col=3,
                 fc_outdim=1, fc1_outdim=32, fc2_outdim=32, fc3_outdim=1,
                 transform_type='Aff'):

        assert model_name in ['ST-CNN', 'ST-FCN'], 'model name must be either ST-CNN or ST-FCN'
        assert transform_type in ['Aff', 'Proj', 'TPS'], 'type of transformation must be one of Aff, Proj, TPS'
        
        self.model_name = model_name
        self.transform_type = transform_type

        self.img_size = img_size
        self.in_planes = in_planes
        self.conv1_kernel = conv1_kernel
        self.conv2_kernel = conv2_kernel
        self.conv1_outdim = conv1_outdim
        self.conv2_outdim = conv2_outdim

        self.conv_out_dim = self.conv2_outdim * ((((self.img_size - self.conv1_kernel) + 1) // 2 - self.conv2_kernel) + 1)**2
        
        self.theta_row = theta_row
        self.theta_col = theta_col
        self.register_buffer('cos_matrix', torch.tensor([[1., 0, 0],
                                                         [0, 1., 0]], requires_grad=False).unsqueeze(0))  # (1,2,3)
        self.register_buffer('sin_matrix', torch.tensor([[0, -1., 0],
                                                         [1., 0, 0]], requires_grad=False).unsqueeze(0))  # (1,2,3)

        self.fc_outdim = fc_outdim

        self.fc1_outdim = fc1_outdim
        self.fc2_outdim = fc2_outdim
        self.fc3_outdim = fc3_outdim

        # --localisation networks --
        if model_name == 'ST-CNN':
            self.loc = nn.Sequential(
                nn.Conv2d(self.in_planes, self.conv1_outdim, self.conv1_kernel),  # (20, 24, 24)
                nn.MaxPool2d(2),  # (20, 12, 12)
                nn.ReLU(True),
                nn.Conv2d(self.conv1_outdim, self.conv2_outdim, self.conv2_kernel),  # (20, 8, 8)
                nn.ReLU(True),
                nn.Flatten(),
                nn.Linear(self.conv_out_dim, self.fc_outdim)  # (6)
            )
        elif model_name == 'ST_FCN':
            self.loc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.in_planes * self.img_size**2, self.fc1_outdim),  # (32)
                nn.ReLU(True),
                nn.Linear(self.fc1_outdim, self.fc2_outdim),  # (32)
                nn.ReLU(True),
                nn.Linear(self.fc2_outdim, self.fc3_outdim)  # (6)
            )

    def generate_theta(self, x):
        theta = self.loc(x)
        theta = theta.unsqueeze(-1)  # (N, 1, 1)
        theta = torch.cos(theta) * self.cos_matrix + torch.sin(theta) * self.sin_matrix

        return theta

    def transform(self, x, theta):
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
        grid_sample = self.transform(x, theta)

        return grid_sample

    def num_params(self):
        return count_params(self)
    
    def hook_fn_backward(module, grad_input, grad_output):
        # TODO
        raise NotImplementedError


class STModel(nn.Module):
    def __init__(self, base_stn, base_nn_model):
        self.base_stn = base_stn
        self.base_nn_model = base_nn_model

        self.norm = None
        self.base_stn.register_backward_hook(self.hook)
    
    def transform(self, x):
        return self.base_stn(x)

    def forward(self, x):
        x = self.transform(x)
        y = self.base_nn_model(x)

        return y

    def hook(self, module, grad_in, grad_out):
        assert isinstance(grad_in, tuple)

        norm = torch.sqrt(sum(sum(p.detach().cpu().numpy()**2) for p in grad_in))

        self.norm = norm


class CNNModel(nn.Module):
    # TODO: the number of filiters are b/t 32~64
    def __init__(self):
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, 9),  # (32, 20, 20)
            nn.MaxPool2d(2),  # (32, 10, 10)
            nn.ReLU(True),
            nn.Conv2d(32, 64, 7),  # (64, 4, 4)
            nn.MaxPool2d(2),  # (64, 2, 2)
            nn.ReLU(True)
        )
        
        self.cls = nn.Linear(64 * 2 * 2, 10)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        y = self.cls(x)

        return y


class FCNModel(nn.Module):
    # TODO: the number of units per layer are 128~256
    def __init__(self):
        self.layers = nn.Sequential(
            nn.Linear(1 * 28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        y = self.layers(x)

        return y
