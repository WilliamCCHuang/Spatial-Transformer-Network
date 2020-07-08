import torch
import torch.nn as nn


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
        assert max(x.size()) == 28, 'input img size should be 28x28'

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
        assert max(input.size()) == 28, 'input img size should be 28x28'

        x = x.view(x.size(0), -1)
        y = self.layers(x)

        return y


class STModel(nn.Module):
    def __init__(self, base_stn, base_nn_model):
        self.base_stn = base_stn
        self.base_nn_model = base_nn_model

        self.norm = None
        self.base_stn.register_backward_hook(self.hook)
        
    def forward(self, x):
        x = self.base_stn(x)
        y = self.base_nn_model(x)

        return y

    def hook(self, module, grad_in, grad_out):
        assert isinstance(grad_in, tuple)

        norm = torch.sqrt(sum(sum(p.detach().cpu().numpy()**2) for p in grad_in))

        self.norm = norm
