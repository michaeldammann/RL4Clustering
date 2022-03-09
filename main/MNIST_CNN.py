import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class MINST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_part = nn.ModuleList([])
        self.conv_part.extend(self.conv_relu_bn_pool_stack(1, 4, 3))
        self.conv_part.extend(self.conv_relu_bn_pool_stack(4, 8, 3))
        self.conv_part.extend(self.conv_relu_bn_pool_stack(8, 16, 3))
        #self.flatten = nn.Flatten()
        self.fc_part = nn.ModuleList([])
        self.fc_part.extend(self.lin_relu_bn_stack(9*16, 120))
        self.fc_part.extend(self.lin_relu_bn_stack(120, config.FEATURE_DIM))
        self.logits = nn.Linear(config.FEATURE_DIM, 10)


    def conv_relu_bn_pool_stack(self, input_dims, output_dims, filter_size):
        return nn.ModuleList([nn.Conv2d(input_dims, output_dims, filter_size, padding='same'),
                              nn.ReLU(),
                              nn.BatchNorm2d(output_dims),
                              nn.MaxPool2d(2, 2)])

    def lin_relu_bn_stack(self, input_dims, output_dims):
        return nn.ModuleList([nn.Linear(input_dims, output_dims),
                              nn.ReLU(),
                              nn.BatchNorm1d(output_dims)])

    def forward(self, x):
        for conv_layer in self.conv_part:
            x = conv_layer(x)
        x = torch.flatten(x, 1)
        for fc_layer in self.fc_part:
            x = fc_layer(x)
        return self.logits(x)


