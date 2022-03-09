import torch
import torch.nn as nn
import torch.nn.functional as F
import config


def conv_relu_bn_pool_stack(input_dims, output_dims, filter_size):
    return nn.ModuleList([nn.Conv2d(input_dims, output_dims, filter_size, padding='same'),
                          nn.ReLU(),
                          nn.BatchNorm2d(output_dims),
                          nn.MaxPool2d(2, 2)])

def lin_relu_bn_stack(input_dims, output_dims):
    return nn.ModuleList([nn.Linear(input_dims, output_dims),
                          nn.ReLU(),
                          nn.BatchNorm1d(output_dims)])

def mnist_cnn():
    conv_part = []
    conv_part.extend(conv_relu_bn_pool_stack(1, 4, 3))
    conv_part.extend(conv_relu_bn_pool_stack(4, 8, 3))
    conv_part.extend(conv_relu_bn_pool_stack(8, 16, 3))
    flatten = nn.Flatten()
    fc_part = []
    fc_part.extend(lin_relu_bn_stack(9 * 16, 120))
    fc_part.extend(lin_relu_bn_stack(120, config.FEATURE_DIM))
    logits = nn.Linear(config.FEATURE_DIM, 10)

    neural_net = []
    neural_net.extend(conv_part)
    neural_net.append(flatten)
    neural_net.extend(fc_part)
    neural_net.append(logits)

    return nn.Sequential(*neural_net)