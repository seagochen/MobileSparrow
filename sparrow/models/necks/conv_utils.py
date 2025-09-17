import torch.nn as nn

def conv1x1(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 1, 1, 0, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True)
    )


def conv3x3(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, 1, 1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True)
    )
