import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def define_network(device=torch.device('cpu')):
    net = Network()

    # USE Multi GPU
    # device_ids = [1, 0] if device.index == 1 else [0, 1]
    # if torch.cuda.device_count() > 1 and torch.cuda.is_available():
    #     net = nn.DataParallel(net, device_ids=device_ids)

    net.to(device)
    net.apply(weights_init)
    return net


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        pass

    def forward(self, *input):

        return None
