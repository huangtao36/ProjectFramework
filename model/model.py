import torch
import torch.nn as nn
from model.network import define_network
from model.model_util import *


class Model(nn.Module):
    def __init__(self, opt, devide=torch.device('cpu')):
        super(Model, self).__init__()
        self.opt = opt
        self.device = devide

        self.save_path = os.path.join(opt.checkpoints_dir, opt.name,
                                      'model')
        self.net = define_network(device=self.device)
        if not opt.isTrain or opt.continue_train:
            load_network(self.net, 'net', 'latest', self.save_path)

        self.optimizer = torch.optim.Adam(
            list(self.net.parameters()), lr=self.opt.lr,
            betas=(self.opt.beta1, 0.999))

    def forward(self, *input):
        pass

    def save(self, which_epoch, save_path=None):

        # self.save_path = os.path.join(
        #     self.opt.checkpoints_dir, self.name, 'model')
        if save_path is None:
            save_path = self.save_path

        print("epoch: {} -- Save network to {}.".format(which_epoch, save_path))
        save_network(self.net, 'net', which_epoch, save_path,
                     devcie=self.device)
        save_network(self.net, 'net', 'latest', save_path,
                     devcie=self.device)
