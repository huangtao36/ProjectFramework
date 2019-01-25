import torch
import torch.nn as nn
from model.network import define_network
from model.model_util import *
from utils import *

class Model(nn.Module):
    def __init__(self, opt, devide=torch.device('cpu')):
        super(Model, self).__init__()
        self.opt = opt
        self.device = devide

    def initial(self):
        self.save_path = os.path.join(self.opt.checkpoints_dir,
                                      self.opt.name,
                                      'model')

        self.criterion = nn.MSELoss()
        self.net = define_network(device=self.device)
        if not self.opt.isTrain or self.opt.continue_train:
            load_network(self.net, 'net', 'latest', self.save_path)

        self.optimizer = torch.optim.Adam(
            list(self.net.parameters()), lr=self.opt.lr,
            betas=(self.opt.beta1, 0.999))

    def set_data(self, input):
        self.input = input.to(self.device)
        self.target = torch.rand((self.input.shape)).to(self.device)

    def get_return(self):
        dic = {
            'loss': self.loss,
        }
        return dic

    def forward(self, ):
        self.output = self.net(self.input)

    def backward(self):
        self.loss = self.criterion(self.output, self.target)

    def optimize_parameters(self):
        # forward
        self.forward()
        # backward
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

        """
        如果有多个网络需要分别优化，需先设另外一个requires_grad = False:
        如： self.set_requires_grad([self.netDA], False)
        优化self.netDA时再设为True:
        """

    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def save(self, which_epoch, save_path=None):

        # self.save_path = os.path.join(
        #     self.opt.checkpoints_dir, self.name, 'model')
        if save_path is None:
            save_path = self.save_path
            mkdirs(save_path)

        print("epoch: {} -- Save network to {}.".format(which_epoch, save_path))
        save_network(self.net, 'net', which_epoch, save_path,
                     devcie=self.device)
        save_network(self.net, 'net', 'latest', save_path,
                     devcie=self.device)
