import torch
from config import configs
from model.model import Model

DEVICE = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    opt = configs()

    model = Model(opt, DEVICE)
