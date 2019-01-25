import os
import visdom
import numpy as np
from PIL import Image
import torch


class Visualizer(object):
    def __init__(self, server, port, env='default'):
        """
        :param server: your server IP
        :param port: your visdom port
        :param env: set an Environment for you experimentation
        """
        self.vis = visdom.Visdom(
            server=server,
            port=port,
            env=env,
            raise_exceptions=True)
        self.index = {}

    def plot_many_stack(self, data_dic, split=False):
        """
        use like this:
        vis.plot_many_stack({'train_loss': loss_meter.value()[0],
                             'test_loss': loss_meter1.value()[0]},
                            split=False)
        :param data_dic:
        :param split:
        :return:
        """
        name = list(data_dic.keys())
        name_total = " ".join(name)

        x = self.index.get(name_total, 0)
        val = list(data_dic.values())

        if len(val) == 1:
            y = np.array(val)
        else:
            y = np.array(val).reshape(-1, len(val))

        if len(val) is not 1 and split:
            for i in range(len(val)):
                self.vis.line(
                    Y=y[:, i], X=np.ones(y[:, i].shape) * x,
                    win=str(name_total[i]),
                    opts=dict(legend=[name[i]],
                              title=str(name[i]),
                              xlabel='epoch',
                              ylabel='loss'),
                    update=None if x == 0 else 'append'
                )
        else:

            self.vis.line(
                Y=y, X=np.ones(y.shape) * x,
                win=str(name_total),
                opts=dict(legend=name,
                          title=name_total,
                          xlabel='epoch',
                          ylabel='loss'),
                update=None if x == 0 else 'append'
            )
        self.index[name_total] = x + 1


def print_current_losses(txt_file, epoch, iter, losses_str):
    message = '(epoch: %d, iters: %d) ' % (epoch, iter)
    message += losses_str

    print(message)
    with open(txt_file, "a") as log_file:
        log_file.write('%s\n' % message)


def mkdirs(paths):
    """
    :param paths: str or str-list
    :return: None
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)
    else:
        if not os.path.exists(paths):
            os.makedirs(paths)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)
