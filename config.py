import argparse
import pickle
from utils import *

"""
配置参数，将存储一份参数到txt文件中，用于实验保存，
另外将存储一份为 params.pkl 文件，在当前目录下，用于 jupyter notebook 调用
(因为 notebook 无法使用 argparse)

jupyter notebook 调用方法：

```Python
with open('./params.pkl', mode='rb') as f:
    opt = pickle.load(f)
```
"""


def configs(save=True, pkl_file='params.pkl'):
 
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--name', type=str, default='experiment_name', help='where to store samples and models')
    parser.add_argument('--gpu_ids', type=str, default='0', help='if there no GPU, auto use cpu')
    parser.add_argument('--dataroot', type=str, default='./datasets/cityscapes/')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
    parser.add_argument('--isTrain', default=True, help='False for test')

    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')

    opt = parser.parse_args()

    # save args to .pkl
    param = open(pkl_file, 'wb')
    pickle.dump(opt, param)

    args = vars(opt)

    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

    # save to the disk
    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    mkdirs(expr_dir)
    if save and not opt.continue_train:
        file_name = os.path.join(expr_dir, 'config.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')

    return opt
