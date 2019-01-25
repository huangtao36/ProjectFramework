import torch
from config import configs
from model.model import Model
from dataloader import loader_data
from utils import *
from torchnet import meter


DEVICE = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    opt = configs()
    experiment_dir = os.path.join(opt.checkpoints_dir, opt.name)
    mkdirs(experiment_dir)

    opt.display_server = 'http://110.65.102.123'
    opt.display_port = 1128
    opt.display_env = 'check'
    vis = Visualizer(server=opt.display_server,
                     port=opt.display_port,
                     env=opt.display_env)

    loss_meter = meter.AverageValueMeter()
    loss_file = os.path.join(experiment_dir, 'loss.txt')
    if os.path.exists(loss_file):
        os.remove(loss_file)
        print("Delete the obsolete loss files: %s!" % loss_file)

    data = loader_data(opt, TrainOrTest='train')
    model = Model(opt, DEVICE)
    model.initial()

    for epoch in range(opt.epoch):
        loss_meter.reset()
        for step, dataitem in enumerate(data):
            model.set_data(dataitem)
            model.optimize_parameters()

            return_dic = model.get_return()
            loss = return_dic['loss']

            # save and show loss
            loss_meter.add(loss.item())
            loss_str = 'mse_loss: %.3f' % loss.item()
            print_current_losses(txt_file=loss_file, epoch=epoch,
                                 iter=step, losses_str=loss_str)

        # visdom show loss plot line
        vis.plot_many_stack({'train_loss': loss_meter.mean}, split=False)

        # save model
        model.save(epoch)
