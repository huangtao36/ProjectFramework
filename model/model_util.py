import torch
import os


def save_network(network, network_label, epoch_label, save_path,
                 devcie=torch.device('cpu')):
    save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
    save_file = os.path.join(save_path, save_filename)
    torch.save(network.cpu().state_dict(), save_file)
    network.to(devcie)


def load_network(network, network_label, epoch_label, save_dir):
    save_filename = '%s_net_%s.pth' % (epoch_label, network_label)

    save_path = os.path.join(save_dir, save_filename)
    if not os.path.isfile(save_path):
        print('%s not exists yet!' % save_path)
        if network_label == 'G':
            raise ('Generator must exist!')
    else:
        # network.load_state_dict(torch.load(save_path))
        try:
            network.load_state_dict(torch.load(save_path))
        except:
            pretrained_dict = torch.load(save_path)
            model_dict = network.state_dict()
            try:
                pretrained_dict = {k: v for k, v in pretrained_dict.items()
                                   if k in model_dict}
                network.load_state_dict(pretrained_dict)
            except:
                print(
                    'Pretrained network %s has fewer layers; The following are not initialized:' % network_label)
                for k, v in pretrained_dict.items():
                    if v.size() == model_dict[k].size():
                        model_dict[k] = v

                if sys.version_info >= (3, 0):  # python3
                    not_initialized = set()
                else:
                    from sets import Set        # python2
                    not_initialized = Set()

                for k, v in model_dict.items():
                    if k not in pretrained_dict or v.size() != \
                            pretrained_dict[k].size():
                        not_initialized.add(k.split('.')[0])

                print(sorted(not_initialized))
                network.load_state_dict(model_dict)