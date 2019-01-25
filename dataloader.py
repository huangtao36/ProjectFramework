import torch.utils.data as DATA
import torch
import torchvision.transforms as transforms


class Dataset(DATA.Dataset):
    def __init__(self, opt, img_transform=None):
        super(Dataset, self).__init__()

        self.virtual_data = torch.rand((1000, 3, 128, 128))
        self.size = 1000
    def __getitem__(self, item):
        return self.virtual_data[item]

    def __len__(self):
        return self.size


def loader_data(opt, TrainOrTest='train'):

    img_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    data = Dataset(opt, img_transform=img_transform)

    load_data = DATA.DataLoader(
        data,
        batch_size=opt.batch_size,
        shuffle=opt.shuffle,
        num_workers=opt.num_works
    )

    return load_data
