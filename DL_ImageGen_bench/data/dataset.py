import torch.utils.data as data

from os import listdir
from os.path import join
from PIL import Image


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y


def load_y_img(filepath):
    img = Image.open(filepath)
    return img


class DatasetFromFolder(data.Dataset):
    """
        从Folder获取dataset(Training Phase)

        Arguments:
            image_dir (str): 数据文件夹的位置 \n
            input_transform (func): 输入图像的前处理操作(crop/resize, 转换为tensor类型) \n
            target_transform (func): 目标图像的前处理操作(crop/resize, 转换为tensor类型) \n		
    """
    def __init__(self, image_dir, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
            在batch中获取单个样本对
            用PIL将图像读入, 经过input_transform和target_transform进行必要的前处理并转换为tensor类型

            Arguments:
                index (int): batch中的sample id \n

            Returns:
                input_image (tensor): 神经网络的输入tensor图像
                target (tensor): 神经网络的目标tensor图像
        """
        input_image = load_img(self.image_filenames[index])
        target = input_image.copy()
        if self.input_transform:
            input_image = self.input_transform(input_image)
        if self.target_transform:
            target = self.target_transform(target)

        return input_image, target

    def __len__(self):
        return len(self.image_filenames)


class DatasetFromEvaFolder(data.Dataset):
    """
        从Folder获取dataset(Evaluation Phase)

        Arguments:
            image_dir (str): 数据文件夹的位置 \n
            input_transform (func): 输入图像的前处理操作(crop/resize, 转换为tensor类型) \n
            target_transform (func): 目标图像的前处理操作(crop/resize, 转换为tensor类型) \n		
    """
    def __init__(self, image_dir, input_transform=None, target_transform=None, opt=None):
        super(DatasetFromEvaFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]

        self.input_transform = input_transform
        self.target_transform = target_transform

        self.have_target = opt.have_target
        self.input_mode = opt.input_mode

        self.resize = opt.resize
        self.upscale = opt.upscale_factor

    def __getitem__(self, index):
        """
            在batch中获取单个样本对
            用PIL将图像读入, 经过input_transform和target_transform进行必要的前处理并转换为tensor类型

            Arguments:
                index (int): batch中的sample id \n

            Returns:
                input_image (tensor): 神经网络的输入tensor图像
                target (tensor): 神经网络的目标tensor图像
        """
        if self.input_mode == 'rgb':
            input_image = load_img(self.image_filenames[index])
        elif self.input_mode == 'y':
            input_image = load_y_img(self.image_filenames[index])
        else:
            input_image = load_img(self.image_filenames[index])

        target = input_image.copy()

        if self.resize:
            input_image = input_image.resize((int(input_image.size[0] / self.upscale),
                                              int(input_image.size[1] / self.upscale)),
                                             Image.BICUBIC)

            target = target.crop([0, 0, (int(target.size[0] / self.upscale)) * self.upscale,
                                  (int(target.size[1] / self.upscale)) * self.upscale])

        if self.input_transform:
            input_image = self.input_transform(input_image)
        if self.target_transform:
            target = self.target_transform(target)

        return input_image, target, self.image_filenames[index]

    def __len__(self):
        return len(self.image_filenames)
