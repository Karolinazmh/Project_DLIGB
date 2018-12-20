from os.path import exists, join, basename
from os import remove
from six.moves import urllib
import tarfile
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

from PIL import Image

from .dataset import DatasetFromFolder
from .dataset import DatasetFromEvaFolder


def download_bsd300(dest="./dataset"):
    output_image_dir = join(dest, "BSDS300/images")

    if not exists(output_image_dir):
        url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
        print("downloading url ", url)

        data = urllib.request.urlopen(url)

        file_path = join(dest, basename(url))
        with open(file_path, 'wb') as f:
            f.write(data.read())

        print("Extracting data")
        with tarfile.open(file_path) as tar:
            for item in tar:
                tar.extract(item, dest)

        remove(file_path)

    return output_image_dir


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform(crop_size, upscale_factor):
    return Compose([
        CenterCrop(crop_size),
        Resize(crop_size // upscale_factor, Image.BICUBIC),
        ToTensor(),
    ])


def target_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),
    ])


def full_feature_transform():
    return Compose([
        ToTensor(),
    ])


def get_training_set(upscale_factor, train_dir):
    """
        获取Training dataset

        Arguments:
            upscale_factor (int): SR 图像的scale倍率 \n
            train_dir (str): 训练库的位置

        Returns:
            dataset (class DatasetFromFolder): 训练数据库		
    """

    # root_dir = download_bsd300()
    # train_dir = join(root_dir, "train_aug")
    # if not exists(train_dir):
    #     train_dir = join(root_dir, "train")
    crop_size = calculate_valid_crop_size(128, upscale_factor)

    return DatasetFromFolder(train_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))


def get_test_set(upscale_factor, test_dir):
    """
        获取Validataion dataset (在train的过程中supervise模型性能的数据库)

        Arguments:
            upscale_factor (int): SR 图像的scale倍率 \n
            test_dir (str): 验证库的位置

        Returns:
            dataset (class DatasetFromFolder): 验证数据库		
    """

    # root_dir = download_bsd300()
    # test_dir = join(root_dir, "test")
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(test_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))


def get_eva_set(opt):
    """
        获取Evaluation dataset

        Arguments:
            opt (dict): 参数列表

        Returns:
            dataset (class DatasetFromFolder): 测试数据库		
    """

    return DatasetFromEvaFolder(opt.evadata,
                                input_transform=full_feature_transform(),
                                target_transform=full_feature_transform(),
                                opt=opt)
