from __future__ import print_function

import argparse
import sys as _sys

import h5py
from torch.utils.data import DataLoader

from data.data import get_training_set, get_validation_set
from data.Deblur_data.data_loader import CreateDataLoader as DeblurDataLoader
from models.C2SRCNN.solver import C2SRCNNTrainer
from models.Deblur.solver import DeblurGANTrainer
from options.Deblur_options.train_options import \
    TrainOptions as DeblurGANTrainOptions
from options.Normal_options.train_options import \
    TrainOptions as NormTrainOption

# ===========================================================
# 1. Options Get from different option Form
# ===========================================================
#
if _sys.argv[2] == 'norm':
    args = NormTrainOption().parse()
elif _sys.argv[2] == 'deblurGAN':
    args = DeblurGANTrainOptions().parse()


def main():
    """
    模型训练阶段(Training Phase)的主入口

    Examples:
        >>> from models.C2SRCNN.solver import C2SRCNNTrainer
        >>> from options.Normal_options.train_options import TrainOptions as NormTrainOption
        >>> from data.data import get_training_set, get_validation_set
        # Step 1. 参数列表获取(Options Get)
        >>> args = DeblurGANTrainOptions().parse()
        # Step 2. 数据获取(Dataset Get)
        >>> train_set = get_training_set(args.upscale_factor, args.traindata)
        >>> validate_set = get_validation_set(args.upscale_factor, args.validata)
        >>> training_data_loader = DataLoader(dataset=train_set, batch_size=args.batchSize, shuffle=True)
        >>> validation_data_loader = DataLoader(dataset=validate_set, batch_size=args.valiBatchSize, shuffle=False)
        # Step 3. 训练过程(Trainer)
        >>> model = C2SRCNNTrainer(args, training_data_loader, validation_data_loader)
        >>> model.run()
    """

    # ===========================================================
    # 2. Dataset Get
    # ===========================================================
    print('===> Loading datasets')

    if args.optform == 'norm':
        train_set = get_training_set(args.upscale_factor, args.traindata)
        validate_set = get_validation_set(args.upscale_factor, args.validata)
        training_data_loader = DataLoader(dataset=train_set, batch_size=args.batchSize, shuffle=True)
        validation_data_loader = DataLoader(dataset=validate_set, batch_size=args.valiBatchSize, shuffle=False)
    elif args.optform == 'deblurGAN':
        train_loader = DeblurDataLoader(args)
        training_data_loader = train_loader.load_data()
        validation_data_loader = None
    else:
        train_set = get_training_set(args.upscale_factor, args.traindata)
        validate_set = get_validation_set(args.upscale_factor, args.validata)
        training_data_loader = DataLoader(dataset=train_set, batch_size=args.batchSize, shuffle=True)
        validation_data_loader = DataLoader(dataset=validate_set, batch_size=args.valiBatchSize, shuffle=False)

    # ===========================================================
    # 3. Training
    # ===========================================================
    if args.optform == 'norm':
        if args.model == 'c2srcnn':
            model = C2SRCNNTrainer(args, training_data_loader, validation_data_loader)
        else:
            raise Exception("the model does not exist")
        model.run()
    elif args.optform == 'deblurGAN':
        model = DeblurGANTrainer(args, training_data_loader)
        model.run(args)


if __name__ == '__main__':
    main()
