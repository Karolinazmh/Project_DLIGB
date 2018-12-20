from __future__ import print_function

from models.Deblur.test_solver import DeblurGANTester
from models.test_solver import XSRCNNTester

from options.Deblur_options.test_options import TestOptions as DeblurGANTestOptions
from options.Normal_options.test_options import TestOptions as NormTestOptions

from data.Deblur_data.data_loader import CreateDataLoader as DeblurDataLoader
from torch.utils.data import DataLoader
from data.data import get_eva_set

import sys as _sys

# ===========================================================
# 1. Options Get from different option Form
# ===========================================================
#
if _sys.argv[2] == 'norm':
    args = NormTestOptions().parse()
elif _sys.argv[2] == 'deblurGAN':
    args = DeblurGANTestOptions().parse()

def main():
    """
    模型训练阶段(Testing Phase)的主入口

    Examples:
        >>> from models.test_solver import XSRCNNTester
        >>> from options.Normal_options.test_options import TestOptions as NormTestOptions
        >>> from data.data import get_eva_set
        # Step 1. 参数列表获取(Options Get)
        >>> args = NormTestOptions().parse()
        # Step 2. 数据获取(Dataset Get)
        >>> test_set = get_eva_set(args)
        >>> test_data_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)
        # Step 3. 测试过程(Trainer)
        >>> model = XSRCNNTester(args, test_data_loader)
        >>> model.run(args)
    """

    # ===========================================================
    # 2. Dataset Get
    # ===========================================================
    print('===> Loading datasets')

    if args.optform == 'norm':
        test_set = get_eva_set(args)
        test_data_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)

    elif args.optform == 'deblurGAN':
        test_loader = DeblurDataLoader(args)
        test_data_loader = test_loader.load_data()
    else:
        test_data_loader = None

    # ===========================================================
    # 3. Evaluation
    # ===========================================================
    if args.optform == 'norm':
        model = XSRCNNTester(args, test_data_loader)
        model.run(args)
    elif args.optform == 'deblurGAN':
        model = DeblurGANTester(args, test_data_loader)
        model.run(args)

if __name__ == '__main__':
    main()
