from __future__ import print_function
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torchnet.meter as tnt

from models.C2SRCNN.model import Net
from PIL import Image
from functools import reduce
from functools import partial
import time
import numpy as np

# Distiller imports
import os
import sys
script_dir = os.path.dirname(__file__)
module_path = os.path.abspath(os.path.join(script_dir, '..', '..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import distiller
import apputils
from distiller.data_loggers import TensorBoardLogger, PythonLogger
from collections import OrderedDict

INPUT_CHANNELS = 1
INPUT_SIZE = [1, 224, 224]

# Distiller loggers
msglogger = apputils.config_pylogger('logging.conf', None)
tflogger = TensorBoardLogger(msglogger.logdir)
tflogger.log_gradients = True
pylogger = PythonLogger(msglogger)

OVERALL_LOSS_KEY = 'Overall Loss'
OBJECTIVE_LOSS_KEY = 'Objective Loss'


class C2SRCNNTrainer(object):
    """
	    SR 神经网络训练过程(C2SRCNN, SRCNN的变体)

        Arguments:
            config (dict): 参数列表 \n
            training_loader (class DataLoader): 训练集DataLoader \n
            testing_loader (class DataLoader): 验证集DataLoader

        Examples:
        >>> from models.C2SRCNN.solver import C2SRCNNTrainer
        >>> model = C2SRCNNTrainer(args, training_data_loader, testing_data_loader)
        >>> model.run()
    """

    def __init__(self, config, training_loader, testing_loader):
        super(C2SRCNNTrainer, self).__init__()
        self.GPU_IN_USE = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.GPU_IN_USE else 'cpu')
        self.model = None
        self.lr = config.lr
        self.nEpochs = config.nEpochs
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.resume = config.resume
        self.seed = config.seed
        self.upscale_factor = config.upscale_factor
        self.training_loader = training_loader
        self.testing_loader = testing_loader
        self.savefile = config.save
        self.freq = config.print_freq
        self.trainOn = config.trainOn

        # distiller arguments
        self.compress = config.compress

        # print opt to logger
        args = vars(config)
        msglogger.info('------------ Options -------------')
        for k, v in sorted(args.items()):
            msglogger.info('%s: %s', k, v)
        msglogger.info('-------------- End ----------------')

    def build_model(self):
        """
	        构建模型, 不只是构建模型结构, 还包括定义loss计算的方式, 优化方式(optimizeer), 还有learning rate策略

            Examples:
            >>> from models.C2SRCNN.model import Net
            >>> self.model = Net(num_channels=1, upscale_factor=self.upscale_factor).to(self.device)  # 定义模型结构
            >>> self.criterion = nn.MSELoss() # 定义loss计算方式
            >>> self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr) # 定义优化方式
            >>> self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[500, 1000, 1500, 2000], gamma=0.5) # 定义lr策略
        """

        if self.resume:
            with open(self.resume, 'rb') as f:
                self.model = torch.load(f)
        else:
            self.model = Net(num_channels=1, upscale_factor=self.upscale_factor).to(self.device)
            self.model.weight_init(mean=0.0, std=0.2)
        self.criterion = nn.MSELoss()
        torch.manual_seed(self.seed)

        if self.GPU_IN_USE:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.criterion.cuda()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # lr decay
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[500, 1000, 1500, 2000], gamma=0.5)

        # print model
        msglogger.info('Raw model:\n\n{0}\n'.format(self.model))

    def img_preprocess(self, data, interpolation='bicubic'):
        """
	        图像进入神经网络前的处理, SRCNN模型进入网络前要进行cheap scale up

            Arguments:
                data (tensor): training_loader送进来的样本 \n
                interpolation (str): upscale的方式(one of bicubic/bilinear/nearest) 

            Returns:
                processed_data (tensor): 经过处理后的神经网络输入
        """
        if interpolation == 'bicubic':
            interpolation = Image.BICUBIC
        elif interpolation == 'bilinear':
            interpolation = Image.BILINEAR
        elif interpolation == 'nearest':
            interpolation = Image.NEAREST

        size = list(data.shape)

        if len(size) == 4:
            target_height = int(size[2] * self.upscale_factor)
            target_width = int(size[3] * self.upscale_factor)
            out_data = torch.FloatTensor(size[0], size[1], target_height, target_width)
            for i, img in enumerate(data):
                transform = transforms.Compose([transforms.ToPILImage(),
                                                transforms.Resize((target_height, target_width),
                                                                  interpolation=interpolation),
                                                transforms.ToTensor()])

                out_data[i, :, :, :] = transform(img)
            return out_data
        else:
            target_height = int(size[1] * self.upscale_factor)
            target_width = int(size[2] * self.upscale_factor)
            transform = transforms.Compose([transforms.ToPILImage(),
                                            transforms.Resize((target_height, target_width),
                                                              interpolation=interpolation),
                                            transforms.ToTensor()])
            return transform(data)

    def save(self):
        """
	        保存模型

            用的是torch.save()方式保存模型, inference时只用torch.load()就可以还原模型, 不用重新构建模型结构
            但是要注意, 在inference时, 定义模型结构的model.py文件要保留在trainging phase时相同的位置

            Examples:
            >>> torch.save(self.model, model_out_path)
        """

        model_out_path = self.savefile
        torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def train(self, epoch, compression_scheduler):
        """
        data: [torch.cuda.FloatTensor], 4 batches: [64, 64, 64, 8]
        """
        losses = OrderedDict([(OVERALL_LOSS_KEY, tnt.AverageValueMeter()),
                              (OBJECTIVE_LOSS_KEY, tnt.AverageValueMeter())])
        batch_time = tnt.AverageValueMeter()

        self.model.train()
        start_time = time.time()
        for batch_num, (data, target) in enumerate(self.training_loader):
            data = self.img_preprocess(data)  # resize input image size
            data, target = data.to(self.device), target.to(self.device)

            if compression_scheduler:
                compression_scheduler.on_minibatch_begin(epoch=epoch, minibatch_id=batch_num,
                                                         minibatches_per_epoch=len(self.training_loader),
                                                         optimizer=self.optimizer)

            # self.model = self.model.to(self.device)
            loss = self.criterion(self.model(data), target)

            losses[OBJECTIVE_LOSS_KEY].add(loss.item())

            if compression_scheduler:
                # Before running the backward phase, we allow the scheduler to modify the loss
                agg_loss = compression_scheduler.before_backward_pass(epoch, minibatch_id=batch_num,
                                                                      minibatches_per_epoch=len(self.training_loader),
                                                                      loss=loss,
                                                                      optimizer=self.optimizer,
                                                                      return_loss_components=True)
                loss = agg_loss.overall_loss
                losses[OVERALL_LOSS_KEY].add(loss.item())
                for lc in agg_loss.loss_components:
                    if lc.name not in losses:
                        losses[lc.name] = tnt.AverageValueMeter()
                    losses[lc.name].add(lc.value.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if compression_scheduler:
                compression_scheduler.on_minibatch_end(epoch=epoch, minibatch_id=batch_num,
                                                       minibatches_per_epoch=len(self.training_loader),
                                                       optimizer=self.optimizer)

            # debug
            # for param_name, param in self.model.named_parameters():
            #     print(param_name)
            #     # print(param)

            # self.draw_model_to_file('arch_after_quantize.png')

            # dummy_input_test = torch.rand((1, 1, 5, 5), requires_grad=False).cuda()
            # # dummy_input_test = dummy_input_test * 2 - 1
            # test_drop_res = self.model(dummy_input_test) - dummy_input_test
            # x = dummy_input_test
            # for mod_name, layer in self.model.named_modules():
            #     if not distiller.has_children(layer):
            #         test_input = x
            #         x = layer(test_input)
            #         print('1')

            # progress_bar(batch_num, len(self.training_loader), 'Loss: %.4f' % (train_loss / (batch_num + 1)))

            # msglogger.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.4f} | ms/batch {:5.2f} | loss {:5.2f}'
            #                .format(epoch, batch_num, len(self.training_loader), lr, elapsed * 1000, cur_loss))

            # log push in
            stats_dict = OrderedDict()
            batch_time.add(time.time() - start_time)
            steps_completed = batch_num + 1
            lr = self.optimizer.param_groups[0]['lr']

            if steps_completed % self.freq == 0:
                for loss_name, meter in losses.items():
                    stats_dict[loss_name] = meter.mean
                stats_dict['LR'] = lr
                stats_dict['Batch Time'] = batch_time.mean*1000
                stats = ('Performance/Training', stats_dict)

                distiller.log_training_progress(stats, self.model.named_parameters(), epoch, steps_completed,
                                                len(self.training_loader), self.freq, [tflogger, pylogger])

            start_time = time.time()

        # print("    Average Loss: {:.4f}".format(train_loss / len(self.training_loader)))
        # write.add_scalar('Train/Loss', train_loss / len(self.training_loader), epoch)

    def test(self, model, epoch):
        """
        data: [torch.cuda.FloatTensor], 10 batches: [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        """
        model.eval()
        avg_psnr = 0
        start_time = time.time()

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.testing_loader):
                data = self.img_preprocess(data)  # resize input image size
                data, target = data.to(self.device), target.to(self.device)
                prediction = model(data)
                mse = self.criterion(prediction, target)
                psnr = 10 * log10(1 / mse.item())
                avg_psnr += psnr
                # progress_bar(batch_num, len(self.testing_loader), 'PSNR: %.4f' % (avg_psnr / (batch_num + 1)))

        # print("    Average PSNR: {:.4f} dB".format(avg_psnr / len(self.testing_loader)))
        # write.add_scalar('Val/PSNR', avg_psnr / len(self.testing_loader), epoch)

        msglogger.info('-' * 89)
        msglogger.info('| end of epoch {:3d} | time: {:5.2f}s | valid psnr {:5.3f} | '
                       .format(epoch, (time.time() - start_time), avg_psnr / len(self.testing_loader)))
        msglogger.info('-' * 89)

        stats = ('Performance/Validation/',
                 OrderedDict([('PSNR',  avg_psnr / len(self.testing_loader))]))
        tflogger.log_training_progress(stats, epoch, 0, total=1, freq=1)

        return avg_psnr / len(self.testing_loader)

    def run(self):
        self.build_model()

        if not self.trainOn:
            self.save()
        else:
            compression_scheduler = None

            if self.compress:
                compression_scheduler = distiller.config.file_config(self.model, self.optimizer, self.compress)

            # self.model.load_state_dict(torch.load('./models/C2SRCNN/C2SRCNN_quant_dorefa_nEpoch2500_weight.pth'))
            # torch.save(self.model, './models/C2SRCNN/C2SRCNN_dorefa_nEpoch2500_final.pth')

            for epoch in range(0, self.nEpochs):
                print("\n===> Epoch {} starts:".format(epoch))

                epoch_start_time = time.time()
                if compression_scheduler:
                    compression_scheduler.on_epoch_begin(epoch)

                # train & validation
                # distiller.log_weights_sparsity(self.model, epoch, loggers=[tflogger, pylogger])
                self.train(epoch, compression_scheduler)

                self.test(self.model, epoch)

                # sparsity logger
                distiller.log_weights_sparsity(self.model, epoch, loggers=[tflogger, pylogger])

                self.scheduler.step(epoch)

                if epoch == (self.nEpochs-1):
                    self.save()

                if compression_scheduler:
                    compression_scheduler.on_epoch_end(epoch, self.optimizer)
