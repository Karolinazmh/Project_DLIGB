from __future__ import print_function
from math import log10

import torch

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
from distiller.data_loggers import TensorBoardLogger, PythonLogger

from models.Deblur.models import create_model
from apputils.Deblur_apputils.metrics import PSNR
from apputils.Deblur_apputils.visualizer import Visualizer
from apputils.Deblur_apputils import html
from apputils.platform_summaries import *

INPUT_CHANNELS = 3
INPUT_SIZE = [3, 1080, 960]  # [channel, height, width]

# Distiller loggers
msglogger = apputils.config_pylogger('logging.conf', None)
tflogger = TensorBoardLogger(msglogger.logdir)
tflogger.log_gradients = True
pylogger = PythonLogger(msglogger)

OVERALL_LOSS_KEY = 'Overall Loss'
OBJECTIVE_LOSS_KEY = 'Objective Loss'


class DeblurGANTester(object):
    def __init__(self, config, testing_loader):
        super(DeblurGANTester, self).__init__()
        self.GPU_IN_USE = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.GPU_IN_USE else 'cpu')

        self.model = None
        self.testing_loader = testing_loader
        self.visualizer = None
        self.webpage = None
        self.psnr = config.psnr
        self.sensitivity = config.sensitivity

        # print opt to logger
        args = vars(config)
        msglogger.info('------------ Options -------------')
        for k, v in sorted(args.items()):
            msglogger.info('%s: %s', k, v)
        msglogger.info('-------------- End ----------------')

    def build_model(self, opt):
        self.model = create_model(opt)

        if opt.wholemodel:
            self.model.netG = torch.load(opt.resumeG)
        else:
            if opt.resumeG:
                try:
                    self.model.netG.load_state_dict(torch.load(opt.resumeG))
                except FileNotFoundError as e:
                    msglogger.info("model G weight file not find, please check")
                    raise e

        # print model
        msglogger.info('G model:\n\n{0}\n'.format(self.model.netG))

    def test(self, model, opt):
        avg_psnr = 0

        for i, data in enumerate(self.testing_loader):
            self.model = create_model(opt)
            self.model.netG = model
            self.model.set_input(data)
            self.model.test()
            visuals = self.model.get_current_visuals()
            img_path = self.model.get_image_paths()
            msglogger.info('process image... %s' % img_path)
            self.visualizer.save_images(self.webpage, visuals, img_path)

            if self.psnr:
                avg_psnr += PSNR(visuals['fake_B'], visuals['real_B'])

            del data
            del visuals
            del self.model
            torch.cuda.empty_cache()

        msglogger.info('Averaget psnr = %s' % (avg_psnr/len(self.testing_loader)))

        return avg_psnr / len(self.testing_loader)

    def run(self, opt):
        self.build_model(opt)

        self.visualizer = Visualizer(opt)
        web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
        self.webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' %
                                 (opt.name, opt.phase, opt.which_epoch))

        resolution = reduce(lambda x, y: x * y, INPUT_SIZE)
        dummy_input = torch.FloatTensor(resolution).view(1, -1, INPUT_SIZE[1], INPUT_SIZE[2]).to(self.device)

        if opt.test_img:
            self.test(self.model.netG, opt)

        if self.sensitivity:
            test_func = partial(self.test, opt=opt)
            sensitivity_analysis(self.model.netG, opt.sensefile, test_func, self.sensitivity)

        if opt.arch:
            draw_model_to_file(self.model.netG, opt.arch, dummy_input)

        if opt.sparsity:
            sparsity_display(self.model.netG, opt.sparsity)

        if opt.macs:
            macs_display(self.model.netG, opt.macs, dummy_input)

        if opt.toonnx:
            transform_to_onnx(self.model.netG, opt.toonnx, dummy_input, opt.quantize_flag)
