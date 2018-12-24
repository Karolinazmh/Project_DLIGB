from __future__ import print_function

import os
import sys
from functools import partial, reduce
from math import log10

import torch
import torch.nn as nn
from PIL import Image

import torchvision.transforms as transforms
from apputils.platform_summaries import *
from distiller.data_loggers import PythonLogger, TensorBoardLogger

script_dir = os.path.dirname(__file__)
module_path = os.path.abspath(os.path.join(script_dir, '..', '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

INPUT_CHANNELS = 1
INPUT_SIZE = [1, 540, 960]   # [channel, height, width]

# Distiller loggers
msglogger = apputils.config_pylogger('logging.conf', None)
tflogger = TensorBoardLogger(msglogger.logdir)
tflogger.log_gradients = True
pylogger = PythonLogger(msglogger)

OVERALL_LOSS_KEY = 'Overall Loss'
OBJECTIVE_LOSS_KEY = 'Objective Loss'


class XSRCNNTester(object):
    """
	    SR 神经网络测试过程(C2SRCNN, SRCNN的变体)

        Arguments:
            config (dict): 参数列表 \n
            testing_loader (class DataLoader): 测试集DataLoader

        Examples:
            >>> from models.test_solver import XSRCNNTester
            >>> model = XSRCNNTester(args, test_data_loader)
            >>> model.run(args)
    """
    def __init__(self, config, testing_loader):
        super(XSRCNNTester, self).__init__()
        self.GPU_IN_USE = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.GPU_IN_USE else 'cpu')

        self.model = None
        self.criterion = None
        self.testing_loader = testing_loader
        self.have_target = config.have_target
        self.resume = config.resume
        self.resize = config.resize
        self.sensitivity = config.sensitivity
        self.upscale_factor = config.upscale_factor
        self.model_name = config.model

        if not os.path.exists('./results'):
            os.mkdir('./results')

        self.model_result_path = './results/%s' % self.model_name
        if not os.path.exists(self.model_result_path):
            os.mkdir(self.model_result_path)

        # print opt to logger
        args = vars(config)
        msglogger.info('------------ Options -------------')
        for k, v in sorted(args.items()):
            msglogger.info('%s: %s', k, v)
        msglogger.info('-------------- End ----------------')

    def build_model(self):
        """
	        构建模型, 从已经存储的权重文件.pth还原模型, 并定义评估metric(比如l2 loss)
            SR 应用目前不支持load_dict()方式还原模型, 如有需要, 可以参考models.Deblur.test_solver.build_model

            Examples:
                >>> self.model = torch.load(model_file.pth)    # 从权重文件还原模型
                >>> self.criterion = nn.MSELoss()              # 定义评估metric
        """
        if self.resume:
            with open(self.resume, 'rb') as f:
                self.model = torch.load(f)
        else:
            raise Exception("in test phase, must have resume weights file")

        self.criterion = nn.MSELoss()

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

    def save_tensor2img(self, tensor, img_save_path):
        """
	        将神经网络输出的tensor数据存储到图像文件

            Arguments:
                tensor (tensor): 神经网络的输出tensor \n
                img_save_path (str): 保存图像文件的位置 
        """
        out_img = tensor.cpu()[0].detach().numpy()
        out_img *= 255.0
        out_img = out_img.clip(0, 255)
        out_img = Image.fromarray(np.uint8(out_img[0]), mode='L')

        out_img.save(img_save_path)

        del out_img

    def test(self, model):
        """
	        Test Process, 测试过程

            Arguments:
                model (class Net): 已训练好的模型 

            Examples:
                >>> output = self.model(data)                       # 模型inferecne
                >>> mse = self.criterion(prediction, target)        # 测试集计算模型loss  
                >>> self.save_tensor2img(output, output_save_path)  # 将模型输出存储为图像文件
        """
        avg_psnr = 0
        save_path = './results/%s/test_image' % self.model_name
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        for batch_num, (data, target, img_path) in enumerate(self.testing_loader):

            if not self.have_target:
                del target

            src_img_name = os.path.basename(img_path[0])
            msglogger.info('process image... %s' % src_img_name)
            src_img_name = os.path.splitext(src_img_name)[0]

            if self.resize:
                data = self.img_preprocess(data)  # resize input image size
            data = data.to(self.device)

            save_data_name = src_img_name + '_' + self.model_name + '_data.bmp'
            data_save_path = os.path.join(save_path, save_data_name)
            self.save_tensor2img(data, data_save_path)

            prediction = model(data)
            save_pred_name = src_img_name + '_' + self.model_name + '_pred.bmp'
            pred_save_path = os.path.join(save_path, save_pred_name)
            self.save_tensor2img(prediction, pred_save_path)

            if self.have_target:
                target = target.to(self.device)
                mse = self.criterion(prediction, target)
                psnr = 10 * log10(1 / mse.item())
                avg_psnr += psnr

                save_target_name = src_img_name + '_' + self.model_name + '_target.bmp'
                target_save_path = os.path.join(save_path, save_target_name)
                self.save_tensor2img(target, target_save_path)

                del target

            del data
            del prediction
            torch.cuda.empty_cache()

        msglogger.info('Averaget psnr = %s' % (avg_psnr/len(self.testing_loader)))

        return avg_psnr / len(self.testing_loader)

    def run(self, opt):
        """
            Tester的主函数, 控制test的主流程

            Arguments:
                opt (dict): 参数列表

            Examples:
                >>> from apputils.platform_summaries import *
                >>> self.build_model()          # 构建模型
                >>> self.test(self.model)       # 测试模型
                >>> draw_model_to_file(model, 'arch.png', torch.FloatTensor(1, 3, 128, 128))          # 画模型结构
                >>> sensitivity_analysis(model, 'sense_file.xlsx', test_func, 'element')              # 权重敏感度分析
                >>> sparsity_display(model, 'spars_file.xlsx')                                        # 权重稀疏性分析
                >>> macs_display(model, 'macs_file.xlsx'， torch.FloatTensor(1, 3, 128, 128))         # 模型计算量估计
                >>> transform_to_onnx(model, 'model.onnx'， torch.FloatTensor(1, 3, 128, 128), False) # 将模型和权重存为onnx格式文件
        """
        self.build_model()

        resolution = reduce(lambda x, y: x * y, INPUT_SIZE)
        dummy_input = torch.FloatTensor(resolution).view(1, -1, INPUT_SIZE[1], INPUT_SIZE[2]).to(self.device)

        if opt.test_img:
            self.test(self.model)

        if self.sensitivity:
            test_func = partial(self.test)
            sensitivity_analysis(self.model, opt.sensefile, test_func, self.sensitivity)

        if opt.arch:
            draw_model_to_file(self.model, opt.arch, dummy_input)

        if opt.sparsity:
            sparsity_display(self.model, opt.sparsity)

        if opt.macs:
            macs_display(self.model, opt.macs, dummy_input)

        if opt.toonnx:
            transform_to_onnx(self.model, opt.toonnx, dummy_input, opt.quantize_flag)
