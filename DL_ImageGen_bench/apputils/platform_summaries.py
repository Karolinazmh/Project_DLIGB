import distiller
import apputils
import numpy as np
import torch.nn as nn
from distiller.quantization.q_utils import *
from distiller.quantization.clipped_linear import ClippedLinearQuantization

import logging
msglogger = logging.getLogger(__name__)


class ClippedOnly(nn.Module):
    def __init__(self, num_bits, clip_val, inplace=False):
        super(ClippedOnly, self).__init__()
        self.num_bits = num_bits
        self.clip_val = clip_val
        self.inplace = inplace

    def forward(self, input):
        input = clamp(input, 0, self.clip_val, self.inplace)
        return input

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return '{0}(num_bits={1}, clip_val={2}{3})'.format(self.__class__.__name__, self.num_bits, self.clip_val,
                                                           inplace_str)


def draw_model_to_file(model, png_fname, dummy_input):
    """
        将模型结构存储到png图像文件

        Arguments:
            model (class Net):      已训练好的模型 \n
            png_fname (str):        图像文件名称和位置 \n
            dummy_input (tensor):   和网络输入相同维度的dummy数据, for example, torch.FloatTensor(1, 3, 128, 128), [batch, channel, height, width]

        Examples:
            >>> from apputils.platform_summaries import *
            >>> draw_model_to_file(model, 'arch.png', torch.FloatTensor(1, 3, 128, 128))
    """
    try:
        g = apputils.SummaryGraph(model, dummy_input)
        apputils.draw_model_to_file(g, png_fname)
        msglogger.info("Network PNG image generation completed")

    except FileNotFoundError as e:
        msglogger.info("An error has occured while generating the network PNG image.")
        msglogger.info("Please check that you have graphviz installed.")
        msglogger.info("\t$ sudo apt-get install graphviz")
        raise e


def sensitivity_analysis(model, sensitivity_file, test_func, sense_type):
    """
        分析网络权重的敏感度, 就是剪枝后的performance loss

        Arguments:
            model (class Net):          已训练好的模型 \n
            sensitivity_file (str):     存储敏感度分析结果的文件名称和位置 \n
            test_func (func):           在Tester中定义的Test Process, 因为要得到performance, 需要每次剪枝后inference一次model \n
            sense_type (str):           敏感度分析的维度(one of 'element'/'filter'/'channel')

        Examples:
            >>> from apputils.platform_summaries import *
            >>> sensitivity_analysis(model, 'sense_file.xlsx', test_func, 'element')
    """
    msglogger.info("Running sensitivity tests")
    # test_func = partial(self.test, opt=opt)
    which_params = [param_name for param_name, _ in model.named_parameters()]
    sensitivity = distiller.perform_sensitivity_analysis_sr(model,
                                                            net_params=which_params,
                                                            sparsities=np.arange(0.0, 0.95, 0.05),
                                                            test_func=test_func,
                                                            group=sense_type)
    distiller.sensitivities_to_csv(sensitivity, sensitivity_file)


def sparsity_display(model, sparsity_file):
    """
        分析网络权重的稀疏性, 就是权重中有多少0

        Arguments:
            model (class Net):          已训练好的模型 \n
            sparsity_file (str):     存储稀疏性分析结果的文件名称和位置

        Examples:
            >>> from apputils.platform_summaries import *
            >>> sparsity_display(model, 'spars_file.xlsx')
    """
    df_sparsity = distiller.weights_sparsity_summary(model)
    # Remove these two columns which contains uninteresting values
    df_sparsity = df_sparsity.drop(['Cols (%)', 'Rows (%)'], axis=1)

    df_sparsity.to_csv(sparsity_file)


def macs_display(model, macs_file, dummy_input):
    """
        分析网络的计算量MACs

        Arguments:
            model (class Net):          已训练好的模型 \n
            macs_file (str):            存储计算量结果的文件名称和位置 \n
            dummy_input (tensor):       和网络输入相同维度的dummy数据, for example, torch.FloatTensor(1, 3, 128, 128), [batch, channel, height, width]

        Examples:
            >>> from apputils.platform_summaries import *
            >>> macs_display(model, 'macs_file.xlsx'， torch.FloatTensor(1, 3, 128, 128))
    """
    df_macs = distiller.model_performance_summary(model, dummy_input, 1)

    df_macs.to_csv(macs_file)

    netG_MACs = df_macs['MACs'].sum()
    msglogger.info("netG MACs: " + "{:,}".format(int(netG_MACs)))


def transform_to_onnx(model, onnx_file, dummy_input, quantize_flag):
    """
        将网络模型和权重存储为onnx格式

        Arguments:
            model (class Net):          已训练好的模型 \n
            onnx_file (str):            onnx文件名称和位置 \n
            dummy_input (tensor):       和网络输入相同维度的dummy数据, for example, torch.FloatTensor(1, 3, 128, 128), [batch, channel, height, width] \n
            quantize_flag (bool):       模型是否为量化后模型, 如果是, 需要在存onnx模型时将ClippedLinearQuantization module替换为ClippedOnly module

        Examples:
            >>> from apputils.platform_summaries import *
            >>> transform_to_onnx(model, 'model.onnx'， torch.FloatTensor(1, 3, 128, 128), False)
    """
    if quantize_flag:
        replace_quantize_module(model)

    # Export the model
    torch_out = torch.onnx.export(model,  # model being run
                                  dummy_input,  # model input (or a tuple for multiple inputs)
                                  onnx_file,  # where to save the model (can be a file or file-like object)
                                  export_params=True,  # store the trained parameter weights inside the model file
                                  verbose=True)  # print the network in onnx file
    return torch_out


def replace_quantize_module(model):
    for name, module in model.named_children():
        if isinstance(module, ClippedLinearQuantization):
            setattr(model, name,
                    ClippedOnly(num_bits=module.num_bits, clip_val=module.clip_val, inplace=module.inplace))

        if distiller.has_children(module):
            replace_quantize_module(module)
