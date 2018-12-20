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
    """Draw a language model graph to a PNG file.

    Caveat: the PNG that is produced has some problems, which we suspect are due to
    PyTorch issues related to RNN ONNX export.
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
    df_sparsity = distiller.weights_sparsity_summary(model)
    # Remove these two columns which contains uninteresting values
    df_sparsity = df_sparsity.drop(['Cols (%)', 'Rows (%)'], axis=1)

    df_sparsity.to_csv(sparsity_file)


def macs_display(model, macs_file, dummy_input):
    df_macs = distiller.model_performance_summary(model, dummy_input, 1)

    df_macs.to_csv(macs_file)

    netG_MACs = df_macs['MACs'].sum()
    msglogger.info("netG MACs: " + "{:,}".format(int(netG_MACs)))


def transform_to_onnx(model, onnx_file, dummy_input, quantize_flag):

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
