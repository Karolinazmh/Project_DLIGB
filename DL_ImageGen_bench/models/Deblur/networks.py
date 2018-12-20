import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
###############################################################################
# Functions
###############################################################################


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, gpu_ids=[], use_parallel = True, learn_residual = False, rnn = 'WoRNN', expend_ends_chnl = 'woExpend', depth_wise = 'woDepthWise', shuffle_net = 'woShuffleNet', depth_wise_deconv = 'woDepthWiseDeConv', deconv2upsampel = 'woDeConv2UpSample'):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual = learn_residual, rnn=rnn, expend_ends_chnl = expend_ends_chnl, depth_wise = depth_wise, shuffle_net = shuffle_net, depth_wise_deconv = depth_wise_deconv, deconv2upsampel = deconv2upsampel)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual = learn_residual, rnn=rnn, expend_ends_chnl = expend_ends_chnl, depth_wise = depth_wise, shuffle_net = shuffle_net, depth_wise_deconv = depth_wise_deconv, deconv2upsampel = deconv2upsampel)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual = learn_residual)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual = learn_residual)
    elif which_model_netG == 'resnet_5blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=5, gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual = learn_residual, rnn=rnn, expend_ends_chnl = expend_ends_chnl, depth_wise = depth_wise, shuffle_net = shuffle_net, depth_wise_deconv = depth_wise_deconv, deconv2upsampel = deconv2upsampel)
    elif which_model_netG == 'resnet_3blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3, gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual = learn_residual, rnn=rnn, expend_ends_chnl = expend_ends_chnl, depth_wise = depth_wise, shuffle_net = shuffle_net, depth_wise_deconv = depth_wise_deconv, deconv2upsampel = deconv2upsampel)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, gpu_ids=[], use_parallel = True):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids, use_parallel=use_parallel)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids, use_parallel=use_parallel)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    if use_gpu:
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


# channel shuffle function from K_H_Cheng
def channel_shuffle(x, groups):  # , batch_size, num_channels, height, width):
    batch_size, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batch_size, groups, channels_per_group, height, width)
    # transpose
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batch_size, -1, height, width)
    return x


##############################################################################
# Classes
##############################################################################


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], use_parallel=True, learn_residual=False, rnn='WoRNN',
                 expend_ends_chnl='woExpend', depth_wise='woDepthWise', shuffle_net='woShuffleNet', depth_wise_deconv='woDepthWiseDeConv', deconv2upsampel='woDeConv2UpSample', padding_type='zero'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel
        self.learn_residual = learn_residual
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if expend_ends_chnl == 'woExpend':
            model = [nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3, bias=use_bias),
                     nn.ReLU(True)]
        else:
            model = [nn.Conv2d(input_nc, ngf*2, kernel_size=7, padding=3, bias=use_bias),
                     # norm_layer(ngf), # modified by KaiKang, check performance without BN layer
                     nn.ReLU(True)]

        # if depth_wise == 'wiDepthWise':
        #    model += [nn.ZeroPad2d(1),
        #            nn.Conv2d(ngf*2, int(ngf*2/4), kernel_size=3, padding=0, bias=use_bias),
        #            nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            if i == 0 and expend_ends_chnl == 'wiExpend':
                model += [nn.Conv2d(ngf * mult * 2, ngf * mult * 2, kernel_size=3,
                                    stride=2, padding=1, bias=use_bias),
                          # norm_layer(ngf * mult * 2), # modified by KaiKang, check performance without BN layer
                          nn.ReLU(True)]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                    stride=2, padding=1, bias=use_bias),
                          # norm_layer(ngf * mult * 2), # modified by KaiKang, check performance without BN layer
                          nn.ReLU(True)]

        mult = 2**n_downsampling
        if depth_wise == 'wiDepthWise':
            model += [nn.Conv2d(ngf*mult, int(ngf*mult/4), kernel_size=3, padding=1, bias=use_bias),
                      nn.ReLU(True)]
        if rnn == 'WoRNN':
            for i in range(n_blocks):
                if n_blocks == 5:
                    if i == 0:
                        if depth_wise == 'wiDepthWise' and shuffle_net == 'wiShuffleNet':
                            model += [ResnetBlock_DepthWise_ShuffleNet(int(ngf * mult / 4), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
                        elif depth_wise == 'wiDepthWise':
                            model += [ResnetBlock_DepthWise(int(ngf*mult/4), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
                        else:
                            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
                    else:
                        if depth_wise == 'wiDepthWise' and shuffle_net == 'wiShuffleNet':
                            model += [ResnetBlock_Dilation2_DepthWise_ShuffleNet(int(ngf * mult / 4), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
                        elif depth_wise == 'wiDepthWise':
                            model += [ResnetBlock_Dilation2_DepthWise(int(ngf*mult/4), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
                        else:
                            model += [ResnetBlock_Dilation2(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
                elif n_blocks == 3:
                    if i == 0:
                        model += [ResnetBlock_Dilation2(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
                    elif i == 1:
                        model += [ResnetBlock_Dilation3(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
                    else:
                        model += [ResnetBlock_Dilation4(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
                else:
                    model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        if depth_wise_deconv == 'woDepthWiseDeConv':
            if depth_wise == 'wiDepthWise':
                model += [nn.Conv2d(int(ngf*mult/4), ngf*mult, kernel_size=3, padding=1, bias=use_bias),
                          nn.ReLU(True)]

            for i in range(n_downsampling):
                mult = 2**(n_downsampling - i)
                if deconv2upsampel == 'woDeConv2UpSample':
                    if i == n_downsampling-1 and expend_ends_chnl == 'wiExpend':
                        model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2)*2,
                                                     kernel_size=2, stride=2,
                                                     padding=0, output_padding=0,
                                                     bias=use_bias),
                                  # norm_layer(int(ngf * mult / 2)), # modified by KaiKang, check performance without BN layer
                                  nn.ReLU(True)]
                    else:
                        model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                                     kernel_size=2, stride=2,
                                                     padding=0, output_padding=0,
                                                     bias=use_bias),
                                   # norm_layer(int(ngf * mult / 2)), # modified by KaiKang, check performance without BN layer
                                  nn.ReLU(True)]
                else:
                    if i == n_downsampling-1 and expend_ends_chnl == 'wiExpend':
                        model += [nn.Upsample(scale_factor=2, mode='nearest'),
                                  nn.Conv2d(ngf * mult, int(ngf * mult / 2)*2,
                                                     kernel_size=3, stride=1,
                                                     padding=1,
                                                     bias=use_bias),
                                  # norm_layer(int(ngf * mult / 2)), # modified by KaiKang, check performance without BN layer
                                  nn.ReLU(True)]
                    else:
                        model += [nn.Upsample(scale_factor=2, mode='nearest'),
                                  nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                                     kernel_size=3, stride=1,
                                                     padding=1,
                                                     bias=use_bias),
                                   # norm_layer(int(ngf * mult / 2)), # modified by KaiKang, check performance without BN layer
                                  nn.ReLU(True)]
        else:
            for i in range(n_downsampling):
                mult = 2**(n_downsampling - i)
                if i == n_downsampling-1 and expend_ends_chnl == 'wiExpend':
                    model += [nn.Conv2d(int(ngf*mult/4), ngf * mult, kernel_size=1, bias=use_bias),
                              nn.ReLU(True),
                              nn.ConvTranspose2d(ngf * mult, ngf * mult,
                                                 kernel_size=2, stride=2,
                                                 padding=0, output_padding=0,
                                                 bias=use_bias, groups=ngf * mult),
                              # norm_layer(int(ngf * mult / 2)), # modified by KaiKang, check performance without BN layer
                              nn.ReLU(True),
                              nn.Conv2d(ngf * mult, int(ngf * mult / 2 * 2), kernel_size=1, bias=use_bias)]
                else:
                    model += [nn.Conv2d(int(ngf * mult / 2 *2 / 4), ngf * mult, kernel_size=1, bias=use_bias),
                              nn.ReLU(True),
                              nn.ConvTranspose2d(ngf * mult, ngf * mult,
                                                 kernel_size=2, stride=2,
                                                 padding=0, output_padding=0,
                                                 bias=use_bias, groups=ngf * mult),
                               # norm_layer(int(ngf * mult / 2)), # modified by KaiKang, check performance without BN layer
                              nn.ReLU(True),
                              nn.Conv2d(ngf * mult, int(ngf * mult / 2 / 4), kernel_size=1, bias=use_bias)]

            if expend_ends_chnl == 'wiExpend':
                model += [nn.Conv2d(int(ngf * mult / 2 / 2), ngf*2, kernel_size=3, padding=1, bias=use_bias),
                          nn.ReLU(True)]
            else:
                model += [nn.Conv2d(int(ngf * mult / 2 / 2), ngf, kernel_size=3, padding=1, bias=use_bias),
                          nn.ReLU(True)]

        if expend_ends_chnl == 'woExpend':
            model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3)]
        else:
            model += [nn.Conv2d(ngf*2, output_nc, kernel_size=7, padding=3)]
        # model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor) and self.use_parallel:
            output = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            output = self.model(input)
        if self.learn_residual:
            # output = input + output
            # output = torch.clamp(output,min = -1,max = 1)
            output = output
        return output


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       # norm_layer(dim), # modified by KaiKang, check performance without BN layer
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ZeroPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)]
                        # , norm_layer(dim) # modified by KaiKang, check performance without BN layer


        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Define a resnet block with Dilation = 2
class ResnetBlock_Dilation2(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock_Dilation2, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ZeroPad2d(2)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(2)]
        elif padding_type == 'zero':
            p = 2
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias, dilation=2),
                       # norm_layer(dim), # modified by KaiKang, check performance without BN layer
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ZeroPad2d(2)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(2)]
        elif padding_type == 'zero':
            p = 2
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias, dilation=2)]
                       # , norm_layer(dim) # modified by KaiKang, check performance without BN layer


        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

# Define a resnet block with Dilation = 3
class ResnetBlock_Dilation3(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock_Dilation3, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ZeroPad2d(3)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(3)]
        elif padding_type == 'zero':
            p = 3
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias, dilation=3),
                       # norm_layer(dim), # modified by KaiKang, check performance without BN layer
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ZeroPad2d(3)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(3)]
        elif padding_type == 'zero':
            p = 3
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias, dilation=3)]
                       # , norm_layer(dim) # modified by KaiKang, check performance without BN layer


        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

# Define a resnet block with Dilation = 4
class ResnetBlock_Dilation4(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock_Dilation4, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ZeroPad2d(4)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(4)]
        elif padding_type == 'zero':
            p = 4
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias, dilation=4),
                       # norm_layer(dim), # modified by KaiKang, check performance without BN layer
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ZeroPad2d(4)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(4)]
        elif padding_type == 'zero':
            p = 4
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias, dilation=4)]
                       # , norm_layer(dim) # modified by KaiKang, check performance without BN layer


        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Define a resnet block with Dilation = 2
class ResnetBlock_Dilation2_Split(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock_Dilation2_Split, self).__init__()
        '''self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ZeroPad2d(2)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(2)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias, dilation=2),
                       # norm_layer(dim), # modified by KaiKang, check performance without BN layer
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ZeroPad2d(2)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(2)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias, dilation=2)]
                       # , norm_layer(dim) # modified by KaiKang, check performance without BN layer


        return nn.Sequential(*conv_block)'''

        self.Conv1_Dila1 = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(dim, int(dim/2), kernel_size=3, padding=0, bias=use_bias, dilation=1),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )

        self.Conv1_Dila2 = nn.Sequential(
            nn.ZeroPad2d(2),
            nn.Conv2d(dim, int(dim/2), kernel_size=3, padding=0, bias=use_bias, dilation=2),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )

        self.Conv2_Dila1 = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(dim, int(dim/2), kernel_size=3, padding=0, bias=use_bias, dilation=1)
        )

        self.Conv2_Dila2 = nn.Sequential(
            nn.ZeroPad2d(2),
            nn.Conv2d(dim, int(dim/2), kernel_size=3, padding=0, bias=use_bias, dilation=2)
        )



    def forward(self, x):
        '''out = x + self.conv_block(x)
        return out'''

        x_RB1D1 = self.Conv1_Dila1(x)
        x_RB1D2 = self.Conv1_Dila2(x)
        x_RB1Ct = torch.cat((x_RB1D1, x_RB1D2), dim=1)
        x_RB2D1 = self.Conv2_Dila1(x_RB1Ct)
        x_RB2D2 = self.Conv2_Dila2(x_RB1Ct)
        x_RB2Ct = torch.cat((x_RB2D1, x_RB2D2), dim=1)
        out = x + x_RB2Ct
        return out


# Define a resnet block with Dilation = 2
class ResnetBlock_Dilation2_Split2(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock_Dilation2_Split2, self).__init__()
        '''self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ZeroPad2d(2)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(2)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias, dilation=2),
                       # norm_layer(dim), # modified by KaiKang, check performance without BN layer
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ZeroPad2d(2)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(2)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias, dilation=2)]
                       # , norm_layer(dim) # modified by KaiKang, check performance without BN layer


        return nn.Sequential(*conv_block)'''

        self.Conv1_Dila1 = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(dim, int(dim/2), kernel_size=3, padding=0, bias=use_bias, dilation=1),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )

        self.Conv1_Dila2 = nn.Sequential(
            nn.ZeroPad2d(2),
            nn.Conv2d(dim, int(dim/2), kernel_size=3, padding=0, bias=use_bias, dilation=2),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )

        self.Conv2_Dila2 = nn.Sequential(
            nn.ZeroPad2d(2),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias, dilation=2)
        )


    def forward(self, x):
        '''out = x + self.conv_block(x)
        return out'''

        x_RB1D1 = self.Conv1_Dila1(x)
        x_RB1D2 = self.Conv1_Dila2(x)
        x_RB1Ct = torch.cat((x_RB1D1, x_RB1D2), dim=1)
        x_RB2D2 = self.Conv2_Dila2(x_RB1Ct)
        out = x + x_RB2D2
        return out


class ResnetBlock_DepthWise(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock_DepthWise, self).__init__()

        self.Conv1 = nn.Sequential(
            nn.Conv2d(dim, dim*4, kernel_size=1, padding=0, bias=use_bias),
            nn.ReLU(True),
            # nn.ZeroPad2d(1),
            nn.Conv2d(dim*4, dim*4, kernel_size=3, padding=1, bias=use_bias, groups=dim*4),
            nn.ReLU(True),
            nn.Conv2d(dim*4, dim, kernel_size=1, padding=0, bias=use_bias),
            # nn.Dropout(0.5)
        )

        self.Conv2 = nn.Sequential(
            nn.Conv2d(dim, dim*4, kernel_size=1, padding=0, bias=use_bias),
            nn.ReLU(True),
            # nn.ZeroPad2d(1),
            nn.Conv2d(dim*4, dim*4, kernel_size=3, padding=1, bias=use_bias, groups=dim*4),
            nn.ReLU(True),
            nn.Conv2d(dim*4, dim, kernel_size=1, padding=0, bias=use_bias)
        )

    def forward(self, x):
        x_conv1 = self.Conv1(x) + x
        x_conv2 = self.Conv2(x_conv1) + x_conv1
        out = x + x_conv2
        return out



class ResnetBlock_Dilation2_DepthWise(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock_Dilation2_DepthWise, self).__init__()

        self.Conv1 = nn.Sequential(
            nn.Conv2d(dim, dim*4, kernel_size=1, padding=0, bias=use_bias),
            nn.ReLU(True),
            # nn.ZeroPad2d(2),
            nn.Conv2d(dim*4, dim*4, kernel_size=3, padding=2, bias=use_bias, groups=dim*4, dilation=2),
            nn.ReLU(True),
            nn.Conv2d(dim*4, dim, kernel_size=1, padding=0, bias=use_bias),
            # nn.Dropout(0.5)
        )

        self.Conv2 = nn.Sequential(
            nn.Conv2d(dim, dim*4, kernel_size=1, padding=0, bias=use_bias),
            nn.ReLU(True),
            # nn.ZeroPad2d(2),
            nn.Conv2d(dim*4, dim*4, kernel_size=3, padding=2, bias=use_bias, groups=dim*4, dilation=2),
            nn.ReLU(True),
            nn.Conv2d(dim*4, dim, kernel_size=1, padding=0, bias=use_bias)
        )

    def forward(self, x):
        x_conv1 = self.Conv1(x) + x
        x_conv2 = self.Conv2(x_conv1) + x_conv1
        out = x + x_conv2
        return out


class ResnetBlock_DepthWise_ShuffleNet(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock_DepthWise_ShuffleNet, self).__init__()

        self.Conv1_b1 = nn.Sequential(
            nn.Conv2d(dim, dim*4, kernel_size=1, padding=0, bias=use_bias, groups=4),
            nn.ReLU(True)
        )

        self.Conv1_b2 = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(dim*4, dim*4, kernel_size=3, padding=0, bias=use_bias, groups=dim*4),
            nn.ReLU(True),
            nn.Conv2d(dim*4, dim, kernel_size=1, padding=0, bias=use_bias)
            # nn.Dropout(0.5)
        )

        self.Conv2 = nn.Sequential(
            nn.Conv2d(dim, dim*4, kernel_size=1, padding=0, bias=use_bias),
            nn.ReLU(True),
            nn.ZeroPad2d(1),
            nn.Conv2d(dim*4, dim*4, kernel_size=3, padding=0, bias=use_bias, groups=dim*4),
            nn.ReLU(True),
            nn.Conv2d(dim*4, dim, kernel_size=1, padding=0, bias=use_bias)
        )

    def forward(self, x):
        x_conv1_b1 = self.Conv1_b1(x)
        x_conv1_b1 = channel_shuffle(x_conv1_b1, 4)  # , 8, 128, 64, 64)  # def channel_shuffle(x, groups, batch_size, num_channels, height, width):
        x_conv1_b2 = self.Conv1_b2(x_conv1_b1)
        x_conv1 = x + x_conv1_b2
        x_conv2 = self.Conv2(x_conv1) + x_conv1
        out = x + x_conv2
        return out


class ResnetBlock_Dilation2_DepthWise_ShuffleNet(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock_Dilation2_DepthWise_ShuffleNet, self).__init__()

        self.Conv1_b1 = nn.Sequential(
            nn.Conv2d(dim, dim*4, kernel_size=1, padding=0, bias=use_bias, groups=4),
            nn.ReLU(True)
        )

        self.Conv1_b2 = nn.Sequential(
            nn.ZeroPad2d(2),
            nn.Conv2d(dim*4, dim*4, kernel_size=3, padding=0, bias=use_bias, groups=dim*4, dilation=2),
            nn.ReLU(True),
            nn.Conv2d(dim*4, dim, kernel_size=1, padding=0, bias=use_bias),
            # nn.Dropout(0.5)
        )

        self.Conv2 = nn.Sequential(
            nn.Conv2d(dim, dim*4, kernel_size=1, padding=0, bias=use_bias),
            nn.ReLU(True),
            nn.ZeroPad2d(2),
            nn.Conv2d(dim*4, dim*4, kernel_size=3, padding=0, bias=use_bias, groups=dim*4, dilation=2),
            nn.ReLU(True),
            nn.Conv2d(dim*4, dim, kernel_size=1, padding=0, bias=use_bias)
        )

    def forward(self, x):
        x_conv1_b1 = self.Conv1_b1(x)
        x_conv1_b1 = channel_shuffle(x_conv1_b1, 4)  #, 8, 128, 64, 64)  # def channel_shuffle(x, groups, batch_size, num_channels, height, width):
        x_conv1_b2 = self.Conv1_b2(x_conv1_b1)
        x_conv1 = x + x_conv1_b2
        x_conv2 = self.Conv2(x_conv1) + x_conv1
        out = x + x_conv2
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[], use_parallel = True, learn_residual = False):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel
        self.learn_residual = learn_residual
        # currently support only input_nc == output_nc
        assert(input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor) and self.use_parallel:
            output = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            output = self.model(input)
        if self.learn_residual:
            output = input + output
            output = torch.clamp(output,min = -1,max = 1)
        return output


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([self.model(x), x], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[], use_parallel = True):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor) and self.use_parallel:
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)
