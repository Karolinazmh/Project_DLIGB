from torch.autograd import Variable
from collections import OrderedDict
import apputils.Deblur_apputils.util as util
from .base_model import BaseModel
from . import networks
import torch


class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    def initialize(self, opt):
        assert(not opt.isTrain)
        BaseModel.initialize(self, opt)
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids, False,
                                      opt.learn_residual, opt.rnn, opt.expend_ends_chnl, opt.depthwise, opt.shufflenet, opt.depthwisedeconv)
        # which_epoch = opt.which_epoch
        # self.load_network(self.netG, 'G', which_epoch)
        #
        # print('---------- Networks initialized -------------')
        # networks.print_network(self.netG)
        # print('-----------------------------------------------')

    def set_input(self, input):
        # we need to use single_dataset mode
        input_A = input['A']
        temp = self.input_A.clone()
        temp.resize_(input_A.size()).copy_(input_A)
        self.input_A = temp
        self.image_paths = input['A_paths']

        self.input_B = input['B']

        # del temporal variable to defend memory leak, 2018-10-19 16:27:27 by ckh
        del input_A
        del temp

    def test(self):
        self.real_A = Variable(self.input_A)
        self.fake_B = torch.clamp(self.netG.eval().forward(self.real_A), min=-1, max=1)
        # modified by KaiKang (add ".eval()" after "netG"), turn off dropout for evaluation
        # Add clip [-1, 1] after forward at test phase, 2018-11-08 09:25:28 by CKH

        self.real_B = Variable(self.input_B)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])
