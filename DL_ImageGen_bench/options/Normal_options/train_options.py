import argparse


class TrainOptions():
	"""
	
	从命令行获取Training Phase的参数列表

    """

	def __init__(self):
		self.parser = argparse.ArgumentParser()
		self.initialized = False

	def initialize(self):
		# hyper-parameters
		self.parser.add_argument('--batchSize', type=int, default=32, help='training batch size')
		self.parser.add_argument('--valiBatchSize', type=int, default=1, help='validation batch size')
		self.parser.add_argument('--nEpochs', type=int, default=20, help='number of epochs to train for')
		self.parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.01')
		self.parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
		self.parser.add_argument('--print-freq', type=int, default=1, help='training log print frequency')
		self.parser.add_argument(
			'--traindata', type=str, default='./dataset/BSDS300/images/train_aug', help='training dataset path')
		self.parser.add_argument(
			'--validata', type=str, default='./dataset/BSDS300/images/test', help='validation dataset path')

		# model configuration
		self.parser.add_argument('--upscale_factor', '-uf', type=int, default=4, help="super resolution upscale factor")
		self.parser.add_argument('--save', type=str, default='checkpoint.pth.tar', help='path to save the final model')
		self.parser.add_argument(
			'--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
		self.parser.add_argument('--model', '-m', type=str, default='fsrcnn', help='choose which model is going to use')
		self.parser.add_argument('--trainOn', action='store_true', help='need train or just build model')
		self.initialized = True

		# distiller configuration
		self.parser.add_argument(
			'--optform', type=str, default='norm', help='use which options form [norm, deblurGAN, SR, NR, MEMC]')
		self.parser.add_argument(
			'--compress', dest='compress', type=str, nargs='?', action='store',
			help='configuration file for pruning the model (default is to use hard-coded schedule)')

	def parse(self):
		"""
		命令行参数解析

		Returns:
			self.opt (dict): 以字典形式存储的参数列表

		Examples:
			>>> from options.Normal_options.train_options import TrainOptions as NormTrainOption
			>>> args = NormTrainOption().parse()
		"""
		if not self.initialized:
			self.initialize()
		self.opt = self.parser.parse_args()
		return self.opt
