import argparse


class TestOptions():
	"""
	
	从命令行获取Testing Phase的参数列表

    """

	def __init__(self):
		self.parser = argparse.ArgumentParser()
		self.initialized = False

	def initialize(self):
		# hyper-parameters
		self.parser.add_argument(
			'--testdata', type=str, default='./dataset/Test/Set29/MR_4x', help='tesing set')
		self.parser.add_argument('--upscale_factor', '-uf', type=int, default=4, help="super resolution upscale factor")
		self.parser.add_argument(
			'--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
		self.parser.add_argument('--model', '-m', type=str, default='fsrcnn', help='choose which model is going to use')
		self.parser.add_argument('--have_target', action='store_true', help='Do we have target img?')
		self.parser.add_argument('--input_mode', type=str, default='rgb', help='input image is rgb(ch=3) or y(ch=1)')
		self.parser.add_argument('--resize', action='store_true', help='do we need resize raw data(if yes, we need preprocess before network it)')

		self.initialized = True

		self.parser.add_argument(
			'--optform', type=str, default='norm', help='use which options form [norm, deblurGAN, SR, NR, MEMC]')
		self.parser.add_argument('--test_img', action='store_true', help='Test on images?')
		self.parser.add_argument('--arch', type=str, default=None, help='architecture file to save')
		self.parser.add_argument(
			'--sense', dest='sensitivity', choices=['element', 'filter', 'channel'],
			help='test the sensitivity of layers to pruning')
		self.parser.add_argument('--sensefile', type=str, help='sensitivity analysis file')
		self.parser.add_argument('--macs', type=str, default=None, help='Macs estimate file to save')
		self.parser.add_argument('--sparsity', type=str, default=None, help='Sparsity of model')
		self.parser.add_argument('--toonnx', type=str, default=None, help='transform onnx file to save')
		self.parser.add_argument('--quantize_flag', action='store_true', help='is model have benn quantized?')

	def parse(self):
		"""
		命令行参数解析

 		Returns:
			self.opt (dict): 以字典形式存储的参数列表

		Examples:
			>>> from options.Normal_options.test_options import TestOptions as NormTestOptions
			>>> args = NormTestOptions().parse()
		"""

		if not self.initialized:
			self.initialize()
		self.opt = self.parser.parse_args()
		return self.opt
