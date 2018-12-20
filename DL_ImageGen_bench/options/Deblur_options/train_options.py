from .base_options import BaseOptions


class TrainOptions(BaseOptions):
	def initialize(self):
		BaseOptions.initialize(self)
		self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
		self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
		self.parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
		self.parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
		self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
		self.parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
		self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
		self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
		self.parser.add_argument('--niter', type=int, default=150, help='# of iter at starting learning rate')
		self.parser.add_argument('--niter_decay', type=int, default=150, help='# of iter to linearly decay learning rate to zero')
		self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
		self.parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
		self.parser.add_argument('--lambda_A', type=float, default=100.0, help='weight for cycle loss (A -> B -> A)')
		self.parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
		self.parser.add_argument('--identity', type=float, default=0.0, help='use identity mapping. Setting identity other than 1 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set optidentity = 0.1')
		self.parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
		self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
		self.parser.add_argument('--rnn', type=str, default='WoRNN', help='WoRNN, without RNN module ; WiRNN, with RNN module')
		self.parser.add_argument('--expend_ends_chnl', type=str, default='woExpend', help='woExpend, origional architecture; wiExpend, expent both ends channels with * 2')
		self.parser.add_argument('--depthwise', type=str, default='woDepthWise', help='woDepthWise, origional architecture; wiDepthWise, introduce depth-wise design compact model size')
		self.parser.add_argument('--shufflenet', type=str, default='woShuffleNet', help='woShuffleNet, origional architecture; wiShuffleNet, introduce shuffle-net V1 design compact model size')
		self.parser.add_argument('--depthwisedeconv', type=str, default='woDepthWiseDeConv', help='woDepthWiseDeConv, origional deconvolution layer; wiDepthWiseDeConv, use depth wise desig for deconvolution layer')
		self.parser.add_argument('--deconv2upsample', type=str, default='woDeConv2UpSample', help='woDeConv2UpSample, origional deconvolution layer; wiDeConv2UpSample, use upsample + convolution instead deconvolution')
		# add resume to support fine-tune training 2018-10-17 11:27:55 CKH
		self.parser.add_argument('--resumeG', type=str, default=None, help='fine-tune start G model weight')
		self.parser.add_argument('--resumeD', type=str, default=None, help='fine-tune start D model weight')
		self.parser.add_argument('--trainD', action='store_true', help='whether train D or not')
		self.parser.add_argument('--Dloops', type=int, default=1, help='the times train D in one batch loop')
		self.parser.add_argument('--trainOn', action='store_true', help='need train or just build model')
		self.isTrain = True
