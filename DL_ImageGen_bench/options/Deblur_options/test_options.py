from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--how_many', type=int, default=5000, help='how many test images to run')
        self.parser.add_argument('--rnn', type=str, default='WoRNN', help='WoRNN, without RNN module ; WiRNN, with RNN module')
        self.parser.add_argument('--expend_ends_chnl', type=str, default='woExpend', help='woExpend, origional architecture; wiExpend, expent both ends channels with * 2')
        self.parser.add_argument('--depthwise', type=str, default='woDepthWise', help='woDepthWise, origional architecture; wiDepthWise, introduce depth-wise design compact model size')
        self.parser.add_argument('--shufflenet', type=str, default='woShuffleNet', help='woShuffleNet, origional architecture; wiShuffleNet, introduce shuffle-net V1 design compact model size')
        self.parser.add_argument('--depthwisedeconv', type=str, default='woDepthWiseDeConv', help='woDepthWiseDeConv, origional deconvolution layer; wiDepthWiseDeConv, use depth wise desig for deconvolution layer')
        self.isTrain = False
        # add compression parameters in test options 2018-10-18 10:45:23 CKH
        self.parser.add_argument('--test_img', action='store_true', help='Test on images?')
        self.parser.add_argument('--targetroot', type=str, default='./dataset/Deblur/')
        self.parser.add_argument('--arch', type=str, default=None, help='architecture file to save')
        self.parser.add_argument('--sense', dest='sensitivity', choices=['element', 'filter', 'channel'],
                                 help='test the sensitivity of layers to pruning')
        self.parser.add_argument('--sensefile', type=str, help='sensitivity analysis file')
        self.parser.add_argument('--macs', type=str, default=None, help='Macs evaluation file to save')
        self.parser.add_argument('--sparsity', type=str, default=None, help='Sparsity of model')
        self.parser.add_argument('--wholemodel', action='store_true', help='weight file include architecture?')
        self.parser.add_argument('--resumeG', type=str, help='trained model file')
        self.parser.add_argument('--psnr', action='store_true', help='calculate psnr metric?')
        self.parser.add_argument('--toonnx', type=str, default=None, help='transform onnx file to save')
        self.parser.add_argument('--quantize_flag', action='store_true', help='is model have benn quantized?')

