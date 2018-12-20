from __future__ import print_function
from math import log10

import torch
import torchnet.meter as tnt
import time

# Distiller imports
import os
import sys
script_dir = os.path.dirname(__file__)
module_path = os.path.abspath(os.path.join(script_dir, '..', '..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import distiller
import apputils
from distiller.data_loggers import TensorBoardLogger, PythonLogger
from collections import OrderedDict

from models.Deblur.models import create_model
from apputils.Deblur_apputils.visualizer import Visualizer
from apputils.Deblur_apputils.metrics import PSNR, SSIM

INPUT_CHANNELS = 3
INPUT_SIZE = [3, 224, 224]

# Distiller loggers
msglogger = apputils.config_pylogger('logging.conf', None)
tflogger = TensorBoardLogger(msglogger.logdir)
tflogger.log_gradients = True
pylogger = PythonLogger(msglogger)

OVERALL_LOSS_KEY = 'Overall Loss'
OBJECTIVE_LOSS_KEY = 'Objective Loss'


class DeblurGANTrainer(object):
    def __init__(self, config, training_loader):
        super(DeblurGANTrainer, self).__init__()

        self.model = None
        self.visualizer = None
        self.total_steps = 0
        self.epoch_iter = 0
        self.save_dir = os.path.join(config.checkpoints_dir, config.name)

        # self.criterion = None
        # self.optimizer = None
        self.scheduler = None

        self.training_loader = training_loader
        # self.testing_loader = testing_loader

        # distiller arguments
        self.compress = config.compress
        self.trainOn = config.trainOn

        # print opt to logger
        args = vars(config)
        msglogger.info('------------ Options -------------')
        for k, v in sorted(args.items()):
            msglogger.info('%s: %s', k, v)
        msglogger.info('-------------- End ----------------')

    def build_model(self, opt):
        self.model = create_model(opt)

        if opt.resumeG:
            self.model.netG.load_state_dict(torch.load(opt.resumeG))
        if opt.resumeD:
            # To solve the pytorch version issue(0.3.1 save model cannot load by 0.4.0)
            # self.model.netD.load_state_dict(torch.load(opt.resumeD))
            model_dict = torch.load(opt.resumeD)
            model_dict_clone = model_dict.copy()  # We can't mutate while iterating

            for key, value in model_dict_clone.items():
                if key.endswith(('running_mean', 'running_var')):
                    del model_dict[key]

            self.model.netD.load_state_dict(model_dict, False)

        # print model
        msglogger.info('G model:\n\n{0}\n'.format(self.model.netG))
        msglogger.info('D model:\n\n{0}\n'.format(self.model.netD))

    def save(self, epoch):
        save_Dfilename = '%s_whole_net_D.pth' % epoch
        save_Dpath = os.path.join(self.save_dir, save_Dfilename)

        save_Gfilename = '%s_whole_net_G.pth' % epoch
        save_Gpath = os.path.join(self.save_dir, save_Gfilename)

        torch.save(self.model.netD, save_Dpath)
        torch.save(self.model.netG, save_Gpath)
        # print("Checkpoint saved to {}".format(model_out_path))

    def train(self, epoch, compression_scheduler, opt):
        losses = OrderedDict([(OVERALL_LOSS_KEY, tnt.AverageValueMeter()),
                              (OBJECTIVE_LOSS_KEY, tnt.AverageValueMeter())])
        # batch_time = tnt.AverageValueMeter()

        # start_time = time.time()
        for i, data in enumerate(self.training_loader):
            iter_start_time = time.time()

            self.total_steps += opt.batchSize
            self.epoch_iter += opt.batchSize
            self.model.set_input(data)

            if compression_scheduler:
                compression_scheduler.on_minibatch_begin(epoch=epoch, minibatch_id=i,
                                                         minibatches_per_epoch=len(self.training_loader),
                                                         optimizer=self.model.optimizer_G)

            # self.model = self.model.to(self.device)
            self.model.forward()

            # whether train D or not
            if opt.which_model_netD != 'NoD':  # modified by KaiKang, add option "NoD"
                if opt.trainD:
                    for iter_d in range(self.model.criticUpdates):
                        self.model.optimizer_D.zero_grad()
                        self.model.backward_D()
                        self.model.optimizer_D.step()
                else:
                    self.model.loss_D = self.model.discLoss.get_loss(self.model.netD, self.model.real_A,
                                                                     self.model.fake_B, self.model.real_B)

            self.model.optimizer_G.zero_grad()

            loss = self.model.get_G_loss_func(opt)
            losses[OBJECTIVE_LOSS_KEY].add(loss.item())

            if compression_scheduler:
                # Before running the backward phase, we allow the scheduler to modify the loss
                agg_loss = compression_scheduler.before_backward_pass(epoch, minibatch_id=i,
                                                                      minibatches_per_epoch=len(self.training_loader),
                                                                      loss=loss,
                                                                      optimizer=self.model.optimizer_G,
                                                                      return_loss_components=True)
                loss = agg_loss.overall_loss
                losses[OVERALL_LOSS_KEY].add(loss.item())
                for lc in agg_loss.loss_components:
                    if lc.name not in losses:
                        losses[lc.name] = tnt.AverageValueMeter()
                    losses[lc.name].add(lc.value.item())

            self.model.backward_G_func(opt)
            self.model.optimizer_G.step()

            if compression_scheduler:
                compression_scheduler.on_minibatch_end(epoch=epoch, minibatch_id=i,
                                                       minibatches_per_epoch=len(self.training_loader),
                                                       optimizer=self.model.optimizer_G)

            # log push in
            stats_dict = OrderedDict()
            # batch_time.add(time.time() - start_time)
            steps_completed = i + 1
            lr = self.model.optimizer_G.param_groups[0]['lr']

            results = None

            if steps_completed % opt.display_freq == 0:
                results = self.model.get_current_visuals()
                self.visualizer.display_current_results(results, epoch)

            if steps_completed % opt.print_freq == 0:
                if results:
                    psnrMetric = PSNR(results['Restored_Train'], results['Sharp_Train'])
                else:
                    results = self.model.get_current_visuals()
                    psnrMetric = PSNR(results['Restored_Train'], results['Sharp_Train'])

                errors = self.model.get_current_errors(opt)

                for loss_name, meter in losses.items():
                    stats_dict[loss_name] = meter.mean
                stats_dict['LR'] = lr
                stats_dict['sample Time'] = (time.time() - iter_start_time) / opt.batchSize * 1000
                stats_dict['G_GAN'] = self.model.loss_G_GAN.data
                stats_dict['G_L1'] = self.model.loss_G_Content.data
                stats_dict['D_loss'] = self.model.loss_D.data
                stats_dict['psnr'] = psnrMetric
                stats = ('Performance/Training', stats_dict)

                distiller.log_training_progress(stats, self.model.netG.named_parameters(), epoch, steps_completed,
                                                len(self.training_loader), opt.print_freq, [tflogger, pylogger])

                if opt.display_id > 0:
                    self.visualizer.plot_current_errors(epoch, float(self.epoch_iter) / len(self.training_loader),
                                                        opt, errors)

                # t = (time.time() - iter_start_time) / opt.batchSize
                # self.visualizer.print_current_errors(epoch, self.epoch_iter, errors, t)

            start_time = time.time()

        # print("    Average Loss: {:.4f}".format(train_loss / len(self.training_loader)))
        # write.add_scalar('Train/Loss', train_loss / len(self.training_loader), epoch)

    def run(self, opt):
        self.build_model(opt)

        if not self.trainOn:
            self.save('initial')
        else:
            self.visualizer = Visualizer(opt)

            compression_scheduler = None

            if self.compress:
                compression_scheduler = distiller.config.file_config(self.model.netG, self.model.optimizer_G, self.compress)

            self.total_steps = 0
            for epoch in range(opt.epoch_count, opt.niter+opt.niter_decay):
                msglogger.info("\n===> Epoch {} starts:".format(epoch))

                epoch_start_time = time.time()
                self.epoch_iter = 0
                if compression_scheduler:
                    compression_scheduler.on_epoch_begin(epoch)

                # train & validation
                self.train(epoch, compression_scheduler, opt)

                # self.test(self.model, epoch)

                # sparsity logger
                if compression_scheduler:
                    distiller.log_weights_sparsity(self.model.netG, epoch, loggers=[tflogger, pylogger])

                # self.scheduler.step(epoch)

                if epoch % opt.save_epoch_freq == 0:
                    msglogger.info('saving the model at the end of epoch %d, iters %d' % (epoch, self.total_steps))
                    self.model.save('latest', opt)
                    self.model.save(epoch, opt)
                    self.save(epoch)

                if epoch == (opt.niter+opt.niter_decay-1):
                    self.model.save('latest', opt)
                    self.save('latest')

                msglogger.info('End of epoch %d / %d \t Time Taken: %d sec' %
                               (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

                if epoch > opt.niter:
                    self.model.update_learning_rate(opt)

                if compression_scheduler:
                    compression_scheduler.on_epoch_end(epoch, self.model.optimizer_G)
