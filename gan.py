
import os
import os.path as osp
import json
from pickletools import optimize
import torch
import pickle
import logging
import numpy as np
from model.discriminators import SNTemporalPatchGANDiscriminator
from model.simvp_model import SimVP
from model.losses import AdversarialLoss, GANLoss
from tqdm import tqdm
from API import *
from utils import *
from PIL import Image as im
import time

class Exp:
    def __init__(self, args):
        super(Exp, self).__init__()
        self.args = args
        self.config = self.args.__dict__
        self.device = self._acquire_device()

        self._preparation()
        print_log(output_namespace(self.args))

        self._get_data()
        self._select_optimizer()
        self._select_criterion()
        self._try_resume_trained_model(self.args.resume_path)

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device('cuda:{}'.format(0))
            print_log('Use GPU: {}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print_log('Use CPU')
        return device

    def _preparation(self):
        # seed
        set_seed(self.args.seed)
        # log and checkpoint
        self.path = osp.join(self.args.res_dir, self.args.ex_name)
        check_dir(self.path)

        self.checkpoints_path = osp.join(self.path, 'checkpoints')
        check_dir(self.checkpoints_path)

        sv_param = osp.join(self.path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(self.args.__dict__, file_obj)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, filename=osp.join(self.path, 'log.log'),
                            filemode='a', format='%(asctime)s - %(message)s')
        # prepare data
        self._get_data()
        # build the model
        self._build_model()

        self.lambda_adv = self.args.lambda_adv

    def _build_model(self):
        args = self.args
        self.model = SimVP(tuple(args.in_shape), args.hid_S,
                           args.hid_T, args.N_S, args.N_T).to(self.device)
        self.discriminator = SNTemporalPatchGANDiscriminator(args.image_channels).to(self.device)
    
    def _try_resume_trained_model(self, path):
        if path:
            if os.path.exists(path):
                print('resuming')
                checkpoint = torch.load(path)
                self.model.load_state_dict(checkpoint['GENERATOR_STATE_DICT'])
                self.optimizer.load_state_dict(checkpoint['GENERATOR_OPTIMIZER_STATE_DICT'])
                self.discriminator.load_state_dict(checkpoint['DISCRIMINATOR_STATE_DICT'])
                self.discriminator_optimizer.load_state_dict(checkpoint['DISCRIMINATOR_OPTIMIZER_STATE_DICT'])
            else:
                raise (ValueError('Resume path does not exist'))

    def _get_data(self):
        config = self.args.__dict__
        self.train_loader, self.vali_loader, self.test_loader, self.data_mean, self.data_std = load_data(**config)
        self.vali_loader = self.test_loader if self.vali_loader is None else self.vali_loader

    def _select_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.lr)
        self.discriminator_optimizer = torch.optim.SGD(
            self.discriminator.parameters(),
            lr=self.args.lr_D, momentum=0.9
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.args.lr, steps_per_epoch=len(self.train_loader), epochs=self.args.epochs)
        return self.optimizer

    def _select_criterion(self):
        self.criterion = torch.nn.MSELoss()
        self.criterion_adv = GANLoss(device=self.device, gan_type='lsgan').to(self.device)

    def _save(self, name=''):
        torch.save({
            'GENERATOR_STATE_DICT': self.model.state_dict(),
            'GENERATOR_OPTIMIZER_STATE_DICT': self.optimizer.state_dict(),
            'DISCRIMINATOR_STATE_DICT': self.discriminator.state_dict(),
            'DISCRIMINATOR_OPTIMIZER_STATE_DICT': self.discriminator_optimizer.state_dict()
        }, os.path.join(
            self.checkpoints_path, name + '.pth'))
        state = self.scheduler.state_dict()
        fw = open(os.path.join(self.checkpoints_path, name + '.pkl'), 'wb')
        pickle.dump(state, fw)

    def _predict(self, batch_x):
        if self.args.aft_seq_length == self.args.pre_seq_length:
            pred_y = self.model(batch_x)
        elif self.args.aft_seq_length < self.args.pre_seq_length:
            pred_y = self.model(batch_x)
            pred_y = pred_y[:, :self.args.aft_seq_length]
        elif self.args.aft_seq_length > self.args.pre_seq_length:
            pred_y = []
            d = self.args.aft_seq_length // self.args.pre_seq_length
            m = self.args.aft_seq_length % self.args.pre_seq_length
            
            cur_seq = batch_x.clone()
            for _ in range(d):
                cur_seq = self.model(cur_seq)
                pred_y.append(cur_seq)

            if m != 0:
                cur_seq = self.model(cur_seq)
                pred_y.append(cur_seq[:, :m])
            
            pred_y = torch.cat(pred_y, dim=1)
        return pred_y

    def train(self, args):
        config = args.__dict__
        recorder = Recorder(verbose=True)

        for epoch in range(config['epochs']):
            start_time = time.time()
            train_loss = []
            non_gan_loss = []
            gan_loss = []
            discriminator_loss = []

            self.model.train()
            self.discriminator.train()
            
            train_pbar = tqdm(self.train_loader)

            for batch_x, batch_y in train_pbar:
                
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                # -----------------
                #  Train Generator
                # -----------------

                self.optimizer.zero_grad()

                pred_y = self._predict(batch_x)
                loss = self.criterion(pred_y, batch_y)
                non_gan_loss.append(loss.item())

                adv_loss = self.criterion_adv(self.discriminator(pred_y), True, is_disc=False)
                gan_loss.append(adv_loss.item())
                adv_loss = adv_loss * self.lambda_adv
                loss += adv_loss
                train_loss.append(loss.item())

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()


                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.discriminator_optimizer.zero_grad()
                d_real = self.discriminator(batch_y)

                loss_d_real = self.criterion_adv(d_real, True, is_disc=True) * 0.5
                loss_d_real.backward()

                d_fake = self.discriminator(pred_y.detach())
                loss_d_fake = self.criterion_adv(d_fake, False, is_disc=True) * 0.5
                loss_d_fake.backward()

                d_loss = loss_d_real + loss_d_fake
                discriminator_loss.append(d_loss.item())

                self.discriminator_optimizer.step()

                train_pbar.set_description('train loss: {0:.4f} - NonGAN loss: {1:.4f} - Raw GAN loss: {2:.9f} - Discriminator loss: {3:.9f}'.format(loss.item(), non_gan_loss[-1], gan_loss[-1], discriminator_loss[-1]))


            train_loss = np.average(train_loss)
            non_gan_loss = np.average(non_gan_loss)
            gan_loss = np.average(gan_loss)
            discriminator_loss = np.average(discriminator_loss)

            if epoch % args.log_step == 0:
                with torch.no_grad():
                    vali_loss = self.vali(self.vali_loader, epoch)
                    self.interpolate(epoch + 1)
                print_log("Epoch: {0} | Train Loss: {1:.4f} - NonGAN loss: {2:.4f} - GAN loss: {3:.9f} - Discriminator loss: {4:.9f} Vali Loss: {5:.4f} | Take {6:.4f} seconds\n".format(
                    epoch + 1, train_loss, non_gan_loss, gan_loss, discriminator_loss, vali_loss, time.time() - start_time))

                recorder(vali_loss, self.model, self.optimizer, self.discriminator, self.discriminator_optimizer , self.path)

            if epoch % args.save_epoch_freq == 0:
                self._save(name=str(epoch + 1))

        best_model_path = self.path + '/' + 'checkpoint.pth'
        self._try_resume_trained_model(best_model_path)
        return self.model

    def vali(self, vali_loader, epoch=0):
        self.model.eval()
        preds_lst, trues_lst, total_loss = [], [], []
        vali_pbar = tqdm(vali_loader)
        for i, (batch_x, batch_y) in enumerate(vali_pbar):
            if i * batch_x.shape[0] > 1000:
                break

            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            pred_y = self.model(batch_x)
            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                 pred_y, batch_y], [preds_lst, trues_lst]))

            loss = self.criterion(pred_y, batch_y)
            vali_pbar.set_description(
                'vali loss: {:.4f}'.format(loss.mean().item()))
            total_loss.append(loss.mean().item())

        total_loss = np.average(total_loss)
        preds = np.concatenate(preds_lst, axis=0)
        trues = np.concatenate(trues_lst, axis=0)

        mse, mae, ssim, psnr = metric(preds, trues, vali_loader.dataset.mean, vali_loader.dataset.std, True)
        print_log('vali mse:{:.4f}, mae:{:.4f}, ssim:{:.4f}, psnr:{:.4f}'.format(mse, mae, ssim, psnr))
        self.model.train()
        return total_loss

    def test(self, args):
        self.model.eval()
        inputs_lst, trues_lst, preds_lst = [], [], []
        for batch_x, batch_y in self.test_loader:
            pred_y = self.model(batch_x.to(self.device))
            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                 batch_x, batch_y, pred_y], [inputs_lst, trues_lst, preds_lst]))

        inputs, trues, preds = map(lambda data: np.concatenate(
            data, axis=0), [inputs_lst, trues_lst, preds_lst])

        folder_path = self.path+'/results/{}/sv/'.format(args.ex_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mse, mae, ssim, psnr = metric(preds, trues, self.test_loader.dataset.mean, self.test_loader.dataset.std, True)
        print_log('mse:{:.4f}, mae:{:.4f}, ssim:{:.4f}, psnr:{:.4f}'.format(mse, mae, ssim, psnr))

        for np_data in ['inputs', 'trues', 'preds']:
            np.save(osp.join(folder_path, np_data + '.npy'), vars()[np_data])
        return mse

    # TODO: Enhance it for passing fixed input image
    def interpolate(self, sub_folder):
        self.model.eval()
        inputs_lst, trues_lst, preds_lst = [], [], []
        for batch_x, batch_y in self.test_loader:
            pred_y = self.model(batch_x.to(self.device))
            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                 batch_x, batch_y, pred_y], [inputs_lst, trues_lst, preds_lst]))
            break
        
        inputs, trues, preds = map(lambda data: np.concatenate(
            data, axis=0), [inputs_lst, trues_lst, preds_lst])
        print(preds.shape)

        folder_path = self.path+f'/interpolation/{sub_folder}/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        for index,pred in enumerate(preds[0]):
            data = im.fromarray(np.uint8(np.squeeze(np.array(pred).transpose(1,2,0)) * 255))
            data.save(os.path.join(folder_path,'pred_'+ str(index) + '.png'))

        for index,pred in enumerate(inputs[0]):
            data = im.fromarray(np.uint8(np.squeeze(np.array(pred).transpose(1,2,0)) * 255))
            
            data.save(os.path.join(folder_path,'input_'+ str(index) + '.png'))

        for index,pred in enumerate(trues[0]):
            data = im.fromarray(np.uint8(np.squeeze(np.array(pred).transpose(1,2,0)) * 255))
            data.save(os.path.join(folder_path,'trues_'+ str(index) + '.png'))