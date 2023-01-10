
import os
import os.path as osp
import json
import torch
import pickle
import logging
import numpy as np
import time

from torchvision.utils import save_image, make_grid
from model import SimVP
from tqdm import tqdm
from API import *
from utils import *


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

        self.build_tensorboard()

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

    def _build_model(self):
        args = self.args
        self.model = SimVP(tuple(args.in_shape), args.hid_S,
                           args.hid_T, args.N_S, args.N_T).to(self.device)

        path = os.path.join(args.model_save_path, f'{args.pretrained_model}_G.pth')
        if path and os.path.exists(path):
            print(f'resuming model at {path}')
            self.model.load_state_dict(torch.load(path))

    def _get_data(self):
        config = self.args.__dict__
        self.train_loader, self.vali_loader, self.test_loader, self.data_mean, self.data_std = load_data(**config)
        self.vali_loader = self.test_loader if self.vali_loader is None else self.vali_loader

    def _select_optimizer(self):
        args = self.args
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.lr)

        path = os.path.join(args.model_save_path, f'{args.pretrained_model}_G_optimizer.pth')
        if path and os.path.exists(path):
            print(f'resuming optimizer at {path}')
            self.optimizer.load_state_dict(torch.load(path))

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.args.lr, steps_per_epoch=len(self.train_loader), epochs=self.args.epochs)
        return self.optimizer

    def _select_criterion(self):
        self.criterion = torch.nn.MSELoss()

    def _save(self, epoch=''):
        torch.save(self.model.state_dict(), os.path.join(
            self.checkpoints_path, f'{epoch}_G.pth'))
        torch.save(self.optimizer.state_dict(), os.path.join(
            self.checkpoints_path, f'{epoch}_G_optimizer.pth'))
        state = self.scheduler.state_dict()
        fw = open(os.path.join(self.checkpoints_path, f'{epoch}.pkl'), 'wb')
        pickle.dump(state, fw)

    def _predict(self, batch_x):
        if self.args.aft_seq_length == self.args.pre_seq_length:
            pred_y = self.model(batch_x)
        elif self.args.aft_seq_length < self.args.pre_seq_length:
            pred_y = self.model(batch_x)
            pred_y = pred_y[:, :self.args.aft_seq_length]
        elif self.args.aft_seq_length > self.args.pre_seq_length:
            print('should here')
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

        # Start with trained model
        if self.args.pretrained_model:
            start = int(self.args.pretrained_model) + 1
        else:
            start = 1

        print_log(f'{"=" * 30} \nStart training from epoch {start}...')

        for epoch in range(start, config['epochs']):
            start_time = time.time()
            train_loss = []
            self.model.train()
            train_pbar = tqdm(self.train_loader)

            for batch_x, batch_y in train_pbar:
                self.optimizer.zero_grad()
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred_y = self._predict(batch_x)

                loss = self.criterion(pred_y, batch_y)
                train_loss.append(loss.item())
                train_pbar.set_description('train loss: {:.4f}'.format(loss.item()))

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            train_loss = np.average(train_loss)

            if epoch == start or epoch % args.log_step == 0:
                with torch.no_grad():
                    vali_loss = self.vali(self.vali_loader)
                log_str= "Epoch: {0} | Train Loss: {1:.4f} Vali Loss: {2:.4f} | Take {3:.4f} seconds\n".format(
                    epoch, train_loss, vali_loss, time.time() - start_time)
                print_log(log_str)

                write_log(self.writer, log_str, epoch, train_loss, vali_loss)

                recorder(vali_loss, self.model, self.path)

            # Sample images
            if epoch == start or epoch % args.sample_epoch == 0:
                self.generate_samples(epoch)

            if epoch == start or epoch % args.save_epoch_freq == 0:
                self._save(epoch=str(epoch))

        best_model_path = self.path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def vali(self, vali_loader):
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

    def build_tensorboard(self):
        from tensorboardX import SummaryWriter
        # from logger import Logger
        # self.logger = Logger(self.log_path)
        check_dir(self.args.log_path)
        self.writer = SummaryWriter(log_dir=self.args.log_path)
    
    @torch.no_grad()
    def generate_samples(self, epoch):
        self.model.eval()

        # TODO: improve this one
        for batch_x, batch_y in self.test_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            pred_y = self._predict(batch_x.to(self.device))
            break

        batch_x = batch_x[0]
        batch_y = batch_y[0]
        pred_y = pred_y[0]

        outputs_and_expectations = torch.cat((pred_y, batch_y), 0)

        path_to_epoch = os.path.join(self.args.sample_path, str(epoch))

        check_dir(path_to_epoch)

        self.writer.add_image(f"inputs", make_grid(batch_x.data, nrow=self.args.pre_seq_length), epoch)
        self.writer.add_image(f"outputs", make_grid(pred_y.data, nrow=self.args.aft_seq_length), epoch)
        self.writer.add_image(f"expected", make_grid(batch_y.data, nrow=self.args.aft_seq_length), epoch)
        save_image(batch_x.data, os.path.join(path_to_epoch, "inputs.png"), nrow=self.args.pre_seq_length)
        save_image(outputs_and_expectations.data, os.path.join(path_to_epoch, "outputs_and_expectations.png"), nrow=self.args.aft_seq_length)
        self.model.train()
    
    def normalize_generated_images(self, batch_images, **kwargs):
        return make_grid(batch_images.mul(255).add_(0.5).clamp_(0, 255), **kwargs)
