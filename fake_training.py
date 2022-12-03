import os.path as osp
import torch
import numpy as np
from model import SimVP
from tqdm import tqdm
from utils import *
from PIL import Image as im
import torch.utils.data as data
from create_parser import create_parser

class FakeDataset(data.Dataset):
    def __init__(self, n_frames_input=10, n_frames_output=10):
        super(FakeDataset, self).__init__()

        self.dataset = None
        
        self.length = int(1e4) if self.dataset is None else self.dataset.shape[1]

        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = self.n_frames_input + self.n_frames_output
        # For generating data

        self.mean = 0
        self.std = 1

    def __getitem__(self, idx):
        output = np.random.rand(self.n_frames_output, 1, 128, 128)
        input = np.random.rand(self.n_frames_input, 1, 128, 128)
        return input, output

    def __len__(self):
        return self.length


def load_data(batch_size, num_workers, **kwargs):

    train_set = FakeDataset(10,20)

    dataloader_train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    mean, std = 0, 1
    return dataloader_train, mean, std

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

    def _acquire_device(self):
        device = torch.device('cpu')
        return device

    def _preparation(self):
        # seed
        set_seed(self.args.seed)
        # log and checkpoint
        self.path = osp.join(self.args.res_dir, self.args.ex_name)
        
        # prepare data
        self._get_data()
        # build the model
        self._build_model()

    def _build_model(self):
        args = self.args
        self.model = SimVP(tuple(args.in_shape), args.hid_S,
                           args.hid_T, args.N_S, args.N_T).to(self.device)
        if not args.resume_path and os.path.exists(args.resume_path):
            self.model.load_state_dict(torch.load(args.resume_path))

    def _get_data(self):
        config = self.args.__dict__
        self.train_loader, self.data_mean, self.data_std = load_data(**config)

    def _select_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.args.lr, steps_per_epoch=len(self.train_loader), epochs=self.args.epochs)
        return self.optimizer

    def _select_criterion(self):
        self.criterion = torch.nn.MSELoss()

    def train(self, args):
        config = args.__dict__

        for epoch in range(config['epochs']):
            train_loss = []
            self.model.train()
            train_pbar = tqdm(self.train_loader)

            for batch_x, batch_y in train_pbar:
                self.optimizer.zero_grad()
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred_y = self.model(batch_x)

                loss = self.criterion(pred_y, batch_y)
                train_loss.append(loss.item())
                train_pbar.set_description('train loss: {:.4f}'.format(loss.item()))

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            train_loss = np.average(train_loss)

            if epoch % args.log_step == 0:
                print_log("Epoch: {0} | Train Loss: {1:.4f}\n".format(
                    epoch + 1, train_loss))

        return self.model


if __name__ == '__main__':
    print('ok')
    args = create_parser().parse_args()
    config = args.__dict__

    exp = Exp(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>  start <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.train(args)