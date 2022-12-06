import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import os

img_shape = (10, 1, 64,64)


class DiscriminatorBlock(nn.Module):
    def __init__(self, filters,strides):
        super(DiscriminatorBlock, self).__init__()
        self.conv0 = nn.Conv2D(filters,kernel_size=4, strides=strides,padding='same')
        self.norm1d = nn.InstanceNorm1d()
        self.leakyReLU = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.conv0(x)
        x = self.norm1d(x)
        x = self.leakyReLU(x)
        return x

class GANLoss(nn.Module):
    """Define GAN loss.
    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self,
                 gan_type,
                 real_label_val=1.0,
                 fake_label_val=0.0,
                 loss_weight=1.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'wgan_softplus':
            self.loss = self._wgan_softplus_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(
                f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input, target):
        """wgan loss.
        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.
        Returns:
            Tensor: wgan loss.
        """
        return -input.mean() if target else input.mean()

    def _wgan_softplus_loss(self, input, target):
        """wgan loss with soft plus. softplus is a smooth approximation to the
        ReLU function.
        In StyleGAN2, it is called:
            Logistic loss for discriminator;
            Non-saturating loss for generator.
        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.
        Returns:
            Tensor: wgan loss.
        """
        return F.softplus(-input).mean() if target else F.softplus(
            input).mean()

    def get_target_label(self, input, target_is_real):
        """Get target label.
        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.
        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """

        if self.gan_type in ['wgan', 'wgan_softplus']:
            return target_is_real
        target_val = (
            self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.
        Returns:
            Tensor: GAN loss value.
        """
        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == 'hinge':
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:  # other gan types
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight

class Discriminator(nn.Module):
    def __init__(self, device):
        super(Discriminator, self).__init__()

        self.conv0 =  nn.Conv2D(64, kernel_size=4, strides=2, padding='same')
        self.leakyReLU = nn.LeakyReLU(negative_slope=0.2)
        self.block0 = DiscriminatorBlock(128, 2)
        self.block1 = DiscriminatorBlock(256, 2)
        self.block2 = DiscriminatorBlock(512, 1)
        self.conv1 = nn.Conv2D(1, kernel_size=4, strides=1, padding='same')

        self.device = device

        self.criterion_adv = GANLoss(gan_type='vanilla').to(self.device)

    def forward(self, x):
        x = self.conv0(x)
        x = self.leakyReLU(x)
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.conv1(x)
        return x

class AdversarialLoss(nn.Module):
    def __init__(self, gpu_id, gan_type='RGAN', gan_k=2,
                 lr_dis=1e-4):
        print('including adv loss')
        super(AdversarialLoss, self).__init__()
        self.gan_type = gan_type
        self.gan_k = gan_k

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        self.device = torch.device('cuda:{}'.format(0))

        self.discriminator = Discriminator(self.device).to(self.device)

        self.optimizer = torch.optim.Adam(
                self.discriminator.parameters(),
                betas=(0, 0.9), eps=1e-8, lr=lr_dis
            )

        self.criterion_adv = GANLoss(gan_type='vanilla').to(self.device)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def forward(self, fake, real):
        # D Loss
        for _ in range(self.gan_k):
            self.set_requires_grad(self.discriminator, True)
            self.optimizer.zero_grad()
            # real
            d_fake = self.discriminator(fake).detach()
            d_real = self.discriminator(real)

            print(d_fake.size())
            d_real_loss = self.criterion_adv(d_real - torch.mean(d_fake), True,
                                               is_disc=True) * 0.5
            d_real_loss.backward()
            # fake
            d_fake = self.discriminator(fake.detach())
            d_fake_loss = self.criterion_adv(d_fake - torch.mean(d_real.detach()), False,
                                                is_disc=True) * 0.5
            d_fake_loss.backward()
            loss_d = d_real_loss + d_fake_loss
            
            self.optimizer.step()

        # G Loss
        self.set_requires_grad(self.discriminator, False)
        d_real = self.discriminator(real).detach()
        d_fake = self.discriminator(fake)
        g_real_loss = self.criterion_adv(d_real - torch.mean(d_fake), False, is_disc=False) * 0.5
        g_fake_loss = self.criterion_adv(d_fake - torch.mean(d_real), True, is_disc=False) * 0.5
        loss_g = g_real_loss + g_fake_loss

        # Generator loss
        return loss_g, loss_d

    def state_dict(self):
        D_state_dict = self.discriminator.state_dict()
        D_optim_state_dict = self.optimizer.state_dict()
        return D_state_dict, D_optim_state_dict
