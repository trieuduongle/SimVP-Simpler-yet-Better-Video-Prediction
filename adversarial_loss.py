from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import os

img_shape = (10, 1, 64,64)

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
        if self.gan_type == 'normalBCE':
            self.loss = nn.BCELoss()
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
    def __init__(self, device, d=64):
        super(Discriminator, self).__init__()
        # First param if the number of channel input+target with batch_size included
        self.conv1 = nn.Conv2d(32, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 1, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, 1, 4, 1, 1)

        self.device = device

        self.criterion_adv = GANLoss(gan_type='vanilla').to(self.device)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input, target):
        # Concatenate on channel dim
        # (batch_size, channel, width, height)
        x = torch.cat([input, target], 1)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.sigmoid(self.conv5(x))

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

        self.criterion_adv = GANLoss(gan_type='normalBCE').to(self.device)

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

    def forward(self, inputs, fake, real):
        inputs = inputs.reshape(10, -1, 64, 64)
        fake=  fake.reshape(10, -1, 64, 64)
        real = real.reshape(10, -1, 64, 64)

        # D Loss
        for _ in range(self.gan_k):
            self.set_requires_grad(self.discriminator, True)
            self.optimizer.zero_grad()
            # check detach
            # real
            d_real = self.discriminator(inputs, real).squeeze()
            d_real_loss = self.criterion_adv(d_real, Variable(torch.ones(d_real.size())), is_disc=True)
            # d_real_loss.backward()
            # fake
            d_fake = self.discriminator(inputs, fake.detach()).squeeze()
            d_fake_loss = self.criterion_adv(d_fake, Variable(torch.zeros(d_fake.size())), is_disc=True)
            # d_fake_loss.backward()
            loss_d = (d_real_loss + d_fake_loss) * 0.5
            loss_d.backward()
            self.optimizer.step()

        # G Loss
        self.set_requires_grad(self.discriminator, False)
        d_fake = self.discriminator(inputs, fake)
        loss_g = self.criterion_adv(d_fake, Variable(torch.ones(d_fake.size())), is_disc=True)

        # Generator loss
        return loss_g, loss_d

    def state_dict(self):
        D_state_dict = self.discriminator.state_dict()
        D_optim_state_dict = self.optimizer.state_dict()
        return D_state_dict, D_optim_state_dict

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()