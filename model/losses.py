import os
import sys

from model.discriminators import SNTemporalPatchGANDiscriminator
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # NOQA

import torch
import torch.nn as nn

# Based on https://github.com/knazeri/edge-connect/blob/master/src/loss.py
class GANLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, device, gan_type='lsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge | l1
        """
        super(GANLoss, self).__init__()

        self.gan_type = gan_type
        self.register_buffer('real_label', torch.tensor(target_real_label).to(device))
        self.register_buffer('fake_label', torch.tensor(target_fake_label).to(device))

        if gan_type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif gan_type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif gan_type == 'hinge':
            self.criterion = nn.ReLU()

        elif gan_type == 'l1':
            self.criterion = nn.L1Loss()

        elif self.gan_type == 'vanilla':
            self.criterion = nn.BCEWithLogitsLoss()

        else:
            raise NotImplementedError(
                f'GAN type {self.gan_type} is not implemented.')

    def __call__(self, outputs, is_real, is_disc=None):
        """
        Args:
            outputs (Tensor): The input for the loss module, i.e., the network
                prediction.
            is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.
        Returns:
            Tensor: GAN loss value.
        """
        if self.gan_type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss

class AdversarialLoss(nn.Module):
    def __init__(self, gpu_id, image_channels = 1, gan_type='RGAN', gan_k=2, lr_dis=1e-4):
        print('including adv loss')
        super(AdversarialLoss, self).__init__()
        self.gan_type = gan_type
        self.gan_k = gan_k

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        self.device = torch.device('cuda:{}'.format(0))

        self.discriminator = SNTemporalPatchGANDiscriminator(image_channels * 2).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            betas=(0, 0.9), eps=1e-8, lr=lr_dis
        )

        self.criterion_adv = GANLoss(device=self.device,gan_type='vanilla').to(self.device)

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
        d_fake_detach = fake.detach()
        for _ in range(self.gan_k):
            self.set_requires_grad(self.discriminator, True)
            self.optimizer.zero_grad()
            # real
            d_fake = self.discriminator(d_fake_detach)
            d_real = self.discriminator(real)
            d_real_loss = self.criterion_adv(d_real, True, is_disc=True)
            # fake
            d_fake = self.discriminator(d_fake_detach)
            d_fake_loss = self.criterion_adv(d_fake, False, is_disc=True)
            loss_d = (d_real_loss + d_fake_loss) * 0.5
            
            loss_d.backward()
            self.optimizer.step()

        # G Loss
        self.set_requires_grad(self.discriminator, False)
        d_fake = self.discriminator(fake)
        loss_g = self.criterion_adv(d_fake, True, is_disc=False)

        # Generator loss
        return loss_g, loss_d

    def state_dict(self):
        D_state_dict = self.discriminator.state_dict()
        D_optim_state_dict = self.optimizer.state_dict()
        return D_state_dict, D_optim_state_dict