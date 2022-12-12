import numpy as np
import torch
import os

class Recorder:
    def __init__(self, verbose=False, delta=0):
        self.verbose = verbose
        self.best_score = None
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, optimizer, discriminator, discriminator_optimizer, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, discriminator, discriminator_optimizer, path)
        elif score >= self.best_score + self.delta:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, discriminator, discriminator_optimizer, path)

    def save_checkpoint(self, val_loss, model, optimizer, discriminator, discriminator_optimizer, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'checkpoint.pth')

        torch.save({
            'GENERATOR_STATE_DICT': model.state_dict(),
            'GENERATOR_OPTIMIZER_STATE_DICT': optimizer.state_dict(),
            'DISCRIMINATOR_STATE_DICT': discriminator.state_dict(),
            'DISCRIMINATOR_OPTIMIZER_STATE_DICT': discriminator_optimizer.state_dict()
        }, os.path.join(
            path, 'checkpoint.pth'))
        self.val_loss_min = val_loss