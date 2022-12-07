import argparse

def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--res_dir', default='./results', type=str)
    parser.add_argument('--ex_name', default='Debug', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=1, type=int)

    # dataset parameters
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--val_batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--data_root', default='./data/')
    parser.add_argument('--dataname', default='mmnist', choices=['mmnist', 'taxibj', 'kth'])
    parser.add_argument('--num_workers', default=8, type=int)

    # model parameters
    parser.add_argument('--in_shape', default=[10, 1, 64, 64], type=int,nargs='*') # [10, 1, 64, 64] for mmnist, [4, 2, 32, 32] for taxibj, [10, 1, 128, 128] for kth  
    parser.add_argument('--hid_S', default=64, type=int)
    parser.add_argument('--hid_T', default=256, type=int)
    parser.add_argument('--N_S', default=4, type=int)
    parser.add_argument('--N_T', default=8, type=int)
    parser.add_argument('--groups', default=4, type=int)
    parser.add_argument('--pre_seq_length', default=10, type=int)
    parser.add_argument('--aft_seq_length', default=20, type=int)

    # Training parameters
    parser.add_argument('--epochs', default=51, type=int)
    parser.add_argument('--log_step', default=1, type=int)
    parser.add_argument('--save_epoch_freq', default=10, type=int)
    parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')
    parser.add_argument('--resume_path', default='', type=str)

    parser.add_argument('--lr_D', default=1e-4, type=float)
    parser.add_argument('--gan_type', default='WGAN_GP', type=str)
    parser.add_argument('--lambda_adv', default=5e-3, type=float)

    parser.add_argument('--nframes_pred', type=int, default=6, help='number of video frames to predict in one sample, default=6')
    parser.add_argument('--batch_norm', type=bool, default=False, help='use batch-normalization (not recommended), default=False')
    parser.add_argument('--w_norm', type=bool, default=True, help='use weight scaling, default=True')
    parser.add_argument('--loss', type=str, default='gan', help='which loss functions to use (choices: `gan`, `lsgan` or `wgan_gp`), default=`wgan_gp`')
    parser.add_argument('--d_gdrop', type=bool, default=False, help='use generalized dropout layer (inject multiplicative Gaussian noise) for discriminator when using LSGAN loss, default=False')
    parser.add_argument('--padding', type=str, default='zero', help='which padding to use (choices: `zero`, `replication`), default=`zero`')
    parser.add_argument('--lrelu', type=bool, default=True, help='use leaky relu instead of relu, default=True')
    parser.add_argument('--d_sigmoid', type=bool, default=False, help='use sigmoid at the end of discriminator, default=False')
    parser.add_argument('--nc', type=int, default=3, help='number of input image color channels, default=3')
    parser.add_argument('--d_cond', type=bool, default=True, help='condition discriminator on input frames, default=`True`')
    return parser
