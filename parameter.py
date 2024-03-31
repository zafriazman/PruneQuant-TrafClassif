import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='pruning and quantization search script')

    # datasets and model
    parser.add_argument('--model', default='CNN1D_TrafficClassification', type=str, help='model name to prune')
    parser.add_argument('--dataset', default='iscx2016vpn', type=str, help='dataset to use (iscx2016vpn/etc)')

    # general settings
    parser.add_argument('--seed', default=2024, type=int, help='random seed to set')
    parser.add_argument('--n_worker', default=0, type=int, help='number of data loader worker')
    parser.add_argument('--data_bsize', default=128, type=int, help='number of data batch size')

    # pruning and quantization
    parser.add_argument('--prune_ratio', default=0.5, type=float, help='0.7 means prune 70 percent of filters')
    parser.add_argument('--qat_lr', default=0.01, type=float, help='learning rate for quantize aware training')
    parser.add_argument('--qat_momentum', default=0.9, type=float, help='momentum for quantize aware training')
    parser.add_argument('--qat_wd', default=4e-5, type=float, help='weight decay for quantize aware training')
    parser.add_argument('--qat_epochs', default=1, type=int, help='training epoch during calibration for quantize aware training')
    parser.add_argument('--min_bitwidth', default=2, type=float, help='minimum bitwidth per layer')
    parser.add_argument('--max_bitwidth', default=8, type=float, help='maximum bitwidth per layer')

    # rl agent
    parser.add_argument('--state_dim', default=11, type=int, help='state dimension (Following AMC paper is 11)')
    parser.add_argument('--solved_reward', default=0, type=int, help='stop training if avg_reward > solved_reward')
    parser.add_argument('--log_interval', default=1, type=int, help='print avg reward in the interval')
    parser.add_argument('--max_episodes', default=500, type=int, help='max training episodes')
    parser.add_argument('--max_timesteps', default=5, type=int, help='max timesteps in one episode')
    # to exploit more, reduce action_std (variance)
    parser.add_argument('--action_std', default=0.5, type=float, help='constant std for action distribution (Multivariate Normal)')
    parser.add_argument('--K_epochs', default=10, type=int, help='update policy for K epochs')
    parser.add_argument('--eps_clip', default=0.2, type=float, help='clip parameter for RL')
    parser.add_argument('--gamma', default=0.99, type=float, help='discount factor')
    parser.add_argument('--lr', default=0.0003, type=float, help='learning rate for optimizer')
    parser.add_argument('--update_timestep', default=50, type=int, help='update policy every n timesteps')

    return parser.parse_args()