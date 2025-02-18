import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
parser.add_argument('--testsize', type=int, default=256, help='testing dataset size')

parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--num_thread', type=int, default=1)
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
parser.add_argument('--load_mit', type=str, default='', help='train from checkpoints')
#============================================================================
# dataset
parser.add_argument('--original_image_root', type=str, default=r'.../', help='the training original images root')
parser.add_argument('--target_image_root', type=str, default=r'.../', help='the training target images root')
parser.add_argument('--gt_root', type=str, default=r'.../', help='the training gt images root')

parser.add_argument('--test_original_image_root', type=str, default=r'.../', help='the test original images root')
parser.add_argument('--test_target_image_root', type=str, default=r'.../', help='the test target images root')
parser.add_argument('--test_gt_root', type=str, default=r'.../', help='the test gt images root')

parser.add_argument('--val_original_image_root', type=str, default=r'.../', help='the val original images root')
parser.add_argument('--val_target_image_root', type=str, default=r'.../', help='the val target images root')
parser.add_argument('--val_gt_root', type=str, default=r'.../', help='the val gt images root')

#============================================================================

parser.add_argument('--save_path', type=str, default=r'.../', help='the path to save models and logs')
opt = parser.parse_args()