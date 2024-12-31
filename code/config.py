import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/CA', help='pesudo_root_path')


# parser.add_argument('--valid_path', type=str,
#                     default='../data/CA', help='valid_path (pesudo)')


parser.add_argument('--exp', type=str,
                    default='Mean_Teacher', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='model', help='model_name')
parser.add_argument('--details', type=str,
                    default='', help='details')
parser.add_argument('--is_breast', type=str, default=True)
# parser.add_argument('--max_iterations', type=int,
#                     default=30000, help='maximum epoch number to train')
parser.add_argument('--max_iterations', type=int,
                    default=250000, help='maximum epoch number to train')
parser.add_argument('--cur_fold', type=int,
                    default=1, help='current fold of k fold validation 1~k')
parser.add_argument('--Kfold', type=int,
                    default=5, help='k of k fold validation ')

parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=2,
                    help='labeled_batch_size per gpu')


parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')

# parser.add_argument('--patch_size', type=list,  default=[96, 224, 224],
#                     help='patch size of network input')
parser.add_argument('--patch_size', type=list,  default=[96, 160, 160],#[80,160,192],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=42, help='random seed')


# costs
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
# parser.add_argument('--consistency_rampup', type=float,
#                     default=200.0, help='consistency_rampup')
parser.add_argument('--consistency_rampup', type=float,
                    default=1666, help='consistency_rampup')
parser.add_argument('--pred_out_dir', type=str,
                    default="pred", help='predict output directoty')
parser.add_argument('--img_suffix', type=str,
                    default="_0000.nii.gz", help='suffix of image')
parser.add_argument('--mask_suffix', type=str,
                    default=".nii.gz", help='suffix of image')

parser.add_argument('--num_classes', type=int,
                    default=6, help='class number of label')

parser.add_argument('--show_image_per_iterations', type=int,
                    default=20, help='default: show image per 20 iterations')
parser.add_argument('--save_model_per_iterations', type=int,
                    default=10000, help='default: save model per 10000 iterations')

parser.add_argument('--breast_line_class', type=int,
                    default=4, help='tumor class number of image')
parser.add_argument('--tumor_class', type=int,
                    default=3, help='tumor class number of image')
parser.add_argument('--gland_class', type=int,
                    default=2, help='gland class number of image')
parser.add_argument('--adipose_class', type=int,
                    default=1, help='gland class number of image')


parser.add_argument('--model_path', type=str,
                    default="", help='model_path')
parser.add_argument('--test_image_folder', type=str,
                    default="", help='test_image_folder')
parser.add_argument('--predict_out_folder', type=str,
                    default="", help='predict_out_folder')

args = parser.parse_args()

