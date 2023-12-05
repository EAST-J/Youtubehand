import argparse

class BaseOptions():
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--exp_name', type=str, default='hand-recon')
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--device_idx', type=int, nargs='+', default=0)
        # dataset hyperparameters
        parser.add_argument('--train_yaml', type=str, default='freihand/train.yaml')
        parser.add_argument('--test_yaml', type=str, default='freihand/test.yaml')
        parser.add_argument("--img_scale_factor", default=1, type=int, help="adjust image resolution.")  
        # network hyperparameters
        parser.add_argument('--out_channels', nargs='+', default=[64, 128, 256, 512], type=int)
        parser.add_argument('--ds_factors', nargs='+', default=[2, 2, 2, 2], type=float)
        parser.add_argument('--in_channels', type=int, default=3)
        parser.add_argument('--seq_length', type=int, default=[27, 27, 27, 27], nargs='+')
        parser.add_argument('--dilation', type=int, default=[1, 1, 1, 1], nargs='+')
        parser.add_argument('--backbone', type=str, default='ResNet18')

        # optimizer hyperparmeters
        parser.add_argument('--optimizer', type=str, default='Adam')
        parser.add_argument('--lr_scheduled', type=str, default='MultiStep')
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--lr_decay', type=float, default=0.1)
        parser.add_argument('--decay_step', type=int, nargs='+', default=[30, ])
        parser.add_argument('--weight_decay', type=float, default=0)

        # training hyperparameters
        parser.add_argument('--split', type=str, default='train')
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--epochs', type=int, default=38)
        parser.add_argument('--resume', default=False, action='store_true')
        parser.add_argument('--checkpoint_path', type=str, default='')
        parser.add_argument('--joint_2d_loss_weight', type=float, default=0.1)
        parser.add_argument('--vertices_loss_weight', type=float, default=1.0)
        parser.add_argument('--joint_3d_loss_weight', type=float, default=1.0)
        parser.add_argument('--edge_loss_weight', type=float, default=1.0)
        parser.add_argument('--normal_loss_weight', type=float, default=0.1)
        # others
        parser.add_argument('--seed', type=int, default=88)
        parser.add_argument("--multiscale_inference", default=False, action='store_true',) 
        parser.add_argument('--logging_steps', type=int, default=100, 
                        help="Log every X steps.")
        # dir setting
        parser.add_argument('--output_dir', type=str, default='./out')
        parser.add_argument('--mano_dir', type=str, default='./template')
        parser.add_argument("--data_dir", default='./data', type=str, required=False, help="Directory with all datasets, each in one subfolder")

        self.initialized = True
        return parser

    def str2bool(self, v):
        return v.lower() in ("yes", "true", "t", "1")

    def parse(self):

        parser = argparse.ArgumentParser(description='mesh generator')
        self.initialize(parser)
        args = parser.parse_args()

        return args
