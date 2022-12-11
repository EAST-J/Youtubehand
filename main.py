from run import Runner
import os.path as osp
import os
from options import BaseOptions
from termcolor import cprint
from tensorboardX import SummaryWriter
from utils.logger import setup_logger
from utils.miscellaneous import mkdir, set_seed
from utils.comm import get_rank
from utils.read import spiral_tramsform
from utils.mano import MANO
from datasets.build import make_hand_data_loader
from model.network import Network
from run import Runner
import torch

def main():
    global logger
    args = BaseOptions().parse()
    args.distributed = False # Not set for multi-GPU training
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = torch.device(args.device)
    torch.cuda.set_device(args.device_idx)
    mkdir(args.output_dir)
    logger = setup_logger("LOGINFO", args.output_dir, get_rank())
    set_seed(args.seed, get_rank())

    args.work_dir = osp.dirname(osp.realpath(__file__))
    # get GCN settings
    template_fp = osp.join(args.work_dir, 'template', 'template.ply')
    transform_fp = osp.join(args.work_dir, 'template', 'transform.pkl')
    spiral_indices_list, down_transform_list, up_transform_list, pkl_file = spiral_tramsform(transform_fp, template_fp, args.ds_factors, args.seq_length, args.dilation)
    mano_model = MANO(args.mano_dir).to(args.device)

    model = Network(args.in_channels, args.out_channels, spiral_indices_list, up_transform_list, down_transform_list, args.backbone)
    # resume from checkpoint
    if args.resume:
        model_path = args.resume
        checkpoint = torch.load(model_path, map_location='cpu')
        if checkpoint.get('model_state_dict', None) is not None:
            checkpoint = checkpoint['model_state_dict']
        model.load_state_dict(checkpoint)
        epoch = checkpoint.get('epoch', -1) + 1
        cprint('Load checkpoint {}'.format(model_path), 'yellow')
    total_params = sum(p.numel() for p in model.parameters())
    cprint('Model total parameter {}'.format(total_params), 'yellow')
    model = model.to(args.device)
    runner = Runner(args, model, mano_model)
    #####
    if args.split == 'train':
        dataloader = make_hand_data_loader(args, args.train_yaml, args.distributed, is_train=True, scale_factor=args.img_scale_factor)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.decay_step, gamma=args.lr_decay)
        # tensorboard
        board_dir = osp.join(args.output_dir, 'board')
        mkdir(board_dir)
        board = SummaryWriter(board_dir)
        runner.train(dataloader, optimizer, scheduler, board, logger)
    elif args.split == 'eval':
        dataloader = make_hand_data_loader(args, args.test_yaml, args.distributed, is_train=False, scale_factor=args.img_scale_factor)
        runner.evaluation(dataloader)
    else:
        img_file_list = os.listdir(osp.join(args.work_dir, 'images'))
        runner.demo(img_file_list)

if __name__ == '__main__':
    main()