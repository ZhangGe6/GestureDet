import argparse
import random
import time
import warnings
import copy

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from core import train, validate
from pose_utils.common import delete_prevoius_checkpoint, save_checkpoint
from model import UltraLightSimplePoseNet
from dataset import HandPoseDataset
from quantiation.utils import quantize_model, freeze_weight_except_bias

# see https://github.com/zhaoweicai/EdMIPS for full and more powerful parser
parser = argparse.ArgumentParser(description='PyTorch demo')
# dataset config
parser.add_argument('--data_root', default='/home/zg/wdir/zg/moyu/GestureDet/Datasets/train_val_jsons', 
                    type=str, help='dataset dir')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--num_workers', default=4, type=int, 
                    help='dataloader num workers')
# model config
parser.add_argument('--arch', metavar='ARCH', default='mobilenetv2',
                    help='model architecture')
# train config
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.00001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('-df', '--disp-freq', default=10, type=int,
                    metavar='N', help='display frequency (default: 10)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--device', default='cuda:0', type=str,
                    help='cpu or gpu')
parser.add_argument('--ckpt_dir', default='./QAF_checkpoints', type=str, 
                    help='checpoints save dir')
parser.add_argument('-s', '--save_ckpt', action='store_true',
                    help='save checkpoint or not')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--resume_path', default=None, type=str, 
                    help='path to resume checkpoint')

def main():
    args = parser.parse_args()
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
                      
    tb_logger = SummaryWriter()
    main_worker(args, tb_logger)

def main_worker(args, tb_logger):
    # set cuda
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.cuda.set_device(args.device)

    # create dataloader
    train_dataset = HandPoseDataset(data_root=args.data_root, split='train')
    test_dataset = HandPoseDataset(data_root=args.data_root, split='test')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
    
    # create model
    model = UltraLightSimplePoseNet().to(args.device)

    # define loss function (criterion) and optimizer
    # criterion = nn.MSELoss().to(args.device)
    criterion = nn.MSELoss().to(args.device)

    if args.resume_path is not None:
        print("resume checkpoint from {}".format(args.resume_path))
        checkpoint = torch.load(args.resume_path)
        # for key in checkpoint.keys():
        #     print(key)
        model.load_state_dict(checkpoint)

    model = quantize_model(model, args)
    model = model.to(args.device)
    print(model)

    # freeze_weight_except_bias(model, verbose=True)

    if args.evaluate:
        # validate(val_loader, model, criterion, args)
        validate(val_loader, model, criterion, 0, args, tb_logger)
        return

    # optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=5e-4) 
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50, 70], gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    best_acc1 = 0
    for epoch in range(args.epochs):
        # train for one epoch
        loss, acc = train(train_loader, model, criterion, optimizer, epoch, args, tb_logger)

        # evaluate on validation set
        print("validating for epoch {} ...".format(epoch))
        eval_acc = validate(val_loader, model, criterion, epoch, args, tb_logger)

        # viz.accuracy_curve(train_acc=train_acc1.cpu(), val_acc=eval_acc1.cpu())
        # remember best acc@1 and save checkpoint
        is_best = eval_acc > best_acc1
        if is_best:
            best_acc1 = eval_acc
            # delete_prevoius_checkpoint(args.arch, args.ckpt_dir)
            save_checkpoint(args.arch, model.state_dict(), epoch, eval_acc, args.ckpt_dir)
            print("==> checkpoint at epoch {} saved".format(epoch))

        scheduler.step()
    


if __name__ == '__main__':
    main()