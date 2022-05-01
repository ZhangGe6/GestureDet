import time
import torch
import torch.optim
from tqdm import tqdm
import numpy as np
import os
import cv2
from pose_utils.metrics import DataLogger, calc_accuracy
from pose_utils.vis import draw_result_from_heatmap

def train(train_loader, model, criterion, optimizer, epoch, args, tb_logger):
    loss_logger = DataLogger()
    acc_logger = DataLogger()
    # switch to train mode
    model.train()
    feat_stride = 4

    pbar = tqdm(total=len(train_loader))
    pbar.set_description('Epoch {}'.format(epoch))
    for i, (imgs, target_masks, target_weights, joints, img_paths) in enumerate(train_loader):
        imgs = imgs.to(args.device)
        target_masks = target_masks.to(args.device)
        target_weights = target_weights.to(args.device)
        # print(target_weights)
        # print(target_masks.shape)
        # print(target_masks[0][0][7])
        # print(target_masks.shape)
        # for j in range(target_masks.shape[2]):
        #     print(target_masks[0][0][j])
        # print(torch.max(target_masks))

        output = model(imgs)
        # print(output.shape)
        # print(target_weights.shape)
        # print(imgs.dtype, target_masks.dtype, target_weights.dtype)
        # print(target_weights)
        loss = 0.5 * criterion(output.mul(target_weights), target_masks.mul(target_weights))
        # loss += torch.norm(output)
        # loss = dice_loss(target_masks.mul(target_weights), output.mul(target_weights))
        acc = calc_accuracy(output.mul(target_weights), target_masks.mul(target_weights))
        # print(loss, acc)

        # measure accuracy and record loss
        batch_size = imgs.size(0)
        loss_logger.update(loss.item(), batch_size)
        acc_logger.update(acc.item(), batch_size)

        # compute gradient and backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.disp_freq == 0:
            # viz.loss_curve(losses.avg)
            step = i + epoch * len(train_loader)
            tb_logger.add_scalar('train/loss', loss_logger.avg, step)
            tb_logger.add_scalar('train/acc', acc_logger.avg, step)

            pbar.set_postfix(
                loss='{:8f}'.format(loss_logger.avg), 
                heatmap_acc='{:8f}'.format(acc_logger.avg)
            )

        if i % (5 * args.disp_freq) == 0:
            step = i + epoch * len(train_loader)
            gt_joints_img = draw_result_from_heatmap(imgs[0], target_masks[0], feat_stride=feat_stride)
            pred_joints_img = draw_result_from_heatmap(imgs[0], output[0], feat_stride=feat_stride)

            tb_logger.add_image('joints_on_img/affined_with_gt_joints', gt_joints_img, step, dataformats='HWC')
            tb_logger.add_image('joints_on_img/affined_with_pred_joints', pred_joints_img, step, dataformats='HWC')
            
        pbar.update()
    pbar.close()

    return loss_logger.avg, acc_logger.avg


def validate(val_loader, model, criterion, epoch, args, tb_logger):
    loss_logger = DataLogger()
    acc_logger = DataLogger()

    model.eval()

    for i, (imgs, target_masks, target_weights, joints, img_paths) in tqdm(enumerate(val_loader)):
        imgs = imgs.to(args.device)
        target_masks = target_masks.to(args.device)
        target_weights = target_weights.to(args.device)
        # joints = joints.to(args.device)
        # print(imgs.shape)
        output = model(imgs)

        loss = 0.5 * criterion(output.mul(target_weights), target_masks.mul(target_weights))
        acc = calc_accuracy(output.mul(target_weights), target_masks.mul(target_weights))

        batch_size = imgs.size(0)
        loss_logger.update(loss.item(), batch_size)
        acc_logger.update(acc.item(), batch_size)
    
        if i % args.disp_freq == 0:
            # viz.loss_curve(losses.avg)
            step = i + epoch * len(val_loader)
            tb_logger.add_scalar('eval/loss', loss_logger.avg, step)
            tb_logger.add_scalar('eval/acc', acc_logger.avg, step)

    return acc_logger.avg


