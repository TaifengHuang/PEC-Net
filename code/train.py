import argparse
import logging
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss, BCELoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from glob import glob

from dataloaders.dataset import (BaseDataSets, RandomGenerator, Flip_Color_Augment, WeakStrongAugment,WeakStrongAugmentMore,
                                 TwoStreamBatchSampler)
from networks.net_factory import net_factory
# from networks.discriminator import Discriminator, PixelDiscriminator
from utils import losses, metrics, ramps
from val_2D import eval_modal_decoder_difference


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def load_net(model, path):
    state = torch.load(str(path))
    model.load_state_dict(state)


def sharpening(P):
    T = 1 / args.temperature
    P_sharpen = P ** T / (P ** T + (1 - P) ** T)
    return P_sharpen


def dice1_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


def get_XOR_region(s1, s2):
    # l1 = torch.argmax(s1, dim=1)
    # l2 = torch.argmax(s2, dim=1)
    threshold = 0.5
    l1 = (s1 > threshold)
    l2 = (s2 > threshold)

    diff_mask = (l1 != l2).float()
    diff_mask = diff_mask.unsqueeze(1)

    return diff_mask


# 数据集：dataset37/brainMRI/COVID-19/Kvasir/luna16/cell/skin
imgs_dir = glob('/root/pycharm_project/datasets/COVID-19/train/imgs/*')  # autodl-tmp/pycharm_project
masks_dir = glob('/root/pycharm_project/datasets/COVID-19/train/masks/*')
val_imgs_dir = glob('/root/pycharm_project/datasets/COVID-19/test/imgs/*')
val_masks_dir = glob('/root/pycharm_project/datasets/COVID-19/test/masks/*')

parser = argparse.ArgumentParser()
# parser.add_argument('--root_path', type=str, default='./data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='COVID-19/Test_+_and_cat_xiaorong_nodecoupling', help='experiment_name')
parser.add_argument('--model', type=str, default='pecnet', help='model_name')
parser.add_argument('--pre_max_iteration', type=int, default=2000, help='maximum pre-train iteration to train')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--critic_lr', type=float, default=0.0001, help='DAN learning rate')
parser.add_argument('--patch_size', type=list, default=[256, 256], help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--num_classes', type=int, default=1, help='output channel of network')
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=8, help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=104, help='labeled data')
# costs
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--temperature', type=float, default=0.1, help='temperature of sharpening')
parser.add_argument('--lamda', type=float, default=0.25, help='weight to balance all losses')

args = parser.parse_args()


def train(args, snapshot_path):
    base_lr = args.base_lr
    labeled_bs = args.labeled_bs
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    labeled_num = args.labeled_num

    # model = net_factory(net_type=args.model, in_chns=3, class_num=num_classes)

    def create_model(ema=False):
        model = net_factory(net_type=args.model, in_chns=3, class_num=num_classes)
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        model = model.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(imgs_dir=imgs_dir, masks_dir=masks_dir, split="train", transform=transforms.Compose([
        # RandomGenerator(args.patch_size)
        WeakStrongAugment(args.patch_size)
    ]))
    db_val = BaseDataSets(imgs_dir=val_imgs_dir, masks_dir=val_masks_dir, split="val")

    n_train = len(db_train)  # 566
    n_val = len(db_val)  # 242
    print("Total train num is: {}, labeled num is: {}".format(
        n_train, labeled_num))

    labeled_idxs = list(range(0, labeled_num))
    unlabeled_idxs = list(range(labeled_num, n_train))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    model.train()

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    ce_loss = BCELoss()

    mse_criterion = losses.mse_loss
    criterion_att = losses.Attention()
    voxel_kl_loss = nn.KLDivLoss(reduction="none")

    # dice_loss = losses.DiceLoss(n_classes=num_classes)
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    lr_ = base_lr
    iterator = tqdm(range(max_epoch), ncols=70)
    cur_threshold = 1 / 2

    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):

            # volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            # volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            weak_batch, strong_batch, label_batch = (
                sampled_batch["image_weak"],
                sampled_batch["image_strong"],
                sampled_batch["label_aug"],
            )
            weak_batch, strong_batch, label_batch = (
                weak_batch.cuda(),
                strong_batch.cuda(),
                label_batch.cuda(),
            )

            output2, output3 = model(weak_batch, 2)
            # output1 = model(strong_batch[labeled_bs:], 1)
            output1, output4 = model(strong_batch[labeled_bs:], 1)
            output5, output6 = model(weak_batch[:labeled_bs], 1)
            # output4 = model(strong2_batch[labeled_bs:], 4)

            output1_soft = torch.sigmoid(output1)
            output2_soft = torch.sigmoid(output2)
            output3_soft = torch.sigmoid(output3)
            output4_soft = torch.sigmoid(output4)
            output5_soft = torch.sigmoid(output5)
            output6_soft = torch.sigmoid(output6)

            loss_seg_dice = 0
            loss_seg_dice += (dice1_loss(output2_soft[:labeled_bs], label_batch[:labeled_bs]) + ce_loss(output2_soft[:labeled_bs],label_batch[:labeled_bs].float())) * 0.5
            loss_seg_dice += (dice1_loss(output3_soft[:labeled_bs], label_batch[:labeled_bs]) + ce_loss(output3_soft[:labeled_bs],label_batch[:labeled_bs].float())) * 0.5
            loss_seg_dice += (dice1_loss(output5_soft[:labeled_bs], label_batch[:labeled_bs]) + ce_loss(output5_soft[:labeled_bs],label_batch[:labeled_bs].float())) * 0.5
            loss_seg_dice += (dice1_loss(output6_soft[:labeled_bs], label_batch[:labeled_bs]) + ce_loss(output6_soft[:labeled_bs],label_batch[:labeled_bs].float())) * 0.5

            # entropy_pre_2 = -torch.sum(output2_soft[labeled_bs:] * torch.log(output2_soft[labeled_bs:] + 1e-16), dim=1)
            # entropy_pre_3 = -torch.sum(output3_soft[labeled_bs:] * torch.log(output3_soft[labeled_bs:] + 1e-16), dim=1)
            entropy_pre_2 = - (output2_soft[labeled_bs:] * torch.log(output2_soft[labeled_bs:] + 1e-16) +(1 - output2_soft[labeled_bs:]) * torch.log(1 - output2_soft[labeled_bs:] + 1e-16))
            entropy_pre_3 = - (output3_soft[labeled_bs:] * torch.log(output3_soft[labeled_bs:] + 1e-16) +(1 - output3_soft[labeled_bs:]) * torch.log(1 - output3_soft[labeled_bs:] + 1e-16))

            mask_2 = (entropy_pre_3 > entropy_pre_2).float()
            mask_3 = 1 - mask_2

            # mask_4 = torch.abs(output2_soft[labeled_bs:] - output3_soft[labeled_bs:]) > 0.5
            # mask_4 = mask_4.float()
            # epsilon = 1e-6
            # weighted_output2 = (output2_soft[labeled_bs:] * mask_4 + epsilon) / 2
            # weighted_output3 = (output3_soft[labeled_bs:] * mask_4 + epsilon) / 2
            # loss_kd = dice1_loss(weighted_output2, weighted_output3)

            # output23_soft = (output2_soft[labeled_bs:] * mask_2.unsqueeze(1) + output3_soft[labeled_bs:] * mask_3.unsqueeze(1)) #/ (mask_2 + mask_3 + 1e-16)
            output23_soft = (output2_soft[labeled_bs:] * mask_2 + output3_soft[labeled_bs:] * mask_3) #/ (mask_2 + mask_3 + 1e-16)


            threshold = 0.5
            pseudo_output23 = (output23_soft > threshold).float()

            pseudo_supervision = 0
            pseudo_supervision += ce_loss(output1_soft, pseudo_output23.detach())
            pseudo_supervision += ce_loss(output4_soft, pseudo_output23.detach())


            mse = mse_criterion(output2_soft[labeled_bs:], output3_soft[labeled_bs:])


            consistency_weight = get_current_consistency_weight(iter_num // 150)
            supervised_loss = loss_seg_dice

            loss = supervised_loss + pseudo_supervision + consistency_weight * mse


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_num = iter_num + 1
            # logging.info('iteration %d :loss : %03f,  loss_seg_dice: %03f, pseudo_supervision: %03f'
            #              % (iter_num, loss, loss_seg_dice, pseudo_supervision))
            logging.info('iteration %d :loss : %03f,  loss_seg_dice: %03f,  pseudo_supervision: %03f,  mse: %03f'
                         % (iter_num, loss, loss_seg_dice, pseudo_supervision, mse))

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                dice_sum, hs_sum = eval_modal_decoder_difference(model, valloader, device, n_val, classes=num_classes)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/model1_val_{}_dice'.format(class_i + 1),
                                      dice_sum, iter_num)
                    writer.add_scalar('info/model1_val_{}_hd95'.format(class_i + 1),
                                      hs_sum, iter_num)

                writer.add_scalar('info/model1_val_dice', dice_sum, iter_num)
                writer.add_scalar('info/model1_val_hd', hs_sum, iter_num)

                if dice_sum > best_performance:
                    best_performance = dice_sum
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num,
                                                                                              best_performance))  # round(best_performance, 4)
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)

                logging.info('iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, dice_sum, hs_sum))
                model.train()

            if iter_num % 15000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break

    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pre_snapshot_path = "../model/{}_{}_labeled/{}/pre_train".format(
        args.exp, args.labeled_num, args.model)
    snapshot_path = "../model/{}_{}_labeled/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    # 保存代码
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    # pre_train(args, pre_snapshot_path)
    # train(args, pre_snapshot_path, snapshot_path)
    train(args, snapshot_path)
