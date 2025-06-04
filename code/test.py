import argparse
import logging
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from glob import glob
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from scipy.spatial.distance import cdist
import torch.nn as nn
from medpy import metric
# import surface_distance as surfdist

torch.cuda.set_device(0)

from dataloaders.dataset import BaseDataSets
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import directed_hausdorff


# from models.deformation import UNET_defor
# from models.unet import UNet
from networks.pecnet import PEC_Net



logging.getLogger().setLevel(logging.INFO)


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_root', default='../model/generalize/Kvasir-SEG_train_180_labeled/mcnet2d_v1/',  # 需要修改
                        type=str, help="model path")
    parser.add_argument('--model', '-m', default='pecnet_best_model.pth',       # 需要修改
                        metavar='FILE', help="Specify the file in which the model is stored")
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white", default=0.5)
    parser.add_argument('--scale', '-s', type=float,help="Scale factor for the input images",
                        default=0.5)
    parser.add_argument('--num_classes', type=int, default=1,help='output channel of network')  # 需要修改

    return parser.parse_args()


args = get_args()

def numpy_haussdorf(pred: np.ndarray, target: np.ndarray) -> float:#求有向 Hausdorff 距离
    return max(directed_hausdorff(pred, target)[0], directed_hausdorff(target, pred)[0])


def haussdorf(preds, target):
    # print(preds.shape,target.shape)
    # preds, target = torch.as_tensor(preds), torch.as_tensor(target)
    # n_pred = preds.cpu().numpy().squeeze(0)
    # n_target = target.cpu().numpy().squeeze(0)

    res = numpy_haussdorf(preds, target)
    return res


def compute_ASD(seg_mask, gt_mask, spacing):
    """
    计算平均表面距离（ASD）

    Parameters:
        seg_mask (np.array): 分割结果的二进制掩模
        gt_mask (np.array): 真实分割的二进制掩模
        spacing (tuple): 图像的像素间距，例如（0.5, 0.5, 0.5）

    Returns:
        float: 平均表面距离（ASD）
    """
    # 获取边界点
    seg_contour = get_contour_points(seg_mask)
    gt_contour = get_contour_points(gt_mask)

    # 将边界点列表转换为二维数组
    seg_contour = np.array(seg_contour)
    gt_contour = np.array(gt_contour)

    # 计算距离矩阵
    dist_matrix = cdist(seg_contour, gt_contour, 'euclidean') * np.prod(spacing)

    # 计算平均表面距离
    ASD = np.mean(np.min(dist_matrix, axis=1))

    return ASD

def get_contour_points(mask):
    """
    获取二进制掩模的边界点坐标

    Parameters:
        mask (np.array): 二进制掩模

    Returns:
        list: 边界点坐标列表，每个边界点为一个二元元组 (x, y)
    """
    contour_points = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] == 1:
                contour_points.append((i, j))
    return contour_points

def dice_coeff(pred, target):
    pred = torch.from_numpy(pred)
    target = torch.from_numpy(target)
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

def hunxiao(preds, target):
    # print(preds.shape,target.shape) # (256, 256) (256, 256)
    n_pred = preds.ravel()
    n_target = target.astype('int64').ravel()
    # print(n_pred.shape, n_target.shape) # (65536,) (65536,)
    tn, fp, fn, tp = confusion_matrix(n_target, n_pred).ravel()

    # X = confusion_matrix(n_target, n_pred).ravel()  # [64907    76   233   320]

    smooh = 1e-10
    sensitivity = tp / (tp + fn + smooh)
    specificity = tn / (tn + fp + smooh)
    Accuracy = (tp + tn) / (tn + tp + fp + fn + smooh)
    precision = tp / (tp + fp + smooh)
    f1_score = (2 * precision * sensitivity) / (precision + sensitivity + smooh)

    return sensitivity, specificity, Accuracy, precision, f1_score

def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)



def predict_img(net, full_img, device, scale_factor=0.5, out_threshold=0.5):
    net.eval()
    img = BaseDataSets.preprocess(full_img, scale_factor)
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img)

    img = img.unsqueeze(0)#增加一个维度
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)


        if args.num_classes > 1:
            probs = F.softmax(output[0], dim=1)
        else:
            probs = torch.sigmoid(output[0])

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(), #是转换数据格式，把数据转换为tensfroms格式。只有转换为tensfroms格式才能进行后面的处理。
                #transforms.Resize(full_img.size[1]),
                transforms.ToTensor() # 转换为tensor格式，这个格式可以直接输入进神经网络了。
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()
        res = np.int64(full_mask > out_threshold) ##判断是0是1

    return res


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8)) #实现array到image的转换

# basedata_new数据
# brainMRI
# brainMR

if __name__ == "__main__":
    start = time.perf_counter()
    if not os.path.exists('../predict_output/'+ args.model_root.split('/',6)[2] +'_'+ args.model_root.split('/',6)[3]+ '/picture/'):
        os.makedirs('../predict_output/' + args.model_root.split('/',6)[2] +'_'+ args.model_root.split('/',6)[3]+ '/picture/')

    test_img_paths = glob('/root/pycharm_project/datasets/generalize/CVC-ClinicDB/test/imgs/*')    # 需要修改   # autodl-tmp/pycharm_project
    test_mask_paths = glob('/root/pycharm_project/datasets/generalize/CVC-ClinicDB/test/masks/*')  # 需要修改   # brainMRI-SEG/CVC-ClinicDB

    net = PEC_Net(3, 1)    # 需要修改


    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)

    net.load_state_dict(torch.load(args.model_root + args.model, map_location=device))

    logging.info("Model loaded !")

    sensitivity = []
    specificity = []
    Accuracy = []
    # precision = []
    f1_score = []
    iou = []
    hau_d = []
    asd = []
    Dice = []
    Jc = []
    Asd = []
    Hd95 = []

    for i in tqdm(range(len(test_img_paths))):
        img = Image.open(test_img_paths[i])
        img = img.convert("RGB")
        mask = Image.open(test_mask_paths[i])
        mask = mask.convert("L")
        w, h = mask.size
        newW, newH = int(args.scale * w), int(args.scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'   #判断语句，小于0，程序停止
        pil_img = mask.resize((256, 256))
        mask_nd = np.array(pil_img)  # np.array函数的作用：列表不存在维度问题，但数组是有维度的，而np.array()的作用就是把列表转化为数组，也可以说是用来产生数组。
        mask_s = mask_nd.astype('float32') / 255  #函数可用于转化dateframe的数据类型

        pd = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)
        # print(pd.shape) # (256, 256)
        # print(mask_s.shape) # (256, 256)
        # lm, ty, acc, pre, f1 = hunxiao(pd, mask_s)
        f1=dice_coeff(pd, mask_s)
        # dice = metric.binary.dc(pd, mask_s)
        # jc = metric.binary.jc(pd, mask_s)
        # asd = metric.binary.asd(pd, mask_s)
        # hd95 = metric.binary.hd95(pd, mask_s)
        jaccard = iou_score(pd,mask_s)
        hd = haussdorf(pd,mask_s)
        # surface_distances = surfdist.compute_surface_distances(mask_nd.astype('bool'), pd, spacing_mm=(1.0, 1.0))
        # ASD = surfdist.compute_average_surface_distance(surface_distances)
        # ASD = compute_ASD(pd, mask_s, (0.5, 0.5) )
        result = mask_to_image(pd) #实现array到image的转换
        # result.save('../predict_output/'+args.model_root.split('/',6)[2] +'_'+args.model_root.split('/',6)[3]+ '/picture/' + os.path.basename(test_img_paths[i]))

        # sensitivity.append(lm)
        # specificity.append(ty)
        # Accuracy.append(acc)
        # # precision.append(pre)
        f1_score.append(f1)

        iou.append(jaccard)
        hau_d.append(hd)
        # asd.append(ASD)

    print('sensitivity: %.4f' % np.mean(sensitivity))
    print('specificity: %.4f' % np.mean(specificity))
    print('Accuracy: %.4f' % np.mean(Accuracy))
    # print('precision: %.4f' % np.mean(precision))
    print('dice(f1_score): %.4f' % np.mean(f1_score))
    print('Jaccard(iou): %.4f' % np.mean(iou))
    print('HD: %.4f' % np.mean(hau_d))
    # print('ASD: %.4f' % np.mean(asd))
    end = time.perf_counter() #结束时间
    print('Running time: %s Seconds' % (end - start))
