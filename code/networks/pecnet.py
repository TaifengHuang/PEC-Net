# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
from torchstat import stat
import torch
import torch.nn as nn
import pdb
from torch.nn import functional as F
from ptflops import get_model_complexity_info
from torch.distributions.uniform import Uniform
# from networks.resnet import resnet34_2d


class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p, mode_upsampling=1):
        super(UpBlock, self).__init__()
        self.mode_upsampling = mode_upsampling
        if mode_upsampling == 0:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
        elif mode_upsampling == 1:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 双线性插值
        elif mode_upsampling == 2:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='nearest')   # 最邻近插值
        elif mode_upsampling == 3:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)   # 双三次插值
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.mode_upsampling != 0:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(
            self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.up_type = self.params['up_type']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0,
                           mode_upsampling=self.up_type)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0,
                           mode_upsampling=self.up_type)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0,
                           mode_upsampling=self.up_type)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0,
                           mode_upsampling=self.up_type)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)
    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)
        return output

class Conv3x3(nn.Module):
    def __init__(self, in_chns, out_chns):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(in_chns, out_chns, kernel_size=3, padding=1)
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        dim_in = 16
        feat_dim = 32
        self.projection_head = nn.Sequential(
            nn.Linear(dim_in, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        for class_c in range(4):
            selector = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(feat_dim, 1)
            )
            self.__setattr__('contrastive_class_selector_' + str(class_c), selector)

        for class_c in range(4):
            selector = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(feat_dim, 1)
            )
            self.__setattr__('contrastive_class_selector_memory' + str(class_c), selector)

    def forward_projection_head(self, features):
        return self.projection_head(features)

    def forward_prediction_head(self, features):
        return self.prediction_head(features)

    def forward(self, x):
        feature = self.encoder(x)
        output, features = self.decoder(feature)
        return output, features


class ResUNet_2d(nn.Module):
    def __init__(self, in_chns, class_num):
        super(ResUNet_2d, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'acti_func': 'relu'}

        # self.encoder = resnet34_2d()
        self.decoder = Decoder(params)
        dim_in = 16
        feat_dim = 32
        self.projection_head = nn.Sequential(
            nn.Linear(dim_in, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        for class_c in range(4):
            selector = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(feat_dim, 1)
            )
            self.__setattr__('contrastive_class_selector_' + str(class_c), selector)

        for class_c in range(4):
            selector = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(feat_dim, 1)
            )
            self.__setattr__('contrastive_class_selector_memory' + str(class_c), selector)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward_projection_head(self, features):
        return self.projection_head(features)
        # return self.decoder(features)

    def forward_prediction_head(self, features):
        return self.prediction_head(features)

    def forward(self, x):
        feature = self.encoder(x)
        output, features = self.decoder(feature)
        return output

class BalancedFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BalancedFusionModule, self).__init__()

        # 1x1 Conv
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # 3x3 Conv with different dilation rates
        self.conv3x3_rate6 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)
        self.conv3x3_rate12 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
        self.conv3x3_rate18 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18)

        # Pooling and 1x1 Conv
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.pooling_conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Final 1x1 Conv after concatenation
        self.final_conv1x1 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)

        # Balanced Dropout (replaceable with standard dropout for simplicity)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, encoder_feature1, encoder_feature2):
        # Concatenate Encoder Features
        x = torch.cat([encoder_feature1, encoder_feature2], dim=1)

        # Branch 1: 1x1 Conv
        branch1 = self.conv1x1(x)

        # Branch 2, 3, 4: 3x3 Conv with different dilation rates
        branch2 = self.conv3x3_rate6(x)
        branch3 = self.conv3x3_rate12(x)
        branch4 = self.conv3x3_rate18(x)

        # Branch 5: Pooling, 1x1 Conv, and Upsampling
        pooled = self.pooling(x)  # Global pooling to 1x1
        branch5 = self.pooling_conv1x1(pooled)
        branch5 = F.interpolate(branch5, size=x.size()[2:], mode='bilinear',
                                align_corners=False)  # Upsample to original size

        # Concatenate all branches
        fused = torch.cat([branch1, branch2, branch3, branch4, branch5], dim=1)

        # Final 1x1 Conv
        fused = self.final_conv1x1(fused)

        # Balanced Dropout
        output = self.dropout(fused)

        return output


class PEC_Net(nn.Module):
    def __init__(self, in_chns, class_num):
        super(PEC_Net, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'acti_func': 'relu'}
        params1 = {'in_chns': in_chns,
                   'feature_chns': [16, 32, 64, 128, 256],
                   'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                   'class_num': class_num,
                   'up_type': 1,
                   'acti_func': 'relu'}
        params2 = {'in_chns': in_chns,
                   'feature_chns': [16, 32, 64, 128, 256],
                   'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                   'class_num': class_num,
                   'up_type': 0,
                   'acti_func': 'relu'}
        params3 = {'in_chns': in_chns,
                   'feature_chns': [16, 32, 64, 128, 256],
                   'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                   'class_num': class_num,
                   'up_type': 2,
                   'acti_func': 'relu'}

        self.encoder_unet = Encoder(params)
        # self.encoder_res = resnet34_2d()
        self.decoder1 = Decoder(params1)
        self.decoder2 = Decoder(params2)
        self.decoder3 = Decoder(params3)
    def forward(self, x, flag=13):
        feature_unet = self.encoder_unet(x)
        feature_res = self.encoder_unet(x)
        if flag == 2:
            feature1 = []
            feature2 = []
            for i in range(len(feature_unet)):
                # module = BalancedFusionModule(2*feature_unet[i].shape[1], feature_unet[i].shape[1]).cuda()
                # module1 = module(feature_unet[i], feature_res[i])
                module0 = torch.cat([feature_unet[i], feature_res[i]], dim=1)
                module = Conv3x3(module0.shape[1], feature_unet[i].shape[1]).cuda()
                module1 = module(module0)

                module2 = feature_unet[i]+feature_res[i]

                feature1.append(module1)
                feature2.append(module2)
            output_2 = self.decoder2(feature1)
            output_3 = self.decoder2(feature2)
            # output_5 = self.decoder2(feature1)
            return output_2, output_3
        # elif flag == 1:
        #     output_1 = self.decoder1(feature_unet)
        #     return output_1
        # elif flag == 4:
        #     output_4 = self.decoder3(feature_res)
        #     return output_4
        else:
            output_1 = self.decoder1(feature_unet)
            output_4 = self.decoder3(feature_res)
            return output_1, output_4

class ECDD_Net(nn.Module):
    def __init__(self, in_chns, class_num):
        super(ECDD_Net, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'acti_func': 'relu'}
        params1 = {'in_chns': in_chns,
                   'feature_chns': [16, 32, 64, 128, 256],
                   'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                   'class_num': class_num,
                   'up_type': 1,
                   'acti_func': 'relu'}
        params2 = {'in_chns': in_chns,
                   'feature_chns': [16, 32, 64, 128, 256],
                   'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                   'class_num': class_num,
                   'up_type': 0,
                   'acti_func': 'relu'}
        params3 = {'in_chns': in_chns,
                   'feature_chns': [16, 32, 64, 128, 256],
                   'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                   'class_num': class_num,
                   'up_type': 2,
                   'acti_func': 'relu'}

        self.encoder_unet = Encoder(params)
        self.decoder1 = Decoder(params1)
        self.decoder2 = Decoder(params2)# 转置卷积
        # self.decoder3 = Decoder(params3)
    def forward(self, x, flag=13):
        feature_unet = self.encoder_unet(x)
        feature_res = self.encoder_unet(x)
        # if flag == 2:
        feature1 = []
        feature2 = []
        for i in range(len(feature_unet)):
            module0 = torch.cat([feature_unet[i], feature_res[i]], dim=1)
            module = Conv3x3(module0.shape[1], feature_unet[i].shape[1]).cuda()
            module1 = module(module0)

            module2 = feature_unet[i]+feature_res[i]

            feature1.append(module1)
            feature2.append(module2)
        output_2 = self.decoder2(feature1)
        output_3 = self.decoder1(feature2)
        return output_2, output_3
        # elif flag == 1:
        #     output_1 = self.decoder1(feature_unet)
        #     return output_1
        # elif flag == 4:
        #     output_4 = self.decoder3(feature_res)
        #     return output_4
        # else:
        #     output_1 = self.decoder1(feature_unet)
        #     output_4 = self.decoder3(feature_res)
        #     return output_1, output_4



if __name__ == '__main__':
    # compute FLOPS & PARAMETERS
    from thop import profile
    from thop import clever_format

    # model = PEC_Net(in_chns=3, class_num=1)
    # input = torch.randn(2, 3, 256, 256)
    # total = sum([param.nelement() for param in model.parameters()])
    # print('Number of parameter: % .2fM' % (total / 1e6))
    #
    # stat(model, (3, 256, 256))

    model = PEC_Net(in_chns=1, class_num=4)

    total = sum([param.nelement() for param in model.parameters()])
    print('Number of parameter: % .2fM' % (total / 1e6))
    stat(model, (1, 256, 256))

    # with torch.cuda.device(0):
    #     macs, params = get_model_complexity_info(model, (1, 256, 256), as_strings=True,
    #                                              print_per_layer_stat=True, verbose=True)
    #     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    #     print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # output = model(input,2)
    # print(output[0].shape)
    # flops, params = profile(model, inputs=(input,))
    # macs, params = clever_format([flops, params], "%.3f")
    # print(macs, params)
    #
    # from ptflops import get_model_complexity_info
    #
    # with torch.cuda.device(0):
    #     macs, params = get_model_complexity_info(model, (1, 112, 112, 80), as_strings=True,
    #                                              print_per_layer_stat=True, verbose=True)
    #     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    #     print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # import pdb; pdb.set_trace()
