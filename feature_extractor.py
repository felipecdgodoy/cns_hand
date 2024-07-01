import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
from metadatanorm2 import MetadataNorm
import pdb

class FeatureExtractorPaired(nn.Module):
    def __init__(self, trainset_size = 0  , in_num_ch=1, img_size=(32,64,64), inter_num_ch=16, kernel_size=3, conv_act='relu',dropout=0.2, batch_size = 80):
        super(FeatureExtractorPaired, self).__init__()
        self.tau = Parameter(torch.ones(2048).float())
        # initial "neutral" tau direction, which gets learned in training
        ## Encode to 2048-vector, set as parameter for easy access during training

        self.relu_maxpool_cur_1 = nn.Sequential(
                             nn.ReLU(inplace=True),
                             nn.MaxPool3d(2)
                        )

        self.relu_maxpool_cur_2 = nn.Sequential(
                            nn.ReLU(inplace=True),
                            nn.MaxPool3d(2)
                        )

        self.relu_maxpool_cur_3 = nn.Sequential(
                            nn.ReLU(inplace=True),
                            nn.MaxPool3d(2)
                        )
        self.relu_maxpool_cur_4 = nn.Sequential(
                            nn.ReLU(inplace=True),
                            nn.MaxPool3d(2)
                        )

        self.conv_1s_1 = nn.Sequential(nn.Conv3d(in_num_ch, inter_num_ch, kernel_size=1,stride=1))
        self.conv_1s_2 = nn.Sequential(nn.Conv3d(inter_num_ch, 2*inter_num_ch, kernel_size=1,stride=1))
        self.conv_1s_3 = nn.Sequential(nn.Conv3d(2*inter_num_ch, 4*inter_num_ch, kernel_size=1,stride=1))
        self.conv_1s_4 = nn.Sequential(nn.Conv3d(4*inter_num_ch, 2*inter_num_ch, kernel_size=1,stride=1))

        self.conv1 = nn.Sequential(
                        nn.Conv3d(in_num_ch, inter_num_ch, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
        )
        self.conv1_b = nn.Sequential(
                        nn.Conv3d(inter_num_ch, inter_num_ch, kernel_size=3, padding=1),
        )

        self.conv2 = nn.Sequential(
                        nn.Conv3d(inter_num_ch, 2*inter_num_ch, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
        )
        self.conv2_b = nn.Sequential(
                        nn.Conv3d(2*inter_num_ch, 2*inter_num_ch, kernel_size=3, padding=1),
        )

        self.conv3 = nn.Sequential(
                        nn.Conv3d(2*inter_num_ch, 4*inter_num_ch, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),

        )
        self.conv3_b = nn.Sequential(
                        nn.Conv3d(4*inter_num_ch, 4*inter_num_ch, kernel_size=3, padding=1),
#
        )

        self.conv4 = nn.Sequential(
                        nn.Conv3d(4*inter_num_ch, 2*inter_num_ch, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),

        )
        self.conv4_b = nn.Sequential(
                        nn.Conv3d(2*inter_num_ch, 2*inter_num_ch, kernel_size=3, padding=1),
        )
    def forward(self, x):
        out1 =  self.conv1(x)
        out1 =  self.conv1_b(out1)
        out1 += self.conv_1s_1(x)
        out1 = self.relu_maxpool_cur_1(out1)

        out2 =  self.conv2(out1)
        out2 =  self.conv2_b(out2)
        out2 += self.conv_1s_2(out1)
        out2 = self.relu_maxpool_cur_2(out2)

        out3 =  self.conv3(out2)
        out3 =  self.conv3_b(out3)
        out3 += self.conv_1s_3(out2)
        out3 = self.relu_maxpool_cur_3(out3)

        out4 =  self.conv4(out3)
        out4 =  self.conv4_b(out4)
        out4 += self.conv_1s_4(out3)
        out4 = self.relu_maxpool_cur_4(out4)

        return out4
class FeatureExtractorRes(nn.Module):
    def __init__(self, cf_kernel, trainset_size = 0  , in_num_ch=1, img_size=(32,64,64), inter_num_ch=16, kernel_size=3, conv_act='relu',dropout=0.2, batch_size = 80):
        super(FeatureExtractorRes, self).__init__()
        self.relu_maxpool_cur_1 = nn.Sequential(
                             nn.ReLU(inplace=True),
                             nn.MaxPool3d(2)
                        )

        self.relu_maxpool_cur_2 = nn.Sequential(
                            nn.ReLU(inplace=True),
                            nn.MaxPool3d(2)
                        )

        self.relu_maxpool_cur_3 = nn.Sequential(
                            nn.ReLU(inplace=True),
                            nn.MaxPool3d(2)
                        )
        self.relu_maxpool_cur_4 = nn.Sequential(
                            nn.ReLU(inplace=True),
                            nn.MaxPool3d(2)
                        )

        self.conv_1s_1 = nn.Sequential(nn.Conv3d(in_num_ch, inter_num_ch, kernel_size=1,stride=1))
        self.conv_1s_2 = nn.Sequential(nn.Conv3d(inter_num_ch, 2*inter_num_ch, kernel_size=1,stride=1))
        self.conv_1s_3 = nn.Sequential(nn.Conv3d(2*inter_num_ch, 4*inter_num_ch, kernel_size=1,stride=1))
        self.conv_1s_4 = nn.Sequential(nn.Conv3d(4*inter_num_ch, 2*inter_num_ch, kernel_size=1,stride=1))

        self.conv1 = nn.Sequential(
                        nn.Conv3d(in_num_ch, inter_num_ch, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
        )
        self.conv1_b = nn.Sequential(
                        nn.Conv3d(inter_num_ch, inter_num_ch, kernel_size=3, padding=1),
        )

        self.conv2 = nn.Sequential(
                        nn.Conv3d(inter_num_ch, 2*inter_num_ch, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
        )
        self.conv2_b = nn.Sequential(
                        nn.Conv3d(2*inter_num_ch, 2*inter_num_ch, kernel_size=3, padding=1),
        )

        self.conv3 = nn.Sequential(
                        nn.Conv3d(2*inter_num_ch, 4*inter_num_ch, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
        )
        self.conv3_b = nn.Sequential(
                        nn.Conv3d(4*inter_num_ch, 4*inter_num_ch, kernel_size=3, padding=1),
        )

        self.conv4 = nn.Sequential(
                        nn.Conv3d(4*inter_num_ch, 2*inter_num_ch, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
        )
        self.conv4_b = nn.Sequential(
                        nn.Conv3d(2*inter_num_ch, 2*inter_num_ch, kernel_size=3, padding=1),
        )


    def forward(self, x):
        out1 =  self.conv1(x)
        out1 =  self.conv1_b(out1)
        out1 += self.conv_1s_1(x)
        out1 = self.relu_maxpool_cur_1(out1)

        out2 =  self.conv2(out1)
        out2 =  self.conv2_b(out2)
        out2 += self.conv_1s_2(out1)
        out2 = self.relu_maxpool_cur_2(out2)

        out3 =  self.conv3(out2)
        out3 =  self.conv3_b(out3)
        out3 += self.conv_1s_3(out2)
        out3 = self.relu_maxpool_cur_3(out3)

        out4 =  self.conv4(out3)
        out4 =  self.conv4_b(out4)
        out4 += self.conv_1s_4(out3)
        out4 = self.relu_maxpool_cur_4(out4)

        return out4


class FeatureExtractor(nn.Module):
    def __init__(self, cf_kernel, trainset_size = 0  , in_num_ch=1, img_size=(32,64,64), inter_num_ch=16, kernel_size=3, conv_act='relu',dropout=0.2, batch_size = 80):
        super(FeatureExtractor, self).__init__()

        self.conv1 = nn.Sequential(
                        nn.Conv3d(in_num_ch, inter_num_ch, kernel_size=3, padding=1),
                        nn.LeakyReLU(inplace=True),
                        nn.MaxPool3d(2))

        self.conv2 = nn.Sequential(
                        nn.Conv3d(inter_num_ch, 2*inter_num_ch, kernel_size=3, padding=1),
                        nn.LeakyReLU(inplace=True),
                        nn.MaxPool3d(2))

        self.conv3 = nn.Sequential(
                        nn.Conv3d(2*inter_num_ch, 4*inter_num_ch, kernel_size=3, padding=1),
                        nn.LeakyReLU(inplace=True),
                        nn.MaxPool3d(2))

        self.conv4 = nn.Sequential(
                        nn.Conv3d(4*inter_num_ch, 2*inter_num_ch, kernel_size=3, padding=1),
                        nn.LeakyReLU(inplace=True),
                        MetadataNorm(batch_size=batch_size, cf_kernel=cf_kernel, num_features = 32*8*8*8, trainset_size = trainset_size, momentum=0.9),
                        nn.MaxPool3d(2))


    def forward(self, x):

        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        return conv4

class fe(nn.Module):
    def __init__(self, trainset_size = 0 ,in_num_ch=1, img_size=(32,64,64), inter_num_ch=16,
                 fc_num_ch=16, kernel_size=3, conv_act='relu',
                 fe_arch='baseline',dropout=0.2,fc_dropout = 0.2, batch_size = 80):
        super().__init__()
        if fe_arch == 'baseline' or fe_arch == 'fe1':
            self.feature_extractor = FeatureExtractorPaired(trainset_size, in_num_ch, img_size, inter_num_ch, kernel_size, conv_act,dropout, batch_size)
            num_feat = int(2*inter_num_ch * (img_size[0]*img_size[1]*img_size[2]) / ((2**4)**3))
        else:
            raise ValueError('Not yet Implemented')


    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(x.shape[0], -1)
        return features
