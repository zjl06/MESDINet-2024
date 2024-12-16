import torch
import torch.nn as nn
import torch.nn.functional as F

import backbone.resnet.resnet as resnet
from functools import partial
from math import log
import math
from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply

class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

###################################################################
# ##################SimAM######################
###################################################################
class SimAM(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)

###################################################################
# ################## Channel Attention Block ######################
###################################################################
class CA_Block(nn.Module):
    def __init__(self, in_dim):
        super(CA_Block, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps (B X C X H X W)
            returns :
                out : channel attentive features
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

###################################################################
# ################## Spatial Attention Block ######################
###################################################################
class SA_Block(nn.Module):
    def __init__(self, in_dim):
        super(SA_Block, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps (B X C X H X W)
            returns :
                out : spatial attentive features
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

###################################################################
# ################## Context Exploration Block ####################
###################################################################
class Context_Exploration_Block(nn.Module):
    def __init__(self, input_channels):
        super(Context_Exploration_Block, self).__init__()
        self.input_channels = input_channels
        self.channels_single = int(input_channels / 4)

        self.p1_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p1 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p1_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p2 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 3, 1, 1),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p3 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 5, 1, 2),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p4 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 7, 1, 3),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.fusion = nn.Sequential(nn.Conv2d(self.input_channels, self.input_channels, 1, 1, 0),
                                    nn.BatchNorm2d(self.input_channels), nn.ReLU())

    def forward(self, x):
        p1_input = self.p1_channel_reduction(x)
        p1 = self.p1(p1_input)
        p1_dc = self.p1_dc(p1)

        p2_input = self.p2_channel_reduction(x) + p1_dc
        p2 = self.p2(p2_input)
        p2_dc = self.p2_dc(p2)

        p3_input = self.p3_channel_reduction(x) + p2_dc
        p3 = self.p3(p3_input)
        p3_dc = self.p3_dc(p3)

        p4_input = self.p4_channel_reduction(x) + p3_dc
        p4 = self.p4(p4_input)
        p4_dc = self.p4_dc(p4)

        ce = self.fusion(torch.cat((p1_dc, p2_dc, p3_dc, p4_dc), 1))

        return ce

###################################################################
# ##################### Positioning Module ########################
###################################################################
class Positioning(nn.Module):
    def __init__(self, channel):
        super(Positioning, self).__init__()
        self.channel = channel
        self.cab = CA_Block(self.channel)
        self.sab = SA_Block(self.channel)
        self.map = nn.Conv2d(self.channel, 1, 7, 1, 3)

    def forward(self, x):
        cab = self.cab(x)
        sab = self.sab(cab)
        map = self.map(sab)

        return sab, map

###################################################################
# ######################## Focus Module ###########################
###################################################################
class Focus(nn.Module):
    def __init__(self, channel1, channel2):
        super(Focus, self).__init__()
        self.channel1 = channel1
        self.channel2 = channel2

        self.up = nn.Sequential(nn.Conv2d(self.channel2, self.channel1, 7, 1, 3),
                                nn.BatchNorm2d(self.channel1), nn.ReLU(), nn.UpsamplingBilinear2d(scale_factor=2))

        self.input_map = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2), nn.Sigmoid())
        self.output_map = nn.Conv2d(self.channel1, 1, 7, 1, 3)

        self.fp = Context_Exploration_Block(self.channel1)
        self.fn = Context_Exploration_Block(self.channel1)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.bn1 = nn.BatchNorm2d(self.channel1)
        self.relu1 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(self.channel1)
        self.relu2 = nn.ReLU()


    def forward(self, x, y, in_map):
        # x; current-level features
        # y: higher-level features
        # in_map: higher-level prediction

        up = self.up(y)

        input_map = self.input_map(in_map)
        f_feature = x * input_map
        b_feature = x * (1 - input_map)

        fp = self.fp(f_feature)
        fn = self.fn(b_feature)

        refine1 = up - (self.alpha * fp)
        refine1 = self.bn1(refine1)
        refine1 = self.relu1(refine1)

        refine2 = refine1 + (self.beta * fn)
        refine2 = self.bn2(refine2)
        refine2 = self.relu2(refine2)

        output_map = self.output_map(refine2)

        return refine2, output_map

###################################################################
# ############# SimAM based Edge Prior Module #####################
###################################################################
class SEPM(nn.Module):
    def __init__(self):
        super(SEPM, self).__init__()
        self.reduce1 = Conv1x1(256, 64)
        self.reduce4 = Conv1x1(3840, 256)
        self.block = nn.Sequential(
            ConvBNR(256 + 64, 256, 3),
            ConvBNR(256, 256, 3),
            nn.Conv2d(256, 1, 1))
        self.simam=SimAM()

    def forward(self, x4, x1):
        size = x1.size()[2:]
        x1 = self.reduce1(x1)
        x1=self.simam(x1)
        x4 = self.reduce4(x4)
        x4 = self.simam(x4)
        x4 = F.interpolate(x4, size, mode='bilinear', align_corners=False)
        out = torch.cat((x4, x1), dim=1)
        out = self.block(out)
        out=self.simam(out)

        return out

###################################################################
# ###########Multiscale Detail Infusion Module#####################
###################################################################
class MDIM(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.convs = nn.ModuleList(
            [nn.Conv2d(c, channel[1], kernel_size=3, stride=1, padding=1) for c in channel])

    def forward(self, x):
        ans = torch.ones_like(x[1])
        target_size = x[1].shape[-2:]

        for i, x in enumerate(x):
            if x.shape[-1] > target_size[0]:
                x = F.adaptive_avg_pool2d(x, (target_size[0], target_size[1]))
            elif x.shape[-1] < target_size[0]:
                x = F.interpolate(x, size=(target_size[0], target_size[1]),
                                      mode='bilinear', align_corners=True)

            ans = ans * self.convs[i](x)
        return ans


###################################################################
# ##########Multiscale Edge Enhanced Guided Module#################
###################################################################
class MEEGM(nn.Module):
    def __init__(self, in_dim, hidden_dim=32, width=4, norm = nn.BatchNorm2d, act=nn.ReLU):
        super().__init__()
        t = int(abs((log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1, bias=False),
            norm(hidden_dim),
            nn.Sigmoid()
        )
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.conv2d = ConvBNR(channel, channel, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.eage_enhance=EdgeEnhancer(in_dim=channel)
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.mid_conv = nn.ModuleList()
        self.edge_enhance = nn.ModuleList()
        for i in range(width - 1):
            self.mid_conv.append(nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
                norm(hidden_dim),
                nn.Sigmoid()
            ))
            self.edge_enhance.append(EdgeEnhancer(hidden_dim))
        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * width, in_dim, 1, bias=False),
            norm(in_dim),
            act()
        )

    def forward(self, x, att):
        mid = self.in_conv(x)
        out = mid
        for i in range(self.width - 1):
            mid = self.pool(mid)
            mid = self.mid_conv[i](mid)
            out = torch.cat([out, self.edge_enhance[i](mid)], dim=1)
        out = self.out_conv(out)
        if out.size() != att.size():
            att = F.interpolate(att, c.size()[2:], mode='bilinear', align_corners=False)
        c = out * att + out
        c = self.conv2d(c)
        f = self.avg_pool(c)
        f = self.conv1d(f.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        f = self.sigmoid(f)
        c = c * f
        return c

class EdgeEnhancer(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.Sigmoid()
        )
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

    def forward(self, x):
        edge = self.pool(x)
        edge = x - edge
        edge = self.out_conv(edge)
        return x + edge

###################################################################
# ##################SE ASPP ######################
###################################################################

class SE_Block(nn.Module):
    def __init__(self, in_planes):
        super(SE_Block, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(in_planes, in_planes // 16, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_planes // 16, in_planes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        out = self.sigmoid(x)
        return out

class SE_ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(SE_ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        print('dim_in:',dim_in)
        print('dim_out:',dim_out)
        self.senet=SE_Block(in_planes=dim_out*5)

    def forward(self, x):
        [b, c, row, col] = x.size()
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        print('feature:',feature_cat.shape)
        seaspp1=self.senet(feature_cat)             
        print('seaspp1:',seaspp1.shape)
        se_feature_cat=seaspp1*feature_cat
        result = self.conv_cat(se_feature_cat)
        print('result:',result.shape)
        return result
###################################################################
# ##################### bottleneck Module ########################
###################################################################
class bottleneck(nn.Module):
    def __init__(self, channel):
        super(bottleneck, self).__init__()
        self.channel = channel
        self.aspp=SE_ASPP(self.channel,512)
        self.map = nn.Conv2d(self.channel, 1, 7, 1, 3)

    def forward(self, x):
        m = self.aspp(x)

        return m

###################################################################
# ################### Overall Architecture #######################
###################################################################
class MESDINet(nn.Module):
    def __init__(self, backbone_path=None):
        super(MESDINet, self).__init__()
        resnet50 = resnet.resnet50(backbone_path)
        self.layer0 = nn.Sequential(resnet50.conv1, resnet50.bn1, resnet50.relu)
        self.layer1 = nn.Sequential(resnet50.maxpool, resnet50.layer1)
        self.layer2 = resnet50.layer2
        self.layer3 = resnet50.layer3
        self.layer4 = resnet50.layer4
        self.cr4 = nn.Sequential(nn.Conv2d(2048, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU())
        self.cr3 = nn.Sequential(nn.Conv2d(1024, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU())
        self.cr2 = nn.Sequential(nn.Conv2d(512, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.cr1 = nn.Sequential(nn.Conv2d(256, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.positioning = Positioning(512)
        self.cem = Context_Exploration_Block(512, 256, 128, 64)
        self.bon4 = bottleneck(512)
        self.sepm = SEPM()
        self.meegm1 = MEEGM(64)
        self.meegm2 = MEEGM(128)
        self.meegm3 = MEEGM(256)
        self.meegm4 = MEEGM(512)
        self.mdim = MDIM([64, 128, 256, 512])
        self.focus3 = Focus(256, 512)
        self.focus2 = Focus(128, 256)
        self.focus1 = Focus(64, 128)
        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True
        self.cxt_num = [0, 3][1]
        self.lateral_channels_in_collection = [256, 512, 1024, 2048]
        self.cxt = self.lateral_channels_in_collection[1:][::-1][-self.cxt_num:] if self.cxt_num else []

    def forward(self, x):
        layer0 = self.layer0(x)  
        layer1 = self.layer1(layer0) 
        layer2 = self.layer2(layer1) 
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        layer4 = torch.cat(
            (
                *[
                     F.interpolate(layer1, size=layer4.shape[2:], mode='bilinear', align_corners=True),
                     F.interpolate(layer2, size=layer4.shape[2:], mode='bilinear', align_corners=True),
                     F.interpolate(layer3, size=layer4.shape[2:], mode='bilinear', align_corners=True),
                 ][-len(self.cxt):],
                layer4
            ),
            dim=1
        )
        fe = self.sepm(layer4, layer1)
        cr4 = self.cr4(layer4)
        cr3 = self.cr3(layer3)
        cr2 = self.cr2(layer2)
        cr1 = self.cr1(layer1)

        edge_att = torch.sigmoid(fe)
        x1a = self.meegm1(cr1, edge_att)
        x2a = self.meegm2(cr2, edge_att)
        x3a = self.meegm3(cr3, edge_att)
        x4a = self.meegm4(cr4, edge_att)
        inputs = [x1a, x2a, x3a, x4a]
        output = self.mdim(inputs)
        cr2 = output + cr2
        cr4, cr3, cr2, cr1 = self.cem(cr4, cr3, cr2, cr1)
        cr4 = self.bon4(cr4)

        positioning, predict4 = self.positioning(cr4)
        focus3, predict3 = self.focus3(cr3, positioning, predict4)
        focus2, predict2 = self.focus2(cr2, focus3, predict3)
        focus1, predict1 = self.focus1(cr1, focus2, predict2)
        predict4 = F.interpolate(predict4, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict3 = F.interpolate(predict3, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict2 = F.interpolate(predict2, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict1 = F.interpolate(predict1, size=x.size()[2:], mode='bilinear', align_corners=True)

        if self.training:
            return predict4, predict3, predict2, predict1

        return torch.sigmoid(predict4), torch.sigmoid(predict3), torch.sigmoid(predict2), torch.sigmoid(
            predict1)
