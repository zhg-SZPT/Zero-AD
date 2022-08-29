import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)


        self.channel_pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.channel_pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.channel_conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.channel_bn1 = nn.BatchNorm2d(mip)
        self.channel_act = h_swish()

        self.channel_conv_h = nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0)
        self.channel_conv_w = nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0)

        self.channel_attention = ChannelAttentionModule(channel=inp, reduction=16)

    # def channel_attention(self, x):
    #     n, c, h, w = x.size()
    #     x_h = self.channel_pool_h(x)    #[32, 128, 64, 1]
    #     x_w = self.channel_pool_w(x).permute(0, 1, 3, 2)   #[32, 128, 1, 64]->[32, 128, 64, 1]
    #
    #     y = torch.cat([x_h, x_w], dim=2)  # [32, 128, 128, 1]
    #     y = self.conv1(y)  # [32, 8, 128, 1]
    #     y = self.bn1(y)  # [32, 8, 128, 1]
    #     y = self.act(y)  # [32, 8, 128, 1]
    #
    #     x_h, x_w = torch.split(y, [h, w], dim=2)
    #     x_w = x_w.permute(0, 1, 3, 2)
    #
    #     a_h = self.conv_h(x_h).sigmoid()  # [32, 128, 64, 1]
    #     a_w = self.conv_w(x_w).sigmoid()  # [32, 128, 1, 64]
    #
    #     a_h = a_h.repeat(1,1,1,h)
    #     a_w = a_w.repeat(1,1,w,1)
    #     x = x * a_h * a_w
    #     return x
        

    def forward(self, x):
        identity = x

        #x = self.channel_attention(x)
        x = self.channel_attention(x) * x
        
        n,c,h,w = x.size()            #[32, 128, 64, 64]
        x_h = self.pool_h(x)          #[32, 128, 64, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)   #[32, 128, 1, 64]->[32, 128, 64, 1]

        y = torch.cat([x_h, x_w], dim=2)   #[32, 128, 128, 1]
        y = self.conv1(y)                  #[32, 8, 128, 1]
        y = self.bn1(y)                    #[32, 8, 128, 1]
        y = self.act(y)                    #[32, 8, 128, 1]
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()   #[32, 128, 64, 1]
        a_w = self.conv_w(x_w).sigmoid()   #[32, 128, 1, 64]

        out = identity * a_w * a_h

        return out

# 通道注意力模块
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel=256, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        mid_channel = channel // reduction
        # 使用自适应池化缩减map的大小，保持通道不变
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=channel, out_features=mid_channel),
            nn.ReLU(),
            nn.Linear(in_features=mid_channel, out_features=channel)
        )
        self.sigmoid = nn.Sigmoid()
        # self.act=SiLU()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        # print('avg',avgout.size())  # [16, 256, 1, 1]
        maxout = self.shared_MLP(self.max_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        # print('max',maxout.size())  # [16, 256, 1, 1]
        return self.sigmoid(avgout + maxout)
