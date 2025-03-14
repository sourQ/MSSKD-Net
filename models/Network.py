import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from models.mvssnet import _PositionAttentionModule,_ChannelAttentionModule


class Convolution(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Convolution, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class Curvature(torch.nn.Module):
    def __init__(self, ratio,in_ch=0,out_ch=0):
        super(Curvature, self).__init__()
        weights = torch.tensor([[[[-1/16, 5/16, -1/16], [5/16, -1, 5/16], [-1/16, 5/16, -1/16]]]])
        self.weight = torch.nn.Parameter(weights).cuda()
        self.ratio = ratio
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        # self.pos_atten = _PositionAttentionModule(in_channels=out_ch)
        # self.chan_atten = _ChannelAttentionModule()
 
    def forward(self, x):
        B, C, H, W = x.size()
        x_origin = x
        x = x.reshape(B*C,1,H,W)
        out = F.conv2d(x, self.weight)
        out = torch.abs(out)
        p = torch.sum(out, dim=-1)
        p = torch.sum(p, dim=-1)
        p=p.reshape(B, C)

        _, index = torch.topk(p, int(self.ratio*C), dim=1)
        ##youhua for xunhuan
        index = index[...,None,None].expand(-1,-1,H,W)
        selected = []
        selected = torch.gather(x_origin,1,index)
        ##youhua for xunhuan
        ##origin
        # selected = []
        # for i in range(x_origin.shape[0]):
            # selected.append(torch.index_select(x_origin[i], dim=0, index=index[i]).unsqueeze(0))
        # selected = torch.cat(selected, dim=0)
        ##origin
        x1 = torch.cat([x_origin,selected],dim=1)
        out = self.conv(x1)
        # out = self.pos_atten(out)
        # out = self.chan_atten(out)

        return out

class distill_loss(nn.Module):
    def __init__(self):
        super(distill_loss, self).__init__()
        self.p = 2

    def forward(self, g_s, g_t):
        loss = sum([self.at_loss(f_s, f_t.detach()) for f_s, f_t in zip(g_s, g_t)])
        return loss

    def at_loss(self, f_s, f_t):
        return (self.at(f_s) - self.at(f_t)).pow(2).mean()

    def at(self, f):
        return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))
    

   
   
    
class Entropy_Hist(nn.Module):
    def __init__(self, ratio, win_w=3, win_h=3):
        super(Entropy_Hist, self).__init__()
        self.win_w = win_w
        self.win_h = win_h
        self.ratio = ratio

    def calcIJ_new(self, img_patch):
        total_p = img_patch.shape[-1] * img_patch.shape[-2]
        if total_p % 2 != 0:
            tem = torch.flatten(img_patch, start_dim=-2, end_dim=-1) 
            center_p = tem[:, :, :, int(total_p / 2)]
            mean_p = (torch.sum(tem, dim=-1) - center_p) / (total_p - 1)
            if torch.is_tensor(img_patch):
                return center_p * 100 + mean_p
            else:
                return (center_p, mean_p)
        else:
            print("modify patch size")

    def histc_fork(ij):
        BINS = 256
        B, C = ij.shape
        N = 16
        BB = B // N
        min_elem = ij.min()
        max_elem = ij.max()
        ij = ij.view(N, BB, C)

        def f(x):
            with torch.no_grad():
                res = []
                for e in x:
                    res.append(torch.histc(e, bins=BINS, min=min_elem, max=max_elem))
                return res
        futures : List[torch.jit.Future[torch.Tensor]] = []

        for i in range(N):
            futures.append(torch.jit.fork(f, ij[i]))

        results = []
        for future in futures:
            results += torch.jit.wait(future)
        with torch.no_grad():
            out = torch.stack(results)
        return out

    def forward(self, img):
        with torch.no_grad():
            B, C, H, W = img.shape
            ext_x = int(self.win_w / 2) # 考虑滑动窗口大小，对原图进行扩边，扩展部分长度
            ext_y = int(self.win_h / 2)

            new_width = ext_x + W + ext_x # 新的图像尺寸
            new_height = ext_y + H + ext_y
            
            # 使用nn.Unfold依次获取每个滑动窗口的内容
            nn_Unfold=nn.Unfold(kernel_size=(self.win_w,self.win_h),dilation=1,padding=ext_x,stride=1)
            # 能够获取到patch_img，shape=(B,C*K*K,L),L代表的是将每张图片由滑动窗口分割成多少块---->28*28的图像，3*3的滑动窗口，分成了28*28=784块
            x = nn_Unfold(img) # (B,C*K*K,L)
            x= x.view(B,C,3,3,-1).permute(0,1,4,2,3) # (B,C*K*K,L) ---> (B,C,L,K,K)
            ij = self.calcIJ_new(x).reshape(B*C, -1) # 计算滑动窗口内中心的灰度值和窗口内除了中心像素的灰度均值,(B,C,L,K,K)---> (B,C,L) ---> (B*C,L)
            
            fij_packed = self.histc_fork(ij)
            p = fij_packed / (new_width * new_height)
            h_tem = -p * torch.log(torch.clamp(p, min=1e-40)) / math.log(2)

            a = torch.sum(h_tem, dim=1) # 对所有二维熵求和，得到这张图的二维熵
            H = a.reshape(B,C) 

            _, index = torch.topk(H, int(self.ratio*C), dim=1) # Nx3
        selected = []
        for i in range(img.shape[0]):
            selected.append(torch.index_select(img[i], dim=0, index=index[i]).unsqueeze(0))
        selected = torch.cat(selected, dim=0)
        
        return selected
    

class Network(nn.Module):
    def __init__(self, in_ch=3, mode='ori', ratio=None):
        super(Network, self).__init__()
        self.mode = mode
        if self.mode == 'ori':
            self.ratio = [0,0]
        if self.mode == 'curvature':
            self.ratio = ratio
            self.ife1 = Curvature(self.ratio[0])
            self.ife2 = Curvature(self.ratio[1])
        if self.mode == 'entropy':
            self.ratio = ratio
            self.ife1 = Entropy_Hist(self.ratio[0])
            self.ife2 = Entropy_Hist(self.ratio[1])

        # ---- U-Net ----
        self.conv1 = Convolution(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)  # feature map = shape(m/2,n/2,64)
        self.conv2 = Convolution(64, 128)
        self.pool2 = nn.MaxPool2d(2)  # feature map = shapem/4,n/4,128)
        self.conv3 = Convolution(128, 256)
        self.pool3 = nn.MaxPool2d(2)  # feature map = shape(m/8,n/8,256)
        self.conv4 = Convolution(256, 512)
        self.pool4 = nn.MaxPool2d(2)  # feature map = shape(m/16,n/16,512)

        self.conv5 = Convolution(512, 1024)  # feature map = shape(m/16,n/16,1024)

        self.up_conv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2, padding=0, output_padding=0)  
        self.conv6 = Convolution(1024, 512)  # feature map = shape(m/8,n/8,512)
        self.up_conv2 = nn.ConvTranspose2d(512, 256, 2, 2, 0, 0)
        self.conv7 = Convolution(int(256*(2+self.ratio[1])), 256)  # feature map = shape(m/4,n/4,256）
        self.up_conv3 = nn.ConvTranspose2d(256, 128, 2, 2, 0, 0)
        self.conv8 = Convolution(int(128*(2+self.ratio[0])), 128)  # feature map = shape(m/2,n/2,128）
        self.up_conv4 = nn.ConvTranspose2d(128, 64, 2, 2, 0, 0)
        self.conv9 = Convolution(128, 64)  # feature map = shape(m,n,64)

        self.out_conv1 = nn.Conv2d(64, 1, 1, 1, 0) 

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        c3 = self.conv3(p2)
        p3 = self.pool3(c3)

        c4 = self.conv4(p3)
        p4 = self.pool4(c4)

        c5 = self.conv5(p4)

        if self.mode != 'ori':
            c2 = torch.cat([c2, self.ife1(c2)])
            c3 = torch.cat([c3, self.ife2(c3)])

        up1 = self.up_conv1(c5)
        merge1 = torch.cat([up1, c4], dim=1)
        c6 = self.conv6(merge1)
        up2 = self.up_conv2(c6)
        merge2 = torch.cat([up2, c3], dim=1)
        c7 = self.conv7(merge2)
        up3 = self.up_conv3(c7)
        merge3 = torch.cat([up3, c2], dim=1)
        c8 = self.conv8(merge3)
        up4 = self.up_conv4(c8)
        merge4 = torch.cat([up4, c1], dim=1)
        c9 = self.conv9(merge4)

        S_g_pred = self.out_conv1(c9) 

        return S_g_pred
