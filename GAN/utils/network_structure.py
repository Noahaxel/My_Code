import torch.nn as nn


ndf=64 #判别网络卷积核个数的倍数
ngf=64 #生成网络卷积核个数的倍数


"""
关于转置卷积：
当padding=0时,卷积核刚好和输入边缘相交一个单位。因此pandding可以理解为卷积核向中心移动的步数。 
同时stride也不再是kernel移动的步数,变为输入单元彼此散开的步数,当stride等于1时,中间没有间隔。
"""

#生成器网络G
class generator(nn.Module):
    def __init__(self,noise_number,number_of_channels):
        """
        noise_number:输入噪声点个数
        number_of_channels:生成图像通道数
        """
        super(generator,self).__init__()
        self.gen = nn.Sequential(
            # 输入大小  batch x noise_number x 1 * 1
            nn.ConvTranspose2d(noise_number , ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 输入大小 batch x (ngf*4) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 8, 2, 2, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 输入大小 batch x (ngf*8) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 8, 2, 2, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 输入大小 batch x (ngf*8) x 16 x 16
            nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 输入大小 batch x (ngf*8) x 32 x 32
            nn.ConvTranspose2d(ngf * 8, ngf * 8, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 输入大小 batch x (ngf*8) x 63 x 63
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 输入大小 batch x (ngf*4) x 125 x 125
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 输入大小 batch x (ngf*2) x 128 x 128
            nn.ConvTranspose2d(ngf * 2, ngf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 输入大小 batch x (ngf*2) x 128 x 128
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 输入大小 batch x (ngf) x 256 x 256
            nn.ConvTranspose2d(ngf   , number_of_channels, 3, 1, 1, bias=False),
            nn.Tanh()
            # 输出大小 batch x (nc) x 256 x 256
       )

    def forward(self, x):
        out = self.gen(x)
        return out
    
#判别器网络D
class discriminator(nn.Module):
    def __init__(self,number_of_channels):
        """
        number_of_channels:输入图像通道数
        """
        super(discriminator, self).__init__()
        self.dis = nn.Sequential(
            # 输入大小 batch x g_d_nc x 256*256
            nn.Conv2d(number_of_channels, ndf  , 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf ),
            nn.LeakyReLU(0.2, inplace=True),
            # 输入大小 batch x ndf x 256*256
            nn.Conv2d(ndf , ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 输入大小 batch x (ndf*2) x 128*128
            nn.Conv2d(ndf * 2, ndf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 输入大小 batch x (ndf*2) x 128*128
            nn.Conv2d(ndf * 2, ndf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 输入大小 batch x (ndf*4) x 125*125
            nn.Conv2d(ndf * 4 , ndf * 8, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 输入大小 batch x (ndf*8) x 63*63
            nn.Conv2d(ndf * 8 , ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 输入大小 batch x (ndf*4) x 32*32
            nn.Conv2d(ndf * 4 , ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 输入大小 batch x (ndf*2) x 16*16
            nn.Conv2d(ndf * 2 , ndf , 2, 2, 0, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # 输入大小 batch x (ndf) x 8*8
            nn.Conv2d(ndf, ndf , 2, 2, 0, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # 输入大小 batch x (ndf) x 4*4
            nn.Conv2d(ndf , 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # 输出大小 batch x 1 x 1*1
        )

    def forward(self, x):
        x=self.dis(x).view(x.shape[0],-1)
        return x