import numpy as np 
import torch as t 
import torch.nn as nn 
from torch.nn.modules import Sequential
 

class ConvLayer(nn.Module):
    def __init__(self, in_channels,out_channels,kernel_size,stride):
        super(ConvLayer,self).__init__()
        zero_padding = int(np.floor(kernel_size / 2))
        self.zero_pad = nn.ZeroPad2d(zero_padding)
        self.conv2d = nn.Conv2d(in_channels,out_channels,kernel_size,stride)
    def forward(self,x):
        out = self.zero_pad(x)
        out = self.conv2d(out)
        return out


class MyEncoder(nn.Module):
    def __init__(self,K):
        super(MyEncoder, self).__init__()
        self.K = K
        self.fc = nn.Linear(self.K**2,self.K**2)
        self.relu = nn.ReLU()
        self.conv_block = nn.Sequential(
            ConvLayer(1,8,kernel_size=3,stride=1), # 第一个卷积核的in_channel=1
            nn.ReLU(),
            ConvLayer(8,8,kernel_size=3,stride=1),
            nn.ReLU()
        )
    def forward(self,x):
        x = x.view(x.size(0),-1)
        x = self.fc(x) 
        x = x.view(x.size(0),1,self.K,self.K) 
        x = self.conv_block(x)
        return x
        
        


class MyDecoder(nn.Module):
    def __init__(self,K):
        super(MyDecoder, self).__init__()
        self.K = K
        self.conv_block = nn.Sequential(
            ConvLayer(8,8,kernel_size=3,stride=1),
            nn.ReLU(),
            ConvLayer(8,1,kernel_size=3,stride=1),
            nn.ReLU()
        )
        self.fc = nn.Linear(self.K**2,self.K)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.conv_block(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x




class MBPANet(nn.Module):
    def __init__(self, K):
        super(MBPANet, self).__init__()
        self.K = K
        self.encoder_net = MyEncoder(self.K)
        self.decoder_net = MyDecoder(self.K)
        self.alg_bank = nn.ModuleList([Sequential(
            ConvLayer(8,8,kernel_size=3,stride=1),
            nn.ReLU()
        ) 
        for i in range(2)])

    def forward(self,x,alg_id=None):
        z = self.encoder_net(x)
        if alg_id is not None:
            z = self.alg_bank[alg_id](z)
        z = self.decoder_net(z)

        return z

         
# test
if __name__ == "__main__":
    x = t.ones(1, 1, 20, 20)
    net = MBPANet(20)
    out = net(x,1)
    print(out)