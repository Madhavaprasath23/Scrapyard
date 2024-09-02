import torch
import torch.nn as nn
from einops import rearrange
from math import sqrt

class CONVFFN(nn.Module):

    def __init__(self,stride=1,in_channels=1,hidden_dimension=1,norm_layer=nn.BatchNorm2d):
        super(CONVFFN,self).__init__()
        self.stride=stride
        self.in_channels=in_channels
        self.hidden_dimension=hidden_dimension
        self.act=nn.GELU()

        # first block
        self.dwconv1=nn.Conv2d(
            in_channels= self.in_channels,
            out_channels= self.in_channels,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=self.in_channels

        )
        self.pwconv1=nn.Conv2d(
            in_channels=self.in_channels,
            out_channels= self.hidden_dimension,
            kernel_size=1,
            stride=1
        )

        self.norm1=norm_layer(self.hidden_dimension)
        # second block 

        self.dwconv2=nn.Conv2d(
            in_channels=self.hidden_dimension,
            out_channels=self.hidden_dimension,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=self.hidden_dimension
        )
        self.pwconv2=nn.Conv2d(
            in_channels=self.hidden_dimension,
            out_channels=self.hidden_dimension,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.norm2=norm_layer(self.hidden_dimension)

        #third block

        self.dwconv3=nn.Conv2d(
            in_channels=self.hidden_dimension,
            out_channels=self.hidden_dimension,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=self.hidden_dimension  
        )

        self.pwconv3=nn.Conv2d(
            in_channels=self.hidden_dimension,
            out_channels=self.in_channels,
            kernel_size=1,
            stride=1
        )
        self.norm3=norm_layer(self.in_channels)

        self.pwconv4=nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=1,
            stride=1
        )
        



        self.apply(self._init_weights)
    
    def _init_weights(self,m):
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self,x,H,W):
        x=rearrange(x,'b (H W) C -> b C H W',H=H,W=W)   # B N C -> B C H W

        x1= self.dwconv1(x)                 
        x1=self.pwconv1(x1)

        x1=self.norm1(x1)
        x1=self.act(x1)


        x1=self.dwconv2(x1)                 
        x1=self.pwconv2(x1)


        x1=self.norm2(x1)
        x1=self.act(x1)

        x1=self.dwconv3(x1)                 
        x1=self.pwconv3(x1)
        
        x1=self.norm3(x1)
        x1=self.act(x1)

        x1=self.pwconv4(x1)

        x=x1

        x=rearrange(x,'b c H W -> b (H W) c')

        return x


