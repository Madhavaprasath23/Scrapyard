import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
import torchmetrics
import datetime
from torch import einsum
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from ConvFFN import CONVFFN
from functools import partial
import copy

class MLP(nn.Module):
    def __init__(self,input_size,hidden_layer_size,total_hidden_layer):
        super().__init__()
        self.fc1=nn.Linear(input_size,hidden_layer_size)
        self.module=nn.ModuleList()
        self.total_hidden_layer=total_hidden_layer
        for i in range(total_hidden_layer):
            self.module.append(
                nn.Linear(hidden_layer_size,hidden_layer_size)
            )
    def forward(self,x):
        x=self.fc1(x)
        for linear in self.module:
            x=linear(x)
        return x



class OverlapPatchEmbedding(nn.Module):

    def __init__(self,image_size=224,patch_size=7,stride=4,in_chans=3,embed_dim=768):
        super().__init__()

        self.image_size=(image_size,image_size)
        self.patch_size=(patch_size,patch_size)

        assert max(self.patch_size)>stride, "larger than stride"

        self.image_size=(image_size,image_size)
        self.patch_size=(patch_size,patch_size)

        self.H,self.W=self.image_size[0]//stride,self.image_size[1]//stride

        self.proj=nn.Conv2d(in_chans,embed_dim,kernel_size=self.patch_size,stride=stride,
                            padding=(self.patch_size[0]//2,self.patch_size[1]//2))
                            
        self.norm=nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class Attention(nn.Module):
    def __init__(self, dim,num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
        super().__init__()
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.dim=dim

        self.head_dim= dim// num_heads
        self.qk_scale = qk_scale or self.head_dim ** -0.5

        self.sr_ratio = sr_ratio

        self.linear = linear

        self.q= nn.Linear(dim, dim, bias=qkv_bias)
        self.kv= nn.Linear(dim, dim*2, bias=qkv_bias)
        self.proj= nn.Linear(dim, dim)  
        self.attn_drop= nn.Dropout(attn_drop)
        self.proj_drop= nn.Dropout(proj_drop)
        self.k_learnable=nn.Parameter(torch.randn(dim), requires_grad=True)
        self.v_learnable=nn.Parameter(torch.randn(dim), requires_grad=True)
        if not linear:
            if self.sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.adaptive_pooling=nn.AdaptiveAvgPool2d(7)
            self.norm = nn.LayerNorm(dim)
            self.act=nn.GELU()
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
    
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self,x,H,W):
        B,N,C=x.shape

        qv=self.q(x)
        qv=rearrange(qv,'b n (d c) -> b d n c',d=self.num_heads)

        if not self.linear:
            if self.sr_ratio >1:
                x_=rearrange(x,'B (H W) C -> B C H W',H=H,W=W)
                x_=rearrange(self.sr(x_),'B C H W -> B (H W) C')
                x_=self.norm(x_)
                kv=self.kv(x_)
                kv=rearrange(kv,'B N (d c qk) -> qk B d N c',d=self.num_heads,c=C//self.num_heads)
            else:
                kv=self.kv(x)
                kv=rearrange(kv,'b n (d c qk) -> qk b d n c',d=self.num_heads,c=C//self.num_heads)
            
        else:
            x_=rearrange(x,'b (H W) C -> b C H W', H=H,W=W)
            x_=self.sr(self.adaptive_pooling(x_))
            x_=rearrange(x_,'b c h w -> b (h w) c')
            x_=self.norm(x_)
            x_=self.act(x_)
            kv=rearrange(kv,'b n (d c qk) -> qk b d n c',d=self.num_heads,c=C//self.num_heads)

        k,v=kv[0],kv[1]


        l_k=repeat(self.k_learnable,'h1 -> b h1',b=B) # Batch h1
        

        l_v=repeat(self.v_learnable,'h1  -> b h1 ',b=B) # Batch h1 


        l_v=rearrange(l_v,"b (a d)-> b a d",a=self.num_heads)
        l_k=rearrange(l_k,"b (a d)-> b a d",a=self.num_heads)

        k=rearrange(k,'b head n c -> b n head c')
        v=rearrange(v,'b head n c -> b n head c')

        k=einsum('b n h c, b p q -> b p n q',k,l_k)
        v=einsum('b n h c, b p q -> b p n q',v,l_v)


        attn=einsum('b i j l, b i k m -> b i j k',qv,k)*self.qk_scale

        

        attn=attn.softmax(dim=-1)

        attn=self.attn_drop(attn)


        attn= einsum('b d l n, b d n j-> b d l j',attn,v)
        attn=rearrange(attn,'b d n c -> b n (c d)')
        
        
        x= self.proj(attn)
        x=self.proj_drop(x)

        return x
            
class Block(nn.Module):
    def __init__(self,dim,heads,qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False,mlp_ratio=4.0):
        super().__init__()
        self.dim=dim
        self.heads=heads
        self.qkv_bias=qkv_bias
        self.qk_scale=qk_scale
        self.attn_drop=DropPath(attn_drop) if attn_drop>0 else nn.Identity()
        self.proj_drop=DropPath(proj_drop) if proj_drop>0 else nn.Identity()
        self.sr_ratio=sr_ratio
        self.linear=linear
        

        self.norm1=nn.LayerNorm(dim)

        self.attn=Attention(dim=self.dim,num_heads=self.heads,qkv_bias=self.qkv_bias,qk_scale=self.qk_scale,
                            attn_drop=attn_drop,proj_drop=proj_drop,sr_ratio=self.sr_ratio,linear=self.linear)
        
        self.norm2=nn.LayerNorm(dim)

        self.mlp_dim= int(dim*mlp_ratio)

        self.conv_mlp= CONVFFN(in_channels=self.dim,hidden_dimension=self.mlp_dim)

    
    def forward(self,x,H,W):

        x=x + self.attn_drop(self.attn(self.norm1(x),H,W))
        x=x+self.proj_drop(self.conv_mlp(self.norm2(x),H,W))
        return x


class LearnablePatchAttentionModel(nn.Module):

    def __init__(self,image_size=224,patch_size=7,embeded_dimesion=[64,128,256,512],depth=[2,2,3,3],
                 num_heads=[2,4,4,8], sr_ratio=[8,4,2,1],norm_layer=nn.LayerNorm,qkv_bias=True,qk_scale=None,attn_drop=0.1,proj_drop=0.1,
                 stages=4,mlp_ratio=[6.0,6.0,4.0,4.0],linear=False):
        
        super().__init__()

        self.image_size=image_size
        self.patch_size=patch_size
        self.embeded_dimesion=embeded_dimesion
        self.depth=depth
        self.num_heads=num_heads
        self.norm_layer=norm_layer
        self.qkv_bias=qkv_bias
        self.qk_scale=qk_scale
        self.attn_drop=attn_drop
        self.proj_drop=proj_drop

        self.stages=stages
        self.mlp_ratio=mlp_ratio

        dpr=[x.item() for x in torch.linspace(0,self.proj_drop,sum(self.depth))]
        cur=0
        
        for i in range(self.stages):

            

            patch_embedding=OverlapPatchEmbedding(
                image_size=self.image_size if i==0 else self.image_size//(2**(i + 1)),
                patch_size= self.patch_size if i==0 else 3,
                stride= 4 if i==0 else 2,
                in_chans=3 if i==0 else self.embeded_dimesion[i-1],
                embed_dim=self.embeded_dimesion[i]
            )

            if i==0:
                #learnable_weights=nn.Parameter(
                 #   torch.randn(2,3196,64),
                  #  requires_grad=True
                #) verision 2
                pass
            attn=nn.ModuleList(
                [
                    Block(
                        dim=embeded_dimesion[i],
                        qkv_bias=True,
                        qk_scale=None,
                        attn_drop=attn_drop,
                        proj_drop=dpr[cur+j],
                        sr_ratio=sr_ratio[i],
                        linear=linear,
                        mlp_ratio=self.mlp_ratio[i],
                        heads=self.num_heads[i]
                        
                    ) for j in range(self.depth[i])
                ]
            )

            norm=norm_layer(embeded_dimesion[i])

            projection_layer=nn.Linear(embeded_dimesion[i],embeded_dimesion[i])

            norm2=norm_layer(embeded_dimesion[i])


            cur+=self.depth[i]

            setattr(self,f'patch_embedding_{i +1}',patch_embedding)
            """if i==0:
                setattr(self,f'learnable_weights_{i +1}',learnable_weights) version 2"""

            setattr(self,f'block_{i +1}',attn)
            setattr(self,f'norm_{i +1}',norm)
            setattr(self,f'projection_layer_{i +1}',projection_layer)
            setattr(self,f'norm2_{i +1}',norm2)

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    

    def forward_features(self,x):
        B=x.shape[0]
        c=0

        for i in range(self.stages):
            patch_embedding=getattr(self,f'patch_embedding_{i +1}')
            Block=getattr(self,f'block_{i +1}')
            norm=getattr(self,f'norm_{i +1}')
            projection_layer=getattr(self,f'projection_layer_{i +1}')
            norm2=getattr(self,f'norm2_{i +1}')

            x,H,W=patch_embedding(x)
            for blk in Block:
                x=blk(x,H,W)

            
            x=norm(x)
            x=projection_layer(x)
            x=norm2(x)
            c+=1

            if i!=self.stages-1:
                x=x.reshape(B,H,W,-1).permute(0,3,1,2).contiguous()
            

        return x.mean(dim=1)
    
    def forward(self,x):
        return self.forward_features(x)


def create_model(type_model="Large",frozen_model="Total",pre_training=False,num_classes=0):
    if type_model=="Small":
        model=create_model_small(type_model,frozen_model)
    elif type_model=="Medium":
        model=create_model_medium(type_model,frozen_model)
    elif type_model=="Large":
        model=create_model_large(type_model,frozen_model)
    else:
        raise Exception(f"No Model name such as {type_model} exist choose from Small Medium Large")
    
    hidden_layer=512
    mlp=MLP(model.embeded_dimesion,hidden_layer_size=hidden_layer,total_hidden_layer=1) if not pre_training else MLP(model.embeded_dimesion,hidden_layer_size=num_classes,total_hidden_layer=1)
    return nn.Sequential(
        model,
        mlp
    )
    


def create_model_small(frozen_model="Total"):
    model=LearnablePatchAttentionModel(
        embeded_dimesion=[32,64,160,256],
        num_heads=[1,2,5,8],
        mlp_ratio=[8,8,4,4],
        norm_layer=partial(nn.LayerNorm,eps=1e-6),
        depth=[2,2,2,2],
        sr_ratio=[8,4,2,1]
    )
    if frozen_model=="Total":
        orginal_dictionary= torch.load("pvt_v2_b2.pth")
        new_state_dict=copy.deepcopy(model.state_dict())
        for i in model.state_dict().keys():
            if i in orginal_dictionary.keys():
                new_state_dict[i]=orginal_dictionary[i]
        
        model.load_state_dict(new_state_dict)
        
        for name,parameters in model.named_parameters():
            if name in orginal_dictionary.keys():
                parameters.requires_grad=False
        
        return model
    
    elif frozen_model=="Partial":
        orginal_dictionary= torch.load("pvt_v2_b2.pth")
        new_state_dict=copy.deepcopy(model.state_dict())
        for i in model.state_dict().keys():
            if i in orginal_dictionary.keys():
                new_state_dict[i]=orginal_dictionary[i]
        model.load_state_dict(new_state_dict)
        return model
    
    elif frozen_model=="None":
        
        return model
    
    else:
        raise Exception("Invalid Mode of freezing the model")


def create_model_medium(frozen_model="Total"):
    model=LearnablePatchAttentionModel(
        embeded_dimesion=[64,128,320,512],
        num_heads=[1,2,5,8],
        mlp_ratio=[8,8,4,4],norm_layer=partial(nn.LayerNorm,eps=1e-6),
        depth=[3,4,6,3],sr_ratio=[8,4,2,1]
    )


    if frozen_model=="Total":
        orginal_dictionary= torch.load("pvt_v2_b2.pth")
        new_state_dict=copy.deepcopy(model.state_dict())
        for i in model.state_dict().keys():
            if i in orginal_dictionary.keys():
                new_state_dict[i]=orginal_dictionary[i]
        
        model.load_state_dict(new_state_dict)
        
        for name,parameters in model.named_parameters():
            if name in orginal_dictionary.keys():
                parameters.requires_grad=False
        
        return model
    
    elif frozen_model=="Partial":
        orginal_dictionary= torch.load("pvt_v2_b2.pth")
        new_state_dict=copy.deepcopy(model.state_dict())
        for i in model.state_dict().keys():
            if i in orginal_dictionary.keys():
                new_state_dict[i]=orginal_dictionary[i]
        model.load_state_dict(new_state_dict)
        return model
    
    elif frozen_model=="None":
        
        return model
    else:
        raise Exception("Invalid Mode of freezing the model")

def create_model_large(frozen_model="Total"):
    model=LearnablePatchAttentionModel(
    num_heads=[1,2,5,8],
    embeded_dimesion=[64,128,320,512],
    depth=[3,6,40,3],
    patch_size=7,
    norm_layer=partial(nn.LayerNorm,eps=1e-6),
    mlp_ratio=[8, 8, 4, 4],
    sr_ratio=[8,4,2,1],linear=False
    )
    
    if frozen_model=="Total":
        orginal_dictionary= torch.load("pvt_v2_b2.pth")
        new_state_dict=copy.deepcopy(model.state_dict())
        for i in model.state_dict().keys():
            if i in orginal_dictionary.keys():
                new_state_dict[i]=orginal_dictionary[i]
        
        model.load_state_dict(new_state_dict)
        
        for name,parameters in model.named_parameters():
            if name in orginal_dictionary.keys():
                parameters.requires_grad=False
        
        return model
    
    elif frozen_model=="Partial":
        orginal_dictionary= torch.load("pvt_v2_b2.pth")
        new_state_dict=copy.deepcopy(model.state_dict())
        for i in model.state_dict().keys():
            if i in orginal_dictionary.keys():
                new_state_dict[i]=orginal_dictionary[i]
        model.load_state_dict(new_state_dict)
        return model
    
    elif frozen_model=="None":
        
        return model
    else:
        Exception("Invalid Mode of freezing the model")

def create_model_large_linear(frozen_model="Total"):
    model=LearnablePatchAttentionModel(
    num_heads=[1,2,5,8],
    embeded_dimesion=[64,128,320,512],
    depth=[3,6,40,3],
    patch_size=7,
    norm_layer=partial(nn.LayerNorm,eps=1e-6),
    mlp_ratio=[8, 8, 4, 4],
    sr_ratio=[8,4,2,1],linear=True
    )
    
    if frozen_model=="Total":
        orginal_dictionary= torch.load("pvt_v2_b2_lin.pth")
        new_state_dict=copy.deepcopy(model.state_dict())
        for i in model.state_dict().keys():
            if i in orginal_dictionary.keys():
                new_state_dict[i]=orginal_dictionary[i]
        
        model.load_state_dict(new_state_dict)
        
        for name,parameters in model.named_parameters():
            if name in orginal_dictionary.keys():
                parameters.requires_grad=False
        
        return model
    
    elif frozen_model=="Partial":
        orginal_dictionary= torch.load("pvt_v2_b2.pth")
        new_state_dict=copy.deepcopy(model.state_dict())
        for i in model.state_dict().keys():
            if i in orginal_dictionary.keys():
                new_state_dict[i]=orginal_dictionary[i]
        model.load_state_dict(new_state_dict)
        return model
    
    elif frozen_model=="None":
        
        return model
    else:
        Exception("Invalid Mode of freezing the model")
