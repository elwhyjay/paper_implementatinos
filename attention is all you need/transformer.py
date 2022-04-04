import numpy as np
import torch
import torch.nn as nn

from math import (
    pi,
    sqrt,
)

from torch import Tensor
from torch.nn import (
    Module,
    Parameter,
    softmax,
)

__all__ = (

)

class GELU(nn.Module):
    def forward(self,x:Tensor):
        return 0.5*x*(1+torch.tanh(sqrt(2/pi)*(x+0.044715*x**3)))

class LayerNorm(nn.Module):
    def __init__(self,dim:int,eps:float= 1e-6):
        super().__init__()

        #parameters
        self.dim =dim
        self.eps=eps

        #layers
        self.alpha = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self,x:Tensor):
        mean = x.mean(-1,keepdim=True)
        std = x.std(-1,keepdim=True)

        return self.alpha*(x-mean)/(std+self.eps)+self.beta

class Attention(nn.Module):
    def forward(self,
                Q:Tensor,
                K:Tensor,
                V:Tensor,
                mask:Tensor=None,
                dropout:float=0.1):
        #Q: [batch,head,seq_len,dim]
        #K: [batch,head,seq_len,dim]
        #V: [batch,head,seq_len,dim]
        #mask: [batch,1,seq_len,seq_len]
        dim = Q.size(-1)
        #attention
        scores = torch.matmul(Q,K.transpose(-1,-2))/sqrt(dim)

        #mask
        if mask is not None:
            scores = scores.masked_fill(mask==0,float('-inf'))

        scores = softmax(scores,dim=-1)

        #dropout
        if dropout>0:
            scores = dropout(scores,p=dropout)
        
        #attention
        x = torch.matmul(scores,V)

        return x, scores

class MultiHeadAttention(nn.Module):
    def __init__(self,
                 head:int,
                 dim:int,
                 dropout:float=0.1):
        super().__init__()
        assert dim%head==0

        #parameters
        self.head = head
        self.dim = dim
        self.dropout = dropout

        self.dim_k = dim//head

        #layers
        self.W_Q = nn.Linear(dim,dim)
        self.W_K = nn.Linear(dim,dim)
        self.W_V = nn.Linear(dim,dim)
        self.W_M = nn.Linear(dim,dim)
        self.attention = Attention()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,
                Q: Tensor,
                K: Tensor,
                V: Tensor,
                mask: Tensor = None
                ):
        b = Q.size(0)
        Q = self.W_Q(Q).view(b,-1,self.head,self.dim_k).transpose(1,2)
        K = self.W_K(K).view(b,-1,self.head,self.dim_k).transpose(1,2)
        V = self.W_V(V).view(b,-1,self.head,self.dim_k).transpose(1,2)
        x, _ =self.attention(Q,K,V,mask=mask,dropout=self.dropout)


        #concat heads
        # transpose: (b * h * n * d) -> (b * n * h * d)
        # contiguous : reorder
        # view: (b * n * h * d) -> (b * n * d)
        x = x.transpose(1,2).contiguous().view(b,-1,self.dim)

        x = self.W_M(x)

        return x

class SublayerConnection(nn.Module):
    def __init__(self,dim:int= 256,dropout_porb:float= 0.1) :
        super().__init__()

        #parameters
        self.dim = dim
        self.dropout_porb = dropout_porb

        #layers
        self.norm = LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_porb)

    def forward(self,x:Tensor,sublayer:Module):
        residual = self.layer_norm(x)
        residual =sublayer(residual)
        residual = self.dropout(residual)
        return x + residual

class PositionWiseFeedForward(nn.Module):
    def __init__(self,
                dim_model:int = 256,
                dim_ff:int = 2048,
                dropout_prob:float = 0.1):
        super().__init__()
            
        #parameters
        self.dim_model = dim_model
        self.dim_ff = dim_ff
        self.dropout_prob = dropout_prob

        #layers
        self.W_1 = nn.Linear(dim_model,dim_ff)
        self.W_2 = nn.Linear(dim_ff,dim_model)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.gelu = GELU()

    def forward(self,x:Tensor):
        x = self.W_1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.W_2(x)

        return x

class Transformer(nn.Moudule):
    def __init__(self,
                dim_model:int = 256,
                dim_ff:int = 2048,
                head:int = 8,
                num_layer:int = 6,
                dropout_prob:float = 0.1):
        super().__init__()

        #parameters
        self.dim_model = dim_model
        self.dim_ff = dim_ff
        self.head = head
        self.num_layer = num_layer
        self.dropout_prob = dropout_prob

        #layers
        self.attention = MultiHeadAttention(head,dim_model,dropout=dropout_prob)
        self.attention_sublayer = SublayerConnection(dim_model,dropout_prob)
        self.pointwise_feed_forward = PositionWiseFeedForward(dim_model,dim_ff,dropout_prob)
        self.pointwise_feed_forward_sublayer = SublayerConnection(dim_model,dropout_prob)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self,x:Tensor, mask:Tensor=None):
        

        x = self.attention_sublayer(x, lambda f : self.attention(f,f,f,mask = mask))
        x = self.pointwise_feed_forward_sublayer(x,lambda f : self.pointwise_feed_forward(f))
        x = self.dropout(x)

        return x