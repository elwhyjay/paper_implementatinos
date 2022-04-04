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

class Transformer(nn.Moudule):
    def __init__(self,
                 head:int,
                 dim:int,
                 layer:int,
                 dropout:float=0.1):
        super().__init__()

        #parameters
        self.head = head
        self.dim = dim
        self.layer = layer
        self.dropout = dropout

        self.dim_K = dim//head

        #layers
        self.W_Q = nn.Linear(dim,dim)
        self.W_K = nn.Linear(dim,dim)
        self.W_V = nn.Linear(dim,dim)
        self.W_M = nn.Linear(dim,dim)
        self.attention = Attention()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,
                Q:Tensor,
                K:Tensor,
                V:Tensor,
                maaks:Tensor = None):
        
        Q
