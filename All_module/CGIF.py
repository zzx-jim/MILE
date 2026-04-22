import torch
import torch.nn as nn
import torch.nn.functional as F
from All_module.Gate_Net import *
from All_module.MyselfAttention import *
from All_module.kan import *
from All_module.M_cross_att import *
from All_module.Transformer_Encode import *

class CGIF(nn.Module):
    def __init__(self, input_size, embed_size):
        super(CGIF, self).__init__()
        self.fc = nn.Linear(input_size, embed_size)

        self.fc_sigmoid_t1 = nn.Linear(input_size, embed_size)

        self.norm_t1 = nn.LayerNorm(embed_size)

        self.norm_t2 = nn.LayerNorm(embed_size)



    def forward(self, h_t):
        h_t = self.fc(h_t)

        sig_t = torch.sigmoid(self.fc_sigmoid_t1(h_t))

        h_t_ = (1-sig_t)*h_t

        h_t_ = self.norm_t1(h_t_)
  
        h_t = sig_t*h_t

        h_t = self.norm_t2(h_t)

        return h_t, h_t_
