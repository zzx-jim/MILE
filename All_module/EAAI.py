import torch
import torch.nn as nn
from All_module.Transformer_Encode import *


class IEA(nn.Module):
    def __init__(self, embed_size, heads):
        super(IEA, self).__init__()
        self.W_t = nn.Linear(embed_size, embed_size, bias=False)  # 文本模态权重
        self.W_a = nn.Linear(embed_size, embed_size, bias=False)  # 文本模态权重
        self.W_v = nn.Linear(embed_size, embed_size, bias=False)  # 文本模态权重


        self.cross_att_t_2_t = TransformerEncoderLayer(embed_dim=embed_size, num_heads=heads, ff_dim=embed_size, dropout=0.5)
        self.cross_att_a_2_t = TransformerEncoderLayer(embed_dim=embed_size, num_heads=heads, ff_dim=embed_size, dropout=0.5)
        self.cross_att_v_2_t = TransformerEncoderLayer(embed_dim=embed_size, num_heads=heads, ff_dim=embed_size, dropout=0.5)

        self.fc = nn.Linear(embed_size*3, embed_size) 
        
    def forward(self, h_t, h_a, h_v, umask):
        
        coss_a2t = self.cross_att_a_2_t(h_t, h_a, umask)
        coss_v2t = self.cross_att_v_2_t(h_t, h_v, umask)
        coss_t2t = self.cross_att_t_2_t(h_t, h_t, umask)

        h_a2t = torch.sigmoid(self.W_a(coss_a2t)) * coss_a2t 

        h_v2t = torch.sigmoid(self.W_v(coss_v2t)) * coss_v2t 

        h_t2t = torch.sigmoid(self.W_t(coss_t2t)) * coss_t2t 

        cat_tav = torch.cat((h_t2t, h_v2t, h_a2t), dim=-1)

        out = self.fc(cat_tav)
        
        

        return out, cat_tav
    

class IRA(nn.Module):
    def __init__(self, embed_size, heads):
        super(IRA, self).__init__()      # self.W_tt = nn.Linear(embed_size, embed_size, bias=False)  # 文本模态权重
        self.W_t = nn.Linear(embed_size, embed_size, bias=False)  # 文本模态权重
        self.W_a = nn.Linear(embed_size, embed_size, bias=False)  # 文本模态权重
        self.W_v = nn.Linear(embed_size, embed_size, bias=False)  # 文本模态权重

        self.cross_att_t_2_t = TransformerEncoderLayer(embed_dim=embed_size, num_heads=heads, ff_dim=embed_size, dropout=0.5)
        self.cross_att_a_2_t = TransformerEncoderLayer(embed_dim=embed_size, num_heads=heads, ff_dim=embed_size, dropout=0.5)
        self.cross_att_v_2_t = TransformerEncoderLayer(embed_dim=embed_size, num_heads=heads, ff_dim=embed_size, dropout=0.5)

        self.fc = nn.Linear(embed_size*3, embed_size)  

    def forward(self, h_tt, h_t, h_a, h_v, umask):
        
        coss_a2t = self.cross_att_a_2_t(h_tt, h_a, umask)
        coss_v2t = self.cross_att_v_2_t(h_tt, h_v, umask)
        coss_t2t = self.cross_att_t_2_t(h_tt, h_t, umask)

        h_a2t = torch.sigmoid(self.W_a(coss_a2t)) * coss_a2t 

        h_v2t = torch.sigmoid(self.W_v(coss_v2t)) * coss_v2t 

        h_t2t = torch.sigmoid(self.W_t(coss_t2t)) * coss_t2t 

        cat_tav = torch.cat((h_t2t, h_v2t, h_a2t), dim=-1)

        out = self.fc(cat_tav)

        return out, cat_tav
    
