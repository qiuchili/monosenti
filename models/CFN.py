# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F

from layers.realnn.transformer import TransformerEncoder


'''
Cognitive Multimodal Fusion Network
'''
class CFN(nn.Module):
    def __init__(self, opt):
        """
        Construct a MulT model.
        """
        super(CFN, self).__init__()

        self.input_dims = opt.input_dims
        self.contracted_dim = opt.contracted_dim
        
        self.output_dim = opt.output_dim        

        # self.orig_d_l, self.orig_d_a, self.orig_d_v = hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v
        # self.d_l, self.d_a, self.d_v = 30, 30, 30
        self.orig_d_l, self.orig_d_v = tuple(self.input_dims)
        self.d_l = self.d_v = self.contracted_dim

        self.num_heads = opt.num_heads
        self.layers = opt.layers

        self.attn_dropout_l = opt.attn_dropout_l
        self.attn_dropout_v = opt.attn_dropout_v
        
        self.self_attn_dropout = opt.self_attn_dropout
        self.relu_dropout = opt.relu_dropout
        self.res_dropout = opt.res_dropout
        self.out_dropout = opt.out_dropout
        self.embed_dropout = opt.embed_dropout
        self.attn_mask = opt.attn_mask
        
        # Linear combination weights
        self.weights = nn.Parameter(torch.zeros(2))

        combined_dims = [2* self.contracted_dim+1 for dim in self.input_dims]

        if opt.embedding_enabled:
            embedding_matrix = torch.tensor(opt.lookup_table, dtype=torch.float)
            self.embed = nn.Embedding.from_pretrained(embedding_matrix, freeze=not opt.embedding_trainable)

        
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.softmax = nn.Softmax(dim = -1)

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)
        

        # 2. Crossmodal Attentions
        self.trans_v_with_l = self.get_network(self_type='vl') 
        self.trans_l_with_v = self.get_network(self_type='lv')
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)
       
        # Projection layers
        self.proj1 = nn.ModuleList([nn.Linear(combined_dim, combined_dim) for combined_dim in combined_dims])
        self.proj2 = nn.ModuleList([nn.Linear(combined_dim, combined_dim) for combined_dim in combined_dims])
        self.out_layer = nn.ModuleList([nn.Linear(combined_dim, self.output_dim) for combined_dim in combined_dims])
        
        
    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout_l

        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v

        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2*self.d_l, self.self_attn_dropout

        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2*self.d_v, self.self_attn_dropout
        else:
            raise ValueError("Unknown network type")
        
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)
            
    def get_decisions(self, in_modalities):
        in_modalities = [self.embed(modality) if len(modality.shape) == 2 \
                       else modality for modality in in_modalities]
        x_l = in_modalities[0]
        x_v = in_modalities[1]
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_v = x_v.transpose(1, 2)
       
        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_l = proj_x_l.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
#        print(proj_x_l.shape)

        ####################################################
        #Word level interactions
        h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)    # Dimension (L, N, d_l)
        h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
        
      
        #####################################################
        # Sentence level interactions
        mean_v = torch.mean(proj_x_v,dim = 0, keepdim = False)
        mean_l = torch.mean(proj_x_l,dim = 0, keepdim = False)
        interaction = self.cosine_similarity(mean_v, mean_l).unsqueeze(dim = -1)
        
        #####################################################     
        #  l->v +v , take the last time stamp
        last_hs = torch.cat([h_l_with_vs, proj_x_v], dim=-1)[-1]     
        
        # combine with the sentence level interaction
        last_hs = torch.cat([last_hs, interaction],dim = -1)
        
        # A residual block
        last_hs_proj = self.proj2[0](F.dropout(F.relu(self.proj1[0](last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        output_l_v = self.out_layer[0](last_hs_proj)
        
        # v->l +l , take the last time stamp
        last_hs = torch.cat([h_v_with_ls, proj_x_l], dim=-1)[-1]
        
        # combine with the sentence level interaction
        last_hs = torch.cat([last_hs, interaction],dim = -1)

        # A residual block
        last_hs_proj = self.proj2[1](F.dropout(F.relu(self.proj1[1](last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        output_v_l = self.out_layer[1](last_hs_proj)
        return output_l_v, output_v_l
        
    def get_hard_decision(self, in_modalities):
        output_l_v, output_v_l = self.get_decisions(in_modalities)
        weights = self.softmax(self.weights)
        if weights[0] > weights[1]:
            output = output_l_v
        else:
            output = output_v_l
        return output
        
    def forward(self, in_modalities):
        output_l_v, output_v_l = self.get_decisions(in_modalities)
        weights = self.softmax(self.weights)
        output = weights[0] * output_l_v +weights[1] * output_v_l
    
        return output