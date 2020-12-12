import torch
from torch import nn
from utils.attention import *
import torch.nn.functional as F
from .SimpleNet import SimpleNet

#Contextual Inter-modal Attention for Multimodal Sentiment Analysis, Ghosal et.al, 2018

class CIMA(nn.Module):
    def __init__(self, opt):
         super(CIMA, self).__init__()
         
         
         self.input_dims = opt.input_dims
         self.orig_d_l, self.orig_d_v, self.orig_d_a = tuple(self.input_dims)
         
         self.drop_gru = opt.drop_gru
         self.drop_rnn = opt.drop_rnn
         self.drop_dense = opt.drop_dense
         
         self.output_dropout_rate = opt.output_dropout_rate
         self.output_cell_dim = opt.output_cell_dim
         self.output_dim =  opt.output_dim
    
         if type(opt.hidden_dims) == int:
             self.hidden_dims = [opt.hidden_dims]
         else:
            self.hidden_dims = [int(s) for s in opt.hidden_dims.split(',')]
            
         if type(opt.output_dims) == int:
             self.output_dims = [opt.output_dims]
         else:
            self.output_dims = [int(s) for s in opt.output_dims.split(',')]   
            
         if opt.embedding_enabled:
            embedding_matrix = torch.tensor(opt.lookup_table, dtype=torch.float)
            self.embed = nn.Embedding.from_pretrained(embedding_matrix, freeze=not opt.embedding_trainable)
            
         self.rnn_l = nn.GRU(self.orig_d_l, self.hidden_dims[0], num_layers=1, batch_first=True, bidirectional=True, dropout=self.drop_gru)
         self.rnn_v = nn.GRU(self.orig_d_v, self.hidden_dims[1], num_layers=1, batch_first=True, bidirectional=True, dropout=self.drop_gru)
         self.rnn_a = nn.GRU(self.orig_d_a, self.hidden_dims[2], num_layers=1, batch_first=True, bidirectional=True, dropout=self.drop_gru)
         
         
         self.fc_l = nn.Linear(2*self.hidden_dims[0], self.output_dims[0])
         self.fc_v = nn.Linear(2*self.hidden_dims[1], self.output_dims[1])
         self.fc_a = nn.Linear(2*self.hidden_dims[2], self.output_dims[2])
         
         
         if self.output_dim == 1:
            self.fc_out = SimpleNet(3*sum(self.output_dims), self.output_cell_dim,
                                    self.output_dropout_rate, self.output_dim)
         else:
            self.fc_out = SimpleNet(3*sum(self.output_dims), self.output_cell_dim,
                                    self.output_dropout_rate, self.output_dim, nn.Softmax(dim = 1))
         
    def forward(self, in_modalities):    
        
        in_modalities = [self.embed(modality) if len(modality.shape) == 2 \
                       else modality for modality in in_modalities]
        
        
        x_l = in_modalities[0]
        x_v = in_modalities[1]
        x_a = in_modalities[2]
       
        #Language
        h_ls = self.rnn_l(x_l)
        if type(h_ls) == tuple:
            h_ls = h_ls[0]
        h_ls = F.dropout(h_ls, p = self.drop_rnn, training = self.training)    
        fc_l = F.dropout(F.relu(self.fc_l(h_ls)), p=self.drop_dense, training=self.training)
        
        #Visual
        h_vs = self.rnn_v(x_v)
        if type(h_vs) == tuple:
            h_vs = h_vs[0]
        h_vs = F.dropout(h_vs, p = self.drop_rnn, training = self.training)
        fc_v = F.dropout(F.relu(self.fc_v(h_vs)), p=self.drop_dense, training=self.training)
        
        #Acoustic
        h_as = self.rnn_a(x_a)
        if type(h_as) == tuple:
            h_as = h_as[0]
        h_as = F.dropout(h_as, p = self.drop_rnn, training = self.training)    
        fc_a = F.dropout(F.relu(self.fc_a(h_as)), p=self.drop_dense, training=self.training)
        
        
        #AV attention 
        atten_av = bi_modal_attention(fc_a,fc_v)
        
        #LA attention 
        atten_la = bi_modal_attention(fc_l,fc_a)
        
        #VT attention 
        atten_vt = bi_modal_attention(fc_v,fc_l)
        
        #Multimodal embedding
        m_embed = torch.cat([atten_av, atten_la, atten_vt,fc_v,fc_a, fc_l], dim=-1)
        m_embed = m_embed.permute(1, 0, 2)
        
        last_h = m_embed[-1]
        output = self.fc_out(last_h)
        
        return output 
