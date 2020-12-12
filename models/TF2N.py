# -*- coding: utf-8 -*-

#CMU Multimodal SDK, CMU Multimodal Model SDK

#Tensor Fusion Network for Multimodal Sentiment Analysis, Amir Zadeh, Minghai Chen, Soujanya Poria, Erik Cambria, Louis-Philippe Morency - https://arxiv.org/pdf/1707.07250.pdf

#in_modalities: is a list of inputs from each modality - the first dimension of all the modality inputs must be the same, it will be the batch size. The second dimension is the feature dimension. There are a total of n modalities.

#out_dimension: the output of the tensor fusion

import torch
import time
from torch import nn
import torch.nn.functional as F
from six.moves import reduce
from torch.autograd import Variable
import numpy
from layers.complexnn import *
from .SimpleNet import SimpleNet

class MLPSubNet(nn.Module):
    '''
    The subnetwork that is used in TFN for video and audio in the pre-fusion stage
    '''

    def __init__(self, in_size, hidden_size, dropout):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(MLPSubNet, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, sequence_len, in_size)
        '''
        dropped = self.drop(x)
        y_1 = torch.relu(self.linear_1(dropped))
        y_2 = torch.relu(self.linear_2(y_1))
        y_3 = torch.relu(self.linear_3(y_2))

        return y_3


class LSTMSubNet(nn.Module):
    '''
    The LSTM-based subnetwork that is used in TFN for text
    '''

    def __init__(self, in_size, hidden_size, out_size, dropout=0.2, device = torch.device('cpu')):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(LSTMSubNet, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.lstm = nn.LSTMCell(in_size, hidden_size, bias = True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, sequence_len, in_size)
        '''
        batch_size = x.shape[0]
        time_stamps = x.shape[1]
        all_h = []
        all_c = []
        _h = torch.zeros(batch_size, self.hidden_size).to(self.device)
        _c = torch.zeros(batch_size, self.hidden_size).to(self.device)
                    
        for t in range(time_stamps):
            _h, _c = self.lstm(x[:,t,:], (_h,_c))
            all_h.append(_h)
            all_c.append(_c)
            
        h = self.dropout(torch.stack(all_h, dim = 1))
        y_1 = self.linear_1(h)
        return y_1
    
    
class TF2N(nn.Module):
    
    def __init__(self,opt):    
        super(TF2N, self).__init__()
        
        self.input_dims = opt.input_dims
        self.output_dim = opt.output_dim
        self.device = opt.device
        self.num_modalities = len(self.input_dims)   
        if type(opt.hidden_dims) == int:
            self.hidden_dims = [opt.hidden_dims]
        else:
            self.hidden_dims = [int(s) for s in opt.hidden_dims.split(',')]
        self.text_out_dim = opt.text_out_dim
#        self.tensor_size = self.text_out_dim+1 
#        for d in self.hidden_dims[1:]:
#            self.tensor_size = self.tensor_size *(d+1) 
        self.tensor_size = self.text_out_dim
        for d in self.hidden_dims[1:]:
            self.tensor_size = self.tensor_size *d 
            
        self.phase_embed = nn.ModuleList([PhaseEmbedding(opt.lookup_table.shape[0], hidden_dim,freeze = opt.phase_freeze) for hidden_dim in [self.text_out_dim]+self.hidden_dims[1:]])
        
        self.modality_weights = nn.Parameter(torch.zeros(len(self.input_dims)))
        self.modality_specific_weights = nn.Parameter(torch.FloatTensor(self.num_modalities))
        self.complex_multiply = ComplexMultiply()

        
        self.post_fusion_dim = opt.post_fusion_dim
        if type(opt.dropout_probs) == float:
            self.dropout_probs = [opt.dropout_probs]
        else:
            self.dropout_probs = [float(s) for s in opt.dropout_probs.split(',')]
        self.post_fusion_dropout_prob = opt.post_fusion_dropout_prob
        if opt.embedding_enabled:
            embedding_matrix = torch.tensor(opt.lookup_table, dtype=torch.float)
            self.embed = nn.Embedding.from_pretrained(embedding_matrix, freeze=not opt.embedding_trainable)
        # define the pre-fusion subnetworks
        self.text_subnet = LSTMSubNet(self.input_dims[0], self.hidden_dims[0], \
                                      self.text_out_dim, dropout = self.dropout_probs[0],device = self.device)
        self.other_subnets = nn.ModuleList([MLPSubNet(input_dim, hidden_dim, dropout_prob) \
                                            for input_dim, hidden_dim, dropout_prob in \
                                            zip(self.input_dims[1:],self.hidden_dims[1:],self.dropout_probs[1:])])
        self.l2_norm = L2Norm(dim = -1, keep_dims = True)
        self.activation = nn.Softmax(dim = 1)
        self.l2_normalization = L2Normalization(dim = -1)

        
        if self.output_dim == 1:
            self.fc_out = nn.Sequential(nn.Dropout(self.post_fusion_dropout_prob),
                                        nn.Linear(self.tensor_size, self.post_fusion_dim),
                                        nn.ReLU(),
                                        nn.Linear(self.post_fusion_dim, self.post_fusion_dim),
                                        nn.ReLU(),
                                        nn.Linear(self.post_fusion_dim, self.output_dim))
        else:
            self.fc_out = nn.Sequential(nn.Dropout(self.post_fusion_dropout_prob),
                                        nn.Linear(self.tensor_size, self.post_fusion_dim),
                                        nn.ReLU(),
                                        nn.Linear(self.post_fusion_dim, self.post_fusion_dim),
                                        nn.ReLU(),
                                        nn.Linear(self.post_fusion_dim, self.output_dim),
                                        nn.Softmax(dim = 1))
    
    def forward(self, in_modalities):
        
        word_indexes = in_modalities[0]
        in_modalities = [self.embed(modality) if len(modality.shape) == 2 \
                       else modality for modality in in_modalities]
        batch_size=in_modalities[0].shape[0]
        time_stamps = in_modalities[0].shape[1]
        weights = []
        
        hidden_unit = self.text_subnet(in_modalities[0]).to(self.device)
        weights.append(self.l2_norm(hidden_unit))
#        weights.append(self.l2_normalization(hidden_units[0]))
        phases = self.phase_embed[0](word_indexes)
        amplitudes = self.l2_normalization(hidden_unit)
        
        [seq_embedding_real, seq_embedding_imag] = self.complex_multiply([phases, amplitudes])
        
        hidden_units_real = [seq_embedding_real]
        hidden_units_imag = [seq_embedding_imag]
#        hidden_units = [seq_embedding_real]
        
        for i in range(self.num_modalities-1):
            hidden_unit = self.other_subnets[i](in_modalities[i+1]).to(self.device)
            weights.append(self.l2_norm(hidden_unit))
            amplitudes =self.l2_normalization(hidden_unit)
            phases = self.phase_embed[i+1](word_indexes)
            [seq_embedding_real, seq_embedding_imag] = self.complex_multiply([phases, amplitudes])
            hidden_units_real.append(seq_embedding_real)
            hidden_units_imag.append(seq_embedding_imag)

#            tensor_product_real = torch.bmm(seq_embeddings_real.unsqueeze(2),seq_embedding_real.unsqueeze(1))-torch.bmm(seq_embeddings_imag.unsqueeze(2),seq_embedding_imag.unsqueeze(1))
#            tensor_product_imag = torch.bmm(seq_embeddings_real.unsqueeze(2),seq_embedding_imag.unsqueeze(1))+torch.bmm(seq_embeddings_imag.unsqueeze(2),seq_embedding_real.unsqueeze(1))
#            
#            seq_embeddings_real = tensor_product_real.view(batch_size, -1)
#            seq_embeddings_real = tensor_product_imag.view(batch_size, -1)
#            tensor_product_real.view(batch_size,-1)
#             _seq_embeddings_r = torch.einsum('abn,abm->abnm',seq_embeddings_real,seq_embedding_real) \
#            -torch.einsum('abn,abm->abnm',seq_embeddings_imag,seq_embedding_imag)
#            _seq_embeddings_i = torch.einsum('abn,abm->abnm',seq_embeddings_real,seq_embedding_imag) \
#            +torch.einsum('abn,abm->abnm',seq_embeddings_imag,seq_embedding_real)
            
#            hidden_units.append(seq_embedding_real)
            
#        tensor_products = []
                
        modality_weights = nn.Softmax(dim=-1)(self.modality_weights)
        weight = torch.zeros_like(weights[0])
        
        for w, modality_weight in zip(weights, modality_weights):
            weight = weight+modality_weight*w
            
         
        real_tensors = []
        imag_tensors = []
        for i in range(time_stamps):   
            w_i = weight[:,i]

#            tensor_product = hidden_units[0][:,i,:]
            tensor_products_real, tensor_products_imag = hidden_units_real[0][:,i,:], hidden_units_imag[0][:,i,:]
            for h_real, h_imag in zip(hidden_units_real[1:], hidden_units_imag[1:]):
#                h_added = h[:,i,:]
                h_added_real = h_real[:,i,:]
                h_added_imag = h_imag[:,i,:]
                tensor_product_real = torch.bmm(tensor_products_real.unsqueeze(2),h_added_real.unsqueeze(1))-torch.bmm(tensor_products_real.unsqueeze(2),h_added_real.unsqueeze(1))
                
                tensor_product_imag = torch.bmm(tensor_products_real.unsqueeze(2),h_added_imag.unsqueeze(1)) +torch.bmm(tensor_products_imag.unsqueeze(2),h_added_real.unsqueeze(1)) 
               
                tensor_products_real = tensor_product_real.view(batch_size,-1)
                tensor_products_imag = tensor_product_imag.view(batch_size,-1)
                tensor_products_real = tensor_products_real* w_i.expand_as(tensor_products_real)
                tensor_products_imag = tensor_products_imag* w_i.expand_as(tensor_products_real)
#            for j in range(len(tensor_product)):
#                tensor_product[j] =  tensor_product[j] * weight[j][i]
            real_tensors.append(tensor_products_real)
            imag_tensors.append(tensor_products_imag)
#            tensor_products_.append(tensor_product)     
            

#        print(tensor_product.shape)
        
        output = self.fc_out(sum(real_tensors))
#        post_fusion_dropped = self.post_fusion_dropout(tensor_product)
#        post_fusion_y_1 = torch.relu(self.post_fusion_layer_1(post_fusion_dropped))
#        post_fusion_y_2 = torch.relu(self.post_fusion_layer_2(post_fusion_y_1))
#        post_fusion_y_3 = torch.sigmoid(self.post_fusion_layer_3(post_fusion_y_2))
#        output = post_fusion_y_3 * self.output_range + self.output_shift

        return output

if __name__=="__main__":
    print("This is a module and hence cannot be called directly ...")
    print("A toy sample will now run ...")
    
    inputx=Variable(torch.Tensor(numpy.zeros([32,40])),requires_grad=True)
    inputy=Variable(torch.Tensor(numpy.array(numpy.zeros([32,12]))),requires_grad=True)
    inputz=Variable(torch.Tensor(numpy.array(numpy.zeros([32,20]))),requires_grad=True)
    modalities=[inputx,inputy,inputz]
    
    fmodel=TF2N([40,12,20],100)
    
    out=fmodel(modalities)
    
    print("Output")
    print(out[0])
    print("Toy sample finished ...")






