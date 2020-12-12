# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from layers.complexnn import *
from .SimpleNet import SimpleNet
import pickle
import os
import numpy as np

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
#        x = torch.mean(x, dim = 1,keepdim = False)
#        normed = self.norm(x)
#        dropped = self.drop(normed)
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
    
class LocalMixtureNN(torch.nn.Module):
    def __init__(self, opt):
        """
        max_sequence_len: input sentence length
        embedding_dim: input dimension
        num_measurements: number of measurement units, also the output dimension

        """
        super(LocalMixtureNN, self).__init__()
        opt.sentiment_dic = None
        self.max_sequence_len = opt.max_seq_len
        self.input_dims = opt.input_dims
        self.text_hidden_dim = opt.text_hidden_dim
        self.feature_indexes = opt.feature_indexes

        
        self.device = opt.device     
        self.phase_dropout_rate = opt.phase_dropout_rate
        if type(opt.subnet_dropout_rates) == float:
            self.subnet_dropout_rates = [opt.subnet_dropout_rates]
        else:
            self.subnet_dropout_rates =  [float(s) for s in opt.subnet_dropout_rates.split(',')]
        self.num_measurements = opt.measurement_size
        self.output_cell_dim = opt.output_cell_dim
        self.output_dropout_rate = opt.output_dropout_rate
        self.output_dim = opt.output_dim
        if type(opt.contracted_dims) == int:
            self.contracted_dims = [opt.contracted_dims]
        else:
            self.contracted_dims =  [int(s) for s in opt.contracted_dims.split(',')]
        
        self.modality_weights = nn.Parameter(torch.zeros(len(self.input_dims)))
        
        # Only textual input, QDNN performed for word pivots
        # Amplitude and Phase embedding
        self.num_modalities = len(self.input_dims)
        self.modality_specific_weights = nn.Parameter(torch.FloatTensor(self.num_modalities))
        
        self.embedding_matrix = torch.tensor(opt.lookup_table, dtype=torch.float) 
        self.embed = nn.Embedding.from_pretrained(self.embedding_matrix, freeze=opt.amplitude_freeze)
        
         # SubEmbedding Networks
        self.proj_layers = nn.ModuleList([])
        for i,ind in enumerate(self.feature_indexes):
            if ind == 0:
                self.proj_layers.append(LSTMSubNet(self.input_dims[i],self.text_hidden_dim,self.contracted_dims[i],dropout = self.subnet_dropout_rates[i], device=self.device))
            else:
                self.proj_layers.append(MLPSubNet(self.input_dims[i],self.contracted_dims[i], self.subnet_dropout_rates[i]))
                
        # Phase Embedding
        #all_feature_names = ['textual','visual','acoustic']

        self.phase_embed = nn.ModuleList()

        for i,ind in enumerate(self.feature_indexes):
            pre_trained_phases = None
#            if 'pretrained_phases_dir' in opt.__dict__:
#                pkl_path = os.path.join(opt.pretrained_phases_dir,all_feature_names[i]+'_phases.pkl')
#                if os.path.exists(pkl_path):
#                    pre_trained_phases = pickle.load(open(pkl_path,'rb'))
            if ind==0:
                self.phase_embed.append(PhaseEmbedding(opt.lookup_table.shape[0], self.contracted_dims[i], embedding_matrix = pre_trained_phases, sentiment_dic = opt.sentiment_dic, freeze = opt.phase_freeze))
            else:
                self.phase_embed.append(PhaseEmbedding(opt.lookup_table.shape[0], self.contracted_dims[i], embedding_matrix = pre_trained_phases, freeze = False))

        self.l2_norm = L2Norm(dim = -1, keep_dims = True)
        self.l2_normalization = L2Normalization(dim = -1)
        self.activation = nn.Softmax(dim = 1)
        self.complex_multiply = ComplexMultiply()
        self.mixture = ComplexMixture(device = self.device)
        
        self.ngram = nn.ModuleList([NGram(gram_n = int(n_value),device = self.device) for n_value in opt.ngram_value.split(',')])
        self.num_measurements = opt.measurement_size
        self.sentiment_dic = opt.sentiment_dic

        
        self.measurement_dim = 1
        for dim in self.contracted_dims:
            self.measurement_dim = self.measurement_dim*dim
            
        self.measurement = ComplexMeasurement2(self.measurement_dim, units = self.num_measurements,device = self.device)   
#        self.hidden_units = opt.hidden_units
        
        self.pooling_type = opt.pooling_type
        
        self.feature_num = 0 
        for one_type in self.pooling_type.split(','):
            one_type = one_type.strip()
            if one_type == 'max':
                # max out the sequence dimension
                feature_num = self.num_measurements
            elif one_type == 'average':
                # average out the sequence dimension
                feature_num = self.num_measurements
            elif one_type == 'none':
                # do nothing at all, flatten
                feature_num = self.max_sequence_len*len(self.ngram)*self.num_measurements
            elif one_type == 'max_col':
                # max out the measurement dimension
                feature_num = self.max_sequence_len*len(self.ngram)
            elif one_type == 'average_col':
                # average out the measurement dimension
                feature_num = self.max_sequence_len*len(self.ngram)
            else:
                print('Wrong input pooling type -- The default flatten layer is used.')
                feature_num = self.max_sequence_len*self.num_measurements
            self.feature_num = self.feature_num + feature_num
            
        
        self.fc_out = nn.Sequential(nn.Dropout(self.output_dropout_rate),
                                    nn.Linear(self.feature_num, self.output_cell_dim),
                                    nn.ReLU(),
                                    nn.Linear(self.output_cell_dim, self.output_cell_dim),
                                    nn.ReLU(),
                                    nn.Linear(self.output_cell_dim, self.output_dim))

    def forward(self, in_modalities):
        """
        In the forward function we accept a Variable of input data and we must 
        return a Variable of output data. We can use Modules defined in the 
        constructor as well as arbitrary operators on Variables.
        """
        batch_size = in_modalities[0].shape[0]
        seq_len = in_modalities[0].shape[1]
        
        word_indexes = in_modalities[0]
        weights = []
        
        
        # The textual modality is not used when building the model
        if len(in_modalities) == self.num_modalities+1:
            in_modalities = in_modalities[1:]
        else:
            in_modalities = [self.embed(modality) if len(modality.shape) == 2 \
                       else modality for modality in in_modalities]
            
        hidden_units_real = []
        hidden_units_imag = []
                

        for i,ind in enumerate(self.feature_indexes):     
            phases = self.phase_embed[i](word_indexes)
                
            amplitudes = self.proj_layers[i](in_modalities[i])
            norms = self.l2_norm(amplitudes)
            amplitudes = self.l2_normalization(amplitudes)
            weights.append(norms)
            [seq_embedding_real, seq_embedding_imag] = self.complex_multiply([phases, amplitudes])
            hidden_units_real.append(seq_embedding_real)
            hidden_units_imag.append(seq_embedding_imag)
            
        modality_weights = nn.Softmax(dim=-1)(self.modality_weights)
        weight = torch.zeros_like(weights[0])
        
        for w, modality_weight in zip(weights, modality_weights):
            weight = weight+modality_weight*w

        real_tensors = []
        imag_tensors = [] 
        for i in range(seq_len):   
            tensor_product_real = torch.ones(batch_size,1).to(self.device)
            tensor_product_imag = torch.ones(batch_size,1).to(self.device)
            
            for h_real, h_imag in zip(hidden_units_real, hidden_units_imag):
               
                h_added_real = h_real[:,i,:]
                h_added_imag = h_imag[:,i,:]
                result_real = torch.bmm(tensor_product_real.unsqueeze(2),h_added_real.unsqueeze(1))-torch.bmm(tensor_product_imag.unsqueeze(2),h_added_imag.unsqueeze(1))
                result_imag = torch.bmm(tensor_product_real.unsqueeze(2),h_added_imag.unsqueeze(1)) +torch.bmm(tensor_product_imag.unsqueeze(2),h_added_real.unsqueeze(1)) 
                
                tensor_product_real = result_real.view(batch_size,-1)
                tensor_product_imag = result_imag.view(batch_size,-1)
                
            real_tensors.append(tensor_product_real)
            imag_tensors.append(tensor_product_imag)
            
        real_tensors = torch.stack(real_tensors,dim = 1)
        imag_tensors = torch.stack(imag_tensors,dim = 1)
        prob_list = []

        for n_gram in self.ngram:
            n_gram_embedding_real = n_gram(real_tensors)
            n_gram_embedding_imag = n_gram(imag_tensors)
            n_gram_weight = n_gram(weight)
            n_gram_weight = self.activation(n_gram_weight)
    
            prob_list.append(self.measurement([n_gram_embedding_real, n_gram_embedding_imag, n_gram_weight]))
            
        probs_tensor = torch.cat(prob_list,dim = 1)
        probs_feature = []
        for one_type in self.pooling_type.split(','):
            one_type = one_type.strip()
            if one_type == 'max':
#                probs = GlobalMaxPooling1D()(self.probs)
                # max out the sequence dimension
                probs,_ = torch.max(probs_tensor,1,False)
                
            elif one_type == 'average':
                # average out the sequence dimension
                probs = torch.mean(probs_tensor,1,False)
                
            elif one_type == 'none':
                # do nothing at all, flatten
                probs = torch.flatten(probs_tensor, start_dim=1, end_dim=2)
                
            elif one_type == 'max_col':
                # max out the measurement dimension
                probs,_ = torch.max(probs_tensor,2,False)
                
            elif one_type == 'average_col':
                # average out the measurement dimension
                probs = torch.mean(probs_tensor,2,False)
            else:
                print('Wrong input pooling type -- The default flatten layer is used.')
                probs = torch.flatten(probs_tensor, start_dim=1, end_dim=2)
            probs_feature.append(probs)
        
        probs = torch.cat(probs_feature, dim = -1)
        output = self.fc_out(probs)
        
        return output
    
    def get_phases(self):
        all_feature_names = ['textual','visual','acoustic']
        phases_dict = {}
        for i in range(self.num_modalities):
            modality = all_feature_names[self.feature_indexes[i]]
            phase_lookup_table = self.phase_embed[i].weight.detach().cpu().numpy()
            phases_dict[modality] = phase_lookup_table
            
        return phases_dict
    
    def get_sub_measurements(self, feature_indexes):
        measurements = self.measurement.kernel.detach().cpu().numpy()
#        for dim in self.contracted_dims
        dim_mul = 1
        for ind in feature_indexes:
            dim_id = self.feature_indexes.index(ind)
            dim_mul = dim_mul * self.contracted_dims[dim_id]
            
        reduced_dm_list = []
        for i in range(self.num_measurements):
            measurement = measurements[i,:,0]+1j*measurements[i,:,1]
            reshaped_mea = measurement.reshape(*self.contracted_dims)
            for j, _feature_id in enumerate(feature_indexes):
                dim_id = self.feature_indexes.index(_feature_id)
                reshaped_mea = np.moveaxis(reshaped_mea,dim_id,j)
            reshaped_mea = reshaped_mea.flatten()

            mea_operator = np.outer(reshaped_mea, reshaped_mea.transpose().conjugate())
            reduced_dm = np.trace(mea_operator.reshape(dim_mul,int(self.measurement_dim/dim_mul), 
                                                       dim_mul,int(self.measurement_dim/dim_mul)),
                                                       axis1= 1,axis2 = 3)
                    
            reduced_dm_list.append(reduced_dm)
        return reduced_dm_list
    
    def get_submodality_results(self, in_modalities, feature_indexes):
        dim_id_list = [self.feature_indexes.index(ind) for ind in feature_indexes]

        batch_size = in_modalities[0].shape[0]
        seq_len = in_modalities[0].shape[1]
        
        word_indexes = in_modalities[0]
        weights = []
        
        in_modalities = [in_modalities[_id] for _id in dim_id_list]
        in_modalities = [self.embed(modality) if len(modality.shape) == 2 \
                       else modality for modality in in_modalities]
        
        hidden_units_real = []
        hidden_units_imag = []    
        for i,ind in enumerate(dim_id_list):     
            #modality_id =self.feature_indexes[i]
            phases = self.phase_embed[ind](word_indexes)        
            amplitudes = self.proj_layers[ind](in_modalities[i])
            norms = self.l2_norm(amplitudes)
            amplitudes = self.l2_normalization(amplitudes)
            weight = self.activation(norms)
            weights.append(weight)
            [seq_embedding_real, seq_embedding_imag] = self.complex_multiply([phases, amplitudes])
            hidden_units_real.append(seq_embedding_real)
            hidden_units_imag.append(seq_embedding_imag)
            
        modality_weights = nn.Softmax(dim=-1)(self.modality_weights[dim_id_list])
        weight = torch.zeros_like(weights[0])
        
        for w, modality_weight in zip(weights, modality_weights):
            weight = weight+modality_weight*w
            
        real_tensors = []
        imag_tensors = [] 
        
        for i in range(seq_len):   
            tensor_product_real = torch.ones(batch_size,1).to(self.device)
            tensor_product_imag = torch.ones(batch_size,1).to(self.device)
            
            for h_real, h_imag in zip(hidden_units_real, hidden_units_imag):
               
                h_added_real = h_real[:,i,:]
                h_added_imag = h_imag[:,i,:]
                result_real = torch.bmm(tensor_product_real.unsqueeze(2),h_added_real.unsqueeze(1))-torch.bmm(tensor_product_imag.unsqueeze(2),h_added_imag.unsqueeze(1))
                
                result_imag = torch.bmm(tensor_product_real.unsqueeze(2),h_added_imag.unsqueeze(1)) +torch.bmm(tensor_product_imag.unsqueeze(2),h_added_real.unsqueeze(1)) 
                
                tensor_product_real = result_real.view(batch_size,-1)
                tensor_product_imag = result_imag.view(batch_size,-1)
                
            real_tensors.append(tensor_product_real)
            imag_tensors.append(tensor_product_imag)
            
        real_tensors = torch.stack(real_tensors,dim = 1)
        imag_tensors = torch.stack(imag_tensors,dim = 1)
        projector_list = self.get_sub_measurements(feature_indexes)
        
        prob_list = []
        for n_gram in self.ngram:
            n_gram_embedding_real = n_gram(real_tensors).detach().cpu().numpy()
            n_gram_embedding_imag = n_gram(imag_tensors).detach().cpu().numpy()
            n_gram_weight = n_gram(weight)
            n_gram_weight = self.activation(n_gram_weight).detach().cpu().numpy()
            
            n_gram_embedding = n_gram_embedding_real+1j*n_gram_embedding_imag
            probs_array = np.zeros((batch_size,seq_len,self.num_measurements))
            for b_id in range(batch_size):
                for l in range(seq_len):
#                    dm = np.zeros(self.measurement_dim, dtype = 'complex64')
                    probs = []
                    for proj in projector_list:
                        prob = 0
                        for w_id in range(n_gram.gram_n): 
                            vec = n_gram_embedding[b_id,w_id,l,:]
                            _weight = n_gram_weight[b_id,w_id,l,0]   
                            prob = prob+ np.matmul(np.matmul(vec, proj),vec.transpose().conjugate())*_weight
                        probs.append(prob.real)
                    probs_array[b_id,l,:] = np.asarray(probs)
                
            prob_list.append(torch.tensor(probs_array, dtype = torch.float).to(self.device))
        probs_tensor = torch.cat(prob_list,dim = 1)
        probs_feature = []
        for one_type in self.pooling_type.split(','):
            one_type = one_type.strip()
            if one_type == 'max':
#                probs = GlobalMaxPooling1D()(self.probs)
                # max out the sequence dimension
                probs,_ = torch.max(probs_tensor,1,False)
                
            elif one_type == 'average':
                # average out the sequence dimension
                probs = torch.mean(probs_tensor,1,False)
                
            elif one_type == 'none':
                # do nothing at all, flatten
                probs = torch.flatten(probs_tensor, start_dim=1, end_dim=2)
                
            elif one_type == 'max_col':
                # max out the measurement dimension
                probs,_ = torch.max(probs_tensor,2,False)
                
            elif one_type == 'average_col':
                # average out the measurement dimension
                probs = torch.mean(probs_tensor,2,False)
            else:
                print('Wrong input pooling type -- The default flatten layer is used.')
                probs = torch.flatten(probs_tensor, start_dim=1, end_dim=2)
            probs_feature.append(probs)
        
        probs = torch.cat(probs_feature, dim = -1)
        output = self.fc_out(probs)
        return(output)