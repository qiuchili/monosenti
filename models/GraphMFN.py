# -*- coding: utf-8 -*-

#CMU Multimodal SDK, CMU Multimodal Model SDK

#Multimodal Language Analysis in the Wild: CMU-MOSEI Dataset and Interpretable Dynamic Fusion Graph, Amir Zadeh, Paul Pu Liang, Jonathan Vanbriesen, Soujanya Poria, Edmund Tong, Erik Cambria, Minghai Chen, Louis-Philippe Morency - http://www.aclweb.org/anthology/P18-1208

#pattern_model: a nn.Sequential model which will be used as core of the models inside the DFG

#in_dimensions: input dimensions of each modality

#out_dimension: output dimension of the pattern models

#efficacy_model: the core of the efficacy model

#in_modalities: inputs from each modality, the same order as in_dimensions

import torch
from torch import nn
import torch.nn.functional as F
import copy
from six.moves import reduce
from itertools import chain,combinations
from collections import OrderedDict
from .LSTHM import LSTHMCell
from .SimpleNet import SimpleNet

class GraphMFN(nn.Module):
    
    def __init__(self,opt):
        super(GraphMFN,self).__init__()
        self.input_dims = opt.input_dims
        
        if type(opt.hidden_dims) == int:
            self.hidden_dims = [opt.hidden_dims]
        else:
            self.hidden_dims = [int(s) for s in opt.hidden_dims.split(',')]
        self.output_dim = opt.output_dim
        self.device = opt.device
        self.compressed_dim = opt.compressed_dim
        self.num_modalities = len(self.input_dims)
        if opt.embedding_enabled:
            embedding_matrix = torch.tensor(opt.lookup_table, dtype=torch.float)
            self.embed = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        
        if type(opt.dfg_input_dims) == int:
            self.dfg_input_dims = [opt.dfg_input_dims]
        else:
            self.dfg_input_dims = [int(s) for s in opt.dfg_input_dims.split(',')]
        
            
        self.dfg_out_dim = opt.dfg_out_dim
        
        if type(opt.dfg_init_cell_dims) == int:
            self.dfg_init_cell_dims = [opt.dfg_init_cell_dims]
        else:
            self.dfg_init_cell_dims = [int(s) for s in opt.dfg_init_cell_dims.split(',')]
        
        if type(opt.dfg_init_dropout_rates) == float:
            self.dfg_init_dropout_rates = [opt.dfg_init_dropout_rates]
        else:
            self.dfg_init_dropout_rates = [float(s) for s in opt.dfg_init_dropout_rates.split(',')]
        
        self.pattern_cell_dim = opt.pattern_cell_dim
        self.pattern_dropout_rate = opt.pattern_dropout_rate
        
        self.efficacy_cell_dim = opt.pattern_cell_dim
        self.efficacy_dropout_rate = opt.pattern_dropout_rate
        
        self.memory_dim = opt.memory_dim
        self.memory_cell_dim = opt.memory_cell_dim
        self.memory_dropout_rate = opt.memory_dropout_rate
        
        if type(opt.gamma_cell_dims) == int:
            self.gamma_cell_dims = [opt.gamma_cell_dims]
        else:
            self.gamma_cell_dims = [int(s) for s in opt.gamma_cell_dims.split(',')]
        
        if type(opt.gamma_dropout_rates) == float:
            self.gamma_dropout_rates = [opt.gamma_dropout_rates]
        else:
            self.gamma_dropout_rates = [float(s) for s in opt.gamma_dropout_rates.split(',')]
        
        self.feedback_cell_dim = opt.feedback_cell_dim
        self.feedback_dropout_rate = opt.feedback_dropout_rate
        
        self.output_cell_dim = opt.output_cell_dim
        self.output_dropout_rate = opt.output_dropout_rate
        
        
        self.lsthms = nn.ModuleList(LSTHMCell(hidden_dim, input_dim,self.compressed_dim, \
                                              device = self.device) for hidden_dim, input_dim \
                                                in zip(self.hidden_dims, self.input_dims))
        
        # Initialization of the LSTM units  
        
        self.dfg_init_networks = nn.ModuleList([SimpleNet(2*hidden_dim, cell_dim,
                                                          dropout, dfg_input_dim,
                                                          nn.Sigmoid()) 
                                        for hidden_dim, dfg_input_dim, cell_dim, 
                                        dropout in zip(self.hidden_dims, self.dfg_input_dims, 
                                                       self.dfg_init_cell_dims,self.dfg_init_dropout_rates)])
                
                
#                nn.Linear(2*hidden_dim,dfg_input_dim).to(self.device) for hidden_dim,dfg_input_dim in zip(self.hidden_dims, self.dfg_input_dims)])
        self.efficacy_dim = opt.efficacy_dim

        pattern_model = SimpleNet(self.efficacy_dim, self.pattern_cell_dim, self.pattern_dropout_rate, self.dfg_out_dim)
          
        efficacy_model = SimpleNet(self.efficacy_dim, self.efficacy_cell_dim, self.efficacy_dropout_rate, self.dfg_out_dim)

        
        self.dfg = DynamicFusionGraph(pattern_model, self.dfg_input_dims, self.dfg_out_dim, efficacy_model,self.device)


        # Retain and Update Gate
        self.fc_gamma =  nn.ModuleList([SimpleNet(self.dfg_out_dim, cell_dim, dropout, self.memory_dim, nn.Sigmoid()) 
                                        for cell_dim, dropout in zip(self.gamma_cell_dims,self.gamma_dropout_rates)])
        
        self.fc_memory = SimpleNet(self.dfg_out_dim,self.memory_cell_dim, self.memory_dropout_rate, self.memory_dim, nn.Tanh())
    
        
        self.feedback = nn.Sequential(nn.Linear(self.dfg_out_dim, self.feedback_cell_dim),
                                       nn.ReLU(),
                                       nn.Dropout(self.feedback_dropout_rate),
                                       nn.Linear(self.feedback_cell_dim, self.compressed_dim),
                                       nn.Sigmoid(),
                                       nn.Softmax(dim = 1)
                                       )
        
        if self.output_dim == 1:
            self.fc_out = SimpleNet(self.memory_dim + sum(self.hidden_dims), 
                                    self.output_cell_dim, self.output_dropout_rate, self.output_dim)
        else:
            self.fc_out = SimpleNet(self.memory_dim + sum(self.hidden_dims), 
                                    self.output_cell_dim, self.output_dropout_rate, self.output_dim, nn.Softmax(dim = 1))

        
    def forward(self,in_modalities):
        
        in_modalities = [self.embed(modality) if len(modality.shape) == 2 \
                       else modality for modality in in_modalities]
        self.batch_size = in_modalities[0].shape[0]
        time_stamps = in_modalities[0].shape[1]
        z = torch.zeros(self.batch_size,self.compressed_dim).to(self.device)
        h = [torch.zeros(self.batch_size,hidden_dim).to(self.device) for hidden_dim in self.hidden_dims]
        c = [torch.zeros(self.batch_size,hidden_dim).to(self.device) for hidden_dim in self.hidden_dims]   
        u = torch.zeros(self.batch_size, self.memory_dim).to(self.device)
        all_h = []  
        all_c = []  
        for t in range(time_stamps):       
            prev_h = h
            new_h = []
            new_c = []
            for i in range(self.num_modalities):
                c_i,h_i = self.lsthms[i](in_modalities[i][:,t,:], c[i],h[i],z)
                
                new_h.append(h_i)
                new_c.append(c_i)
                
            # Concatenate h_t and h (h_(t-1)) and build DFG
#            prev_c_cat = torch.cat(prev_h, dim = 1)
#            new_c_cat = torch.cat(new_c,dim = 1)
        
#            h_star = torch.cat([*prev_h, *new_h],dim = 1)
            dfg_in_modalities = []
            for i in range(self.num_modalities):
                in_modality = self.dfg_init_networks[i](torch.cat([prev_h[i], new_h[i]],dim = 1))
                dfg_in_modalities.append(in_modality)
            
            dfg_output = self.dfg(dfg_in_modalities)[0]
            gamma1 = self.fc_gamma[0](dfg_output)
            gamma2 = self.fc_gamma[1](dfg_output)
            new_memory = self.fc_memory(dfg_output)
            u = gamma1 * u + gamma2 * new_memory
                         
            # Update h and z
            z = self.feedback(dfg_output)
#            z = torch.softmax(z,dim = 1)
            h = new_h
            c = new_c
            
            all_h.append(h)
            all_c.append(c)
            
        last_h = all_h[-1]
        total_output = torch.cat([*last_h,u],dim = 1)
        output = self.fc_out(total_output)
        return output


class DynamicFusionGraph(nn.Module):

    def __init__(self,pattern_model,in_dimensions,out_dimension,efficacy_model,device = torch.device('cpu')):
        super(DynamicFusionGraph, self).__init__()

        self.num_modalities=len(in_dimensions)
        self.in_dimensions=in_dimensions
        self.out_dimension=out_dimension

        #in this part we sort out number of connections, how they will be connected etc.
        self.powerset=list(chain.from_iterable(combinations(range(self.num_modalities), r) for r in range(self.num_modalities+1)))[1:]

        #initializing the models inside the DFG
        self.input_shapes={tuple([key]):value for key,value in zip(range(self.num_modalities),in_dimensions)}
        self.networks={}
        self.total_input_efficacies=0
        self.device = device
        for key in self.powerset[self.num_modalities:]:
            #connections coming from the unimodal components
            unimodal_dims=0
            for modality in key:
                unimodal_dims+=in_dimensions[modality]
            multimodal_dims=((2**len(key)-2)-len(key))*out_dimension
            self.total_input_efficacies+=2**len(key)-2
            #for the network that outputs key component, what is the input dimension
            final_dims=unimodal_dims+multimodal_dims
            self.input_shapes[key]=final_dims
            pattern_copy=copy.deepcopy(pattern_model)
            final_model=nn.Sequential(*[nn.Linear(self.input_shapes[key],list(pattern_copy.children())[0].in_features)],pattern_copy).to(self.device)
            self.networks[key]=final_model
        #finished construction weights, now onto the t_network which summarizes the graph
        self.total_input_efficacies+=2**self.num_modalities-1
        self.t_in_dimension=unimodal_dims+(2**self.num_modalities-(self.num_modalities)-1)*out_dimension
        pattern_copy=copy.deepcopy(pattern_model)
        self.t_network=nn.Sequential(*[nn.Linear(self.t_in_dimension,list(pattern_copy.children())[0].in_features),pattern_copy]).to(self.device)
        self.efficacy_model=nn.Sequential(*[nn.Linear(sum(in_dimensions),list(efficacy_model.children())[0].in_features),efficacy_model,nn.Linear(list(efficacy_model.children())[-1].out_features,self.total_input_efficacies)]).to(self.device)

    def forward(self,in_modalities):

        outputs={}
        for modality,index in zip(in_modalities,range(len(in_modalities))):
            outputs[tuple([index])]=modality
        
        efficacies=self.efficacy_model(torch.cat([x for x in in_modalities],dim=1))
        efficacy_index=0
        for key in self.powerset[self.num_modalities:]:
            small_power_set=list(chain.from_iterable(combinations(key, r) for r in range(len(key)+1)))[1:-1]
            this_input=torch.cat([outputs[x]*efficacies[:,efficacy_index+y].view(-1,1) for x,y in zip(small_power_set,range(len(small_power_set)))],dim=1)
            outputs[key]=self.networks[key](this_input)
            efficacy_index+=len(small_power_set)

        small_power_set.append(tuple(range(self.num_modalities)))
        t_input=torch.cat([outputs[x]*efficacies[:,efficacy_index+y].view(-1,1) for x,y in zip(small_power_set,range(len(small_power_set)))],dim=1)
        t_output=self.t_network(t_input)
        return t_output,outputs,efficacies