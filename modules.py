import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math


class DynamicFilterGNN(nn.Module):
    def __init__(self, in_features, out_features,num_nodes, bias=True):
        super(DynamicFilterGNN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.base_filter = nn.Parameter(torch.Tensor(num_nodes, num_nodes))
        # use_gpu = torch.cuda.is_available()
        # self.filter_adjacency_matrix = None
        # #self.base_filter = nn.Parameter(torch.Tensor(in_features, in_features))
        # if use_gpu:
        #     self.filter_adjacency_matrix = Variable(filter_adjacency_matrix.cuda(), requires_grad=False)
        # else:
        # self.filter_adjacency_matrix = Variable(filter_adjacency_matrix, requires_grad=False)
        self.transform = nn.Linear(num_nodes, num_nodes)
        self.weight = Parameter(torch.Tensor(num_nodes, num_nodes))

        if bias:
            self.bias = Parameter(torch.Tensor(num_nodes))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.base_filter.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj_mat):
        transformed_filter = self.transform(self.base_filter)
        transformed_adjacency = 0.9*adj_mat+0.1*transformed_filter
        input = input.transpose(-1,-2)
        result_embed = F.linear(input, transformed_adjacency.matmul(self.weight), self.bias)
        result_embed = result_embed.transpose(-1,-2).contiguous()
        return result_embed

    
    def get_transformed_adjacency(self):
        transformed_filter = self.transform(self.base_filter)
        transformed_adjacency = 0.9 * self.filter_adjacency_matrix + 0.1 * transformed_filter
        return transformed_adjacency
    
    
    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', bias=' + str(self.bias is not None) + ')'

