# Discrete Graph Learning
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import math

def batch_cosine_similarity(x, y):
    l2_x = torch.norm(x, dim=2, p=2) + 1e-7  # avoid 0, l2 norm, num_heads x batch_size x hidden_dim==>num_heads x batch_size
    l2_y = torch.norm(y, dim=2, p=2) + 1e-7  # avoid 0, l2 norm, num_heads x batch_size x hidden_dim==>num_heads x batch_size
    l2_m = torch.matmul(l2_x.unsqueeze(dim=2), l2_y.unsqueeze(dim=2).transpose(1, 2))
    l2_z = torch.matmul(x, y.transpose(1, 2))
    cos_affnity = l2_z / l2_m
    adj = cos_affnity
    return adj

def batch_dot_similarity(x, y):
    QKT = torch.bmm(x, y.transpose(-1, -2)) / math.sqrt(x.shape[2])
    W = torch.softmax(QKT, dim=-1)
    return W

def sample_gumbel(shape, eps=1e-20, device=None):
    uniform = torch.rand(shape).to(device)
    return -torch.autograd.Variable(torch.log(-torch.log(uniform + eps) + eps))


def gumbel_softmax_sample(logits, temperature, eps=1e-10):
    sample = sample_gumbel(logits.size(), eps=eps, device=logits.device)
    y = logits + sample
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, hard=False, eps=1e-10):
    y_soft = gumbel_softmax_sample(logits, temperature=temperature, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        y_hard = torch.zeros(*shape).to(logits.device)
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        y = torch.autograd.Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


class DiscreteGraphLearning(nn.Module):
    """Dynamic graph learning module."""

    def __init__(self, dataset_name, 
                 k , 
                 input_seq_len,
                 num_nodes,
                 output_seq_len,
                 embedding_dim=96):
        super().__init__()
        self.k = k          # the "k" of knn graph
        self.num_nodes=num_nodes
        self.train_length = {"PEMS04": 6000}[dataset_name]
        self.dim_fc = {"PEMS04":95712}[dataset_name]

        self.node_feats = torch.from_numpy(np.load("dataset/data_{0}_{1}.npy".format(input_seq_len, output_seq_len))).float()[:self.train_length, :]

        self.embedding_dim = embedding_dim
        self.conv1 = torch.nn.Conv1d(1, 8, 10, stride=1)
        self.conv2 = torch.nn.Conv1d(8, 16, 10, stride=1) 

        self.fc = torch.nn.Linear(self.dim_fc, self.embedding_dim)
        self.bn1 = torch.nn.BatchNorm1d(8)
        self.bn2 = torch.nn.BatchNorm1d(16)
        self.bn3 = torch.nn.BatchNorm1d(self.embedding_dim)

        self.fc_cat = nn.Linear(self.embedding_dim, 2)
        self.fc_out = nn.Linear((self.embedding_dim) * 2, self.embedding_dim)
        self.dropout = nn.Dropout(0.5)
        
        def encode_one_hot(labels):
            classes = set(labels)
            classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
            labels_one_hot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
            return labels_one_hot

        self.rel_rec = torch.FloatTensor(np.array(encode_one_hot(np.where(np.ones((self.num_nodes, self.num_nodes)))[0]), dtype=np.float32))
        self.rel_send = torch.FloatTensor(np.array(encode_one_hot(np.where(np.ones((self.num_nodes, self.num_nodes)))[1]), dtype=np.float32))
    def get_k_nn_neighbor(self, data, pls=5 , k=10*500, metric="cosine"):
        """找到k近邻然后构图
        data: tensor B, N, D
        metric: cosine or dot
        """
        if metric == "cosine":
            batch_sim = batch_cosine_similarity(data, data)
        elif metric == "dot":
            batch_sim = batch_dot_similarity(data, data)    # B, N, N
        else:
            assert False, "unknown metric"

        batch_size, num_nodes, _ = batch_sim.shape

        adj = batch_sim.view(batch_size, num_nodes*num_nodes)
        res = torch.zeros_like(adj)
        top_k, indices = torch.topk(adj, k, dim=-1) # B,N*K 
        res.scatter_(-1, indices, top_k)
        adj = torch.where(res != 0, 1.0, 0.0).detach().clone()
        adj = adj.view(batch_size, num_nodes, num_nodes)

        top_pls,plsedge = torch.topk(batch_sim,pls,dim=2)
        res = torch.zeros_like(batch_sim)
        res.scatter_(-1,plsedge,top_pls)
        adj += torch.where(res != 0, 1.0 ,0.0).detach().clone()

        adj = torch.clamp(adj,max=1,min=0)

        adj.requires_grad = False
        return adj

    def forward(self, long_term_history):
        device = long_term_history.device
        _, _, num_nodes = long_term_history.shape

        global_feat = self.node_feats.to(device).transpose(1, 0).view(num_nodes, 1, -1) #torch.Size([207, 1, 23990]) , 207 是图节点数
        global_feat = self.bn1(F.relu(self.conv1(global_feat)))  # torch.Size([207, 8, 23981])      
        global_feat = self.bn2(F.relu(self.conv2(global_feat))) # torch.Size([207, 16, 23972])
        global_feat = global_feat.view(num_nodes, -1) # global feat 
        global_feat = self.bn3(F.relu(self.fc(global_feat)))
        # global_feat = global_feat.unsqueeze(0).expand(batch_size, num_nodes, -1)    # Gi in Eq. (2) torch.Size([32, 207, 100])
        node_feat = global_feat
        receivers = torch.matmul(self.rel_rec.to(node_feat.device), node_feat)
        senders = torch.matmul(self.rel_send.to(node_feat.device), node_feat)
        edge_feat = torch.cat([senders, receivers], dim=-1)
        edge_feat = torch.relu(self.fc_out(edge_feat))
        bernoulli_unnorm = self.fc_cat(edge_feat)

        sampled_adj = gumbel_softmax(bernoulli_unnorm, temperature=0.5, hard=True)
        sampled_adj = sampled_adj[..., 0].clone().reshape(num_nodes, -1)
        mask = torch.eye(num_nodes, num_nodes).bool().to(sampled_adj.device)
        sampled_adj.masked_fill_(mask, 0)

        return sampled_adj
