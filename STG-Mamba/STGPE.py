import torch
import torch.nn as nn
import math
class GraphTemporalEmbedding(nn.Module):
    def __init__(self, node_feature_size, embedding_size, num_nodes, stepsize):
        super(GraphTemporalEmbedding, self).__init__()
        self.embedding_size = embedding_size
        self.node_feature_size = node_feature_size
        self.num_nodes = num_nodes
        self.stepsize = stepsize

        # Linear layer for node feature embedding
        self.node_embedding = nn.Linear(node_feature_size, embedding_size)
        
        # GCN layer
        self.gcn = GraphConvolution(node_feature_size, embedding_size)

        # Create absolute position embeddings
        self.position_embeds = self.create_position_embeddings(stepsize, embedding_size)

    @staticmethod
    def create_position_embeddings(stepsize, embedding_size):
        position = torch.arange(stepsize, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2, dtype=torch.float) * (-math.log(10000.0) / embedding_size))
        pos_embed = torch.zeros(stepsize, embedding_size, dtype=torch.float)
        pos_embed[:, 0::2] = torch.sin(position * div_term)
        pos_embed[:, 1::2] = torch.cos(position * div_term)
        return pos_embed.unsqueeze(0)  # Shape: (1, stepsize, embedding_size)

    def forward(self, x, A):
        batchsize, stepsize, num_nodes, feature_size = x.size()
        assert num_nodes == self.num_nodes, "Number of nodes must match."
        assert stepsize == self.stepsize, "Stepsize must match."

        # Embedding for node features
        # node_embedded = self.node_embedding(x)  # (batchsize, stepsize, num_nodes, embedding_size)

        # Graph convolution to incorporate adjacency matrix information
        gcn_embedded = self.gcn(x, A)  # (batchsize, stepsize, num_nodes, embedding_size)

        # Apply absolute position embeddings
        pos_embeds = self.position_embeds.repeat(batchsize, 1, 1).unsqueeze(2).expand(-1, -1, num_nodes, -1)  # (batchsize, stepsize, num_nodes, embedding_size)
        #node_embedded = pos_embeds.cuda()
        combined_embedding = gcn_embedded.cuda() + pos_embeds.cuda() #+ node_embedded.cuda()

        # Reshape to desired output
        output_embedding = combined_embedding.view(batchsize, stepsize * num_nodes, self.embedding_size)
        return output_embedding

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x, A):
        batchsize, stepsize, num_nodes, feature_size = x.size()
        
        # Ensure A is on the same device as x
        device = x.device
        A = torch.tensor(A, dtype=torch.float, device=device).unsqueeze(0).expand(batchsize, stepsize, num_nodes, num_nodes)
        
        # Normalize A to ensure numerical stability
        D = torch.sum(A, dim=-1, keepdim=True)
        D_inv_sqrt = torch.pow(D, -0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0
        A_norm = D_inv_sqrt * A * D_inv_sqrt.transpose(-1, -2)
        
        x = torch.einsum('bsij,bsjf->bsif', A_norm, x)
        x = self.linear(x)
        
        return x

# # Example usage
# batchsize = 32
# stepsize = 10
# nodes = 5
# embedding_size = 96
# node_feature_size = 1

# model = GraphTemporalEmbedding(node_feature_size, embedding_size, nodes, stepsize)

# input_tensor = torch.randn(batchsize, stepsize, nodes, node_feature_size)
# adj_matrix = torch.randn(batchsize, nodes, nodes)  # Example adjacency matrix

# output = model(input_tensor, adj_matrix)
# print(output.shape)  # Should be (batchsize, stepsize * nodes, embedding_size)