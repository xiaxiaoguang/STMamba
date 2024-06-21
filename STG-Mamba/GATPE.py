import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data, Batch

class GATPEEmbedding(nn.Module):
    def __init__(self, node_feature_size, embedding_size, num_nodes, stepsize, heads=1, concat=True):
        super(GATPEEmbedding, self).__init__()
        self.embedding_size = embedding_size
        self.node_feature_size = node_feature_size
        self.num_nodes = num_nodes
        self.stepsize = stepsize

        # Linear layer for node feature embedding
        self.node_embedding = nn.Linear(node_feature_size, embedding_size)
        
        # GATv2 layer
        self.gatv2 = GATv2Encoder(node_feature_size, embedding_size, heads=heads, concat=concat)

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
        gcn_embedded = self.gatv2(x, A)  # (batchsize, stepsize, num_nodes, embedding_size)

        # Apply absolute position embeddings
        pos_embeds = self.position_embeds.repeat(batchsize, 1, 1).unsqueeze(2).expand(-1, -1, num_nodes, -1)  # (batchsize, stepsize, num_nodes, embedding_size)
        #node_embedded = pos_embeds.cuda()
        combined_embedding = gcn_embedded.cuda() + pos_embeds.cuda() #+ node_embedded.cuda()

        # Reshape to desired output
        output_embedding = combined_embedding.view(batchsize, stepsize * num_nodes, self.embedding_size)
        return output_embedding

class GATv2Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, concat=True):
        super(GATv2Encoder, self).__init__()
        self.gatv2_conv = GATv2Conv(in_channels, out_channels, heads=heads, concat=concat)
        
    def forward(self, x, A):
        batchsize, stepsize, num_nodes, feature_size = x.size()
        device = x.device

        # Ensure A is on the same device as x
        A = torch.tensor(A).to(device)

        # Convert adjacency matrix to edge_index
        edge_index = torch.nonzero(A).t().contiguous()

        # Prepare a list to collect all data objects
        data_list = []
        for b in range(batchsize):
            for t in range(stepsize):
                features = x[b, t]
                data = Data(x=features, edge_index=edge_index)
                data_list.append(data)

        # Create a batch of data objects
        batch = Batch.from_data_list(data_list).to(device)
        
        # Apply GATv2Conv
        encoded_features = self.gatv2_conv(batch.x, batch.edge_index)
        
        # Reshape the encoded features to the original batch and time step dimensions
        encoded_features = encoded_features.view(batchsize, stepsize, num_nodes, -1)

        return encoded_features

# Example usage
batchsize = 32
stepsize = 10
nodes = 5
embedding_size = 96
node_feature_size = 1
heads = 1

model = GATPEEmbedding(node_feature_size, embedding_size, nodes, stepsize, heads=heads, concat=True)

input_tensor = torch.randn(batchsize, stepsize, nodes, node_feature_size)
adj_matrix = torch.randint(0, 2, (nodes, nodes))  # Example adjacency matrix

output = model(input_tensor, adj_matrix)
print(output.shape)  # Should be (batchsize, stepsize * nodes, embedding_size)
