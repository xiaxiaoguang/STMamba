import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data

class GATv2Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, concat=True):
        super(GATv2Encoder, self).__init__()
        self.gatv2_conv = GATv2Conv(in_channels, out_channels, heads=heads, concat=concat)
        
    def forward(self, x, edge_index):
        return self.gatv2_conv(x, edge_index)

def encode_graph_features(adj_matrix, X, in_channels, out_channels, heads=1, concat=True):
    batch_size, time_steps, node_num, _ = X.shape

    # Create edge index from adjacency matrix
    edge_index = torch.nonzero(adj_matrix).t().contiguous()

    # Initialize GATv2 encoder
    gatv2_encoder = GATv2Encoder(in_channels, out_channels, heads, concat)
    
    # To store the encoded features
    encoded_features = torch.zeros(batch_size, time_steps, node_num, out_channels * heads if concat else out_channels)
    
    for b in range(batch_size):
        for t in range(time_steps):
            # Get features for the current batch and time step
            x = X[b, t]

            # Create PyG Data object
            data = Data(x=x, edge_index=edge_index)

            # Encode features
            encoded_features[b, t] = gatv2_encoder(data.x, data.edge_index)
    
    return encoded_features

# Example usage
if __name__ == "__main__":
    batch_size = 2
    time_steps = 3
    node_num = 5
    in_channels = 4
    out_channels = 8
    heads = 2

    # Random adjacency matrix (symmetric for undirected graph)
    adj_matrix = torch.randint(0, 2, (node_num, node_num))
    adj_matrix = adj_matrix | adj_matrix.t()

    # Random traffic flow features
    X = torch.rand(batch_size, time_steps, node_num, in_channels)
    print(X.shape)
    # Encode graph features
    encoded_features = encode_graph_features(adj_matrix, X, in_channels, out_channels, heads=heads, concat=True)
    print(encoded_features.shape)
