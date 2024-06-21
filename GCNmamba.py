import torch.nn as nn
import torch
from mamba_ssm import Mamba
from DGL import DiscreteGraphLearning
from modules import DynamicFilterGNN

class OldPatchEmbedding(nn.Module):
    """Patchify time series."""

    def __init__(self, patch_size, in_channel, embed_dim, norm_layer, num_feats=45):
        super().__init__()
        self.output_channel = embed_dim
        self.len_patch = patch_size
        self.input_channel = in_channel
        self.output_channel = embed_dim
        self.input_embedding = nn.Conv2d(
            in_channel,
            embed_dim,
            kernel_size=(self.len_patch, num_feats),
            stride=(self.len_patch, num_feats))
        self.norm_layer = norm_layer if norm_layer is not None else nn.Identity()

    def forward(self, long_term_history):
        batch_size, num_nodes, num_feat, len_time_series = long_term_history.shape
        long_term_history = long_term_history.unsqueeze(-1)
        long_term_history = long_term_history.reshape(batch_size * num_nodes, 1, len_time_series, num_feat)
        output = self.input_embedding(long_term_history)
        output = self.norm_layer(output)
        output = output.squeeze(-1).view(batch_size, num_nodes, self.output_channel, -1)
        assert output.shape[-1] == len_time_series / self.len_patch
        return output



class GCNmamba(nn.Module):
    def __init__(self, patch_size, in_channel,
                 num_len,pred_len,num_nodes,
                 embed_dim, norm_layer, num_feats,
                   d_model, d_state, d_conv, expand, n_layers):
        super(GCNmamba, self).__init__()
        self.patch_embedding = OldPatchEmbedding(patch_size, in_channel, embed_dim, norm_layer, num_feats)
        self.embed_dim = embed_dim
        self.node_num = num_nodes
        self.n_layers = n_layers
        self.patch_size = patch_size
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            ) for _ in range(n_layers)
        ])
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(self.node_num,self.node_num,kernel_size=(2,1),stride=(2,1)),
            nn.Conv2d(self.node_num,self.node_num,kernel_size=(2,1),stride=(2,1)),
            nn.Conv2d(self.node_num,self.node_num,kernel_size=(2,1)),
            nn.Conv2d(self.node_num,self.node_num,kernel_size=(2,1)),
        ])
        self.GraphGenerate = DiscreteGraphLearning("PEMS04",5,num_len,num_nodes,pred_len)
        self.graph_layers = nn.ModuleList([
            DynamicFilterGNN(embed_dim,embed_dim,num_nodes),
            DynamicFilterGNN(embed_dim,embed_dim,num_nodes),
            DynamicFilterGNN(embed_dim,embed_dim,num_nodes),
            DynamicFilterGNN(embed_dim,embed_dim,num_nodes)
        ])
        self.linear = nn.Linear(d_model, pred_len)

    def forward(self, inputs):
        batch_size, step_size, node_num = inputs.shape
        sampled_adj = self.GraphGenerate(inputs)

        self.node_num = node_num
        inputs = inputs.permute(0, 2, 1).unsqueeze(2)  # [batch_size, node_num, 1, step_size]
        embedded = self.patch_embedding(inputs)  # [batch_size, node_num, embed_dim, step_size]
        step_size //= self.patch_size
        embedded = embedded.view(batch_size, node_num * step_size, self.embed_dim)  # [batch_size, node_num * step_size, embed_dim]
        
        for i in range(self.n_layers):
            embedded = self.mamba_layers[i](embedded)
            embedded = embedded.view(batch_size,node_num,-1,self.embed_dim)
            embedded = nn.SiLU()(self.conv_layers[i](embedded))
            embedded = embedded.reshape(-1,node_num,self.embed_dim).contiguous()
            embedded = nn.SiLU()(self.graph_layers[i](embedded,sampled_adj))
            embedded = embedded.view(batch_size,-1,self.embed_dim)

        output = embedded.view(batch_size, node_num, -1)  # [batch_size, node_num, step_size, embedding_dim]        
        output = self.linear(output).transpose(1,2).contiguous()  # [batch_size, node_num * step_size, 1]
        return output

# if __name__ == "__main__":
#     # Example usage
#     patch_size = 1
#     in_channel = 1
#     embed_dim = 96
#     norm_layer = None
#     num_feats = 1
#     fea_size = 96
#     d_model = 96
#     d_state = 64
#     d_conv = 4
#     expand = 2
#     n_layers = 2
#     model = GCNmamba(patch_size, in_channel, embed_dim, norm_layer, num_feats, fea_size, d_model, d_state, d_conv, expand, n_layers).cuda()
#     # Test
#     input_tensor = torch.randn(48, 12, 312).cuda()
#     output = model(input_tensor)
#     print(output.shape)  # Should output [48, 12, 312]
