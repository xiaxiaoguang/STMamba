import torch.nn as nn
import torch
from mamba_ssm import Mamba

class imamba(nn.Module):
    def __init__(self,
                 num_len,pred_len,embed_dim,
                   d_state, d_conv, expand, n_layers):
        super(imamba, self).__init__()
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.linear1 = nn.Linear(num_len,embed_dim)
        self.mamba1_layers = nn.ModuleList([
            Mamba(
                d_model=embed_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            ) for _ in range(n_layers)
        ])
        self.mamba2_layers = nn.ModuleList([
            Mamba(
                d_model=embed_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            ) for _ in range(n_layers)
        ])
        self.linear2 = nn.Linear(embed_dim, pred_len)

    def forward(self, inputs):
        embedded = self.linear1(inputs.permute(0, 2, 1).contiguous())
        for i in range(self.n_layers):
            result1 = self.mamba1_layers[i](embedded)
            embedded = torch.flip(embedded,[1])
            result2 = self.mamba2_layers[i](embedded)
            result2 = torch.flip(result2,[1])
            embedded = nn.SiLU()(result1 + result2)
        output = self.linear2(embedded).transpose(1,2).contiguous()  # [batch_size, node_num * step_size, 1]
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
