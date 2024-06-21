import torch.nn as nn
import torch
class OldPatchEmbedding(nn.Module):
    """Patchify time series."""

    def __init__(self, patch_size, in_channel, embed_dim, norm_layer, num_feats=45):
        super().__init__()
        self.output_channel = embed_dim
        self.len_patch = patch_size             # the L 12
        self.input_channel = in_channel
        self.output_channel = embed_dim
        self.input_embedding = nn.Conv2d(
            in_channel,
            embed_dim,
            kernel_size=(self.len_patch, num_feats),
            stride=(self.len_patch, num_feats))
        self.norm_layer = norm_layer if norm_layer is not None else nn.Identity()

    def forward(self, long_term_history):
        """
        Args:
            long_term_history (torch.Tensor): Very long-term historical MTS with shape [B, N, 1, P * L],
                                                which is used in the TSFormer.
                                                P is the number of segments (patches).

        Returns:
            torch.Tensor: patchified time series with shape [B, N, d, P]
        """
        batch_size, num_nodes, num_feat, len_time_series = long_term_history.shape
        # torch.Size([32, 207, 1, 2016])
        long_term_history = long_term_history.unsqueeze(-1)  # B, N, C, L, 1
        # B*N,  C, L, 1 因为batch和实际上每个元素都是一样的，因为我们此时只考虑时间序列
        long_term_history = long_term_history.reshape(batch_size * num_nodes, 1, len_time_series, num_feat)
        # torch.Size([6624, 1, 2016, 1])
        # B*N,  d, L/P, 1 这里就是卷积一下
        output = self.input_embedding(long_term_history)
        # torch.Size([6624, 96, 168, 1]) 输出结果 ,2016 / 12 相当于所有的长度为12的窗口，然后96则是embed_dim嵌入维度
        # norm 实际是None
        output = self.norm_layer(output)
        # reshape
        output = output.squeeze(-1).view(batch_size, num_nodes, self.output_channel, -1)    # B, N, d, P

        assert output.shape[-1] == len_time_series / self.len_patch
        return output

# 实例化模块
patch_size = 1  
in_channel = 1  # 输入张量的通道数
embed_dim = 96  # 嵌入的维度，可以根据需要调整
norm_layer = None  # 选择使用 LayerNorm 作为归一化层
num_feats = 1  # 输入特征的数量

patch_embedding = OldPatchEmbedding(patch_size, in_channel, embed_dim, norm_layer, num_feats)

# 测试
input_tensor = torch.randn(48, 312, 1, 12)
output = patch_embedding(input_tensor)
print(output.shape)  # 应输出 [48, 312, 96, 12]
