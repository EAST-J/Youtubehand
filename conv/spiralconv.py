# spiralconv implemented by https://github.com/sw-gong/spiralnet_plus
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

class Spiralconv(nn.Module):
    def __init__(self, in_channels, out_channels, indices, dim=1):
        super().__init__()
        self.dim = dim
        self.indices = indices
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_length = indices.size(1)
        self.layer = nn.Linear(in_channels * self.seq_length, out_channels)
        
    def forward(self, x):
        n_nodes, _ = self.indices.size()
        if x.dim() == 2:
            x = torch.index_select(x, 0, self.indices.to(x.device).reshape(-1))
            x = x.reshape(n_nodes, -1)
        elif x.dim() == 3:
            bs = x.size(0)
            x = torch.index_select(x, self.dim, self.indices.to(x.device).reshape(-1))
            x = x.reshape(bs, n_nodes, -1)
        else:
            raise RuntimeError(
            "Spiral Conv x.dim() is expected as 2 or 3, but get {}".format(x.dim())
        )
        x = self.layer(x)
        return x

def Pool(x, trans, dim=1):
    """
    x:input feature
    trans:sample matrix
    dim:sample dim
    return:sampled feature
    """
    trans = trans.to(x.device)
    row, col = trans._indices()
    value = trans._values().unsqueeze(-1)
    out = torch.index_select(x, dim, col) * value
    out = scatter_add(out, row, dim, dim_size=trans.size(0))
    return out


class SpiralDeblock(nn.Module):
    # Decoder that include upsampling and GCN
    def __init__(self, in_channels, out_channels, indices):
        super().__init__()
        self.conv = Spiralconv(in_channels=in_channels, out_channels=out_channels, indices=indices)

    def forward(self, x, up_transform):
        out = Pool(x, up_transform)
        out = F.relu(self.conv(out))

        return out
