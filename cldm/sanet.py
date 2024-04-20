import torch

from torch import nn
from ldm.modules.attention import CrossAttention
from einops import rearrange, repeat

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class sanet(nn.Module):
    def __init__(self, in_channels, n_heads=8, d_head=64,
                 dropout=0.) -> None:
        super().__init__()
        inner_dim = n_heads * d_head
        self.attention = CrossAttention(query_dim=inner_dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                                         context_dim=inner_dim)
        self.norm = Normalize(inner_dim)
    
    def forward(self, x, y):
        b, c, h, w = x.shape
        x = self.norm(x)
        y = self.norm(y)

        x = rearrange(x, 'b c h w -> b c (h w)').contiguous()
        x = rearrange(x, 'b c h -> b h c').contiguous()

        y = rearrange(y, 'b c h w -> b c (h w)').contiguous()
        y = rearrange(y, 'b c h -> b h c').contiguous()

        out = self.attention(x, y)

        out = x + out
        out = rearrange(out, 'b h c -> b c h').contiguous()
        out = rearrange(out, 'b c (h w) -> b c h w', h=h).contiguous()
        return out
    
class SANET(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        channel_size = [40, 40, 40, 40, 80, 80, 80, 160, 160, 160, 160, 160, 160]
        self.net = nn.ModuleList([sanet(in_channels=320, n_heads=8, d_head=channel_size[d],dropout=0.)
            for d in range(13)])
    def forward(self, x, y):
        h = []
        for i in range(13):
            h.append(self.net[i](x[i], y[i]))
        return h
