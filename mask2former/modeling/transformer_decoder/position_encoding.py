# Modified by Feng Chen from https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/transformer_decoder/position_encoding.py
"""
Various positional encodings for the transformer.
"""
import math

import torch
from torch import nn
import numpy as np


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
    
    def __repr__(self, _repr_indent=4):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
            "normalize: {}".format(self.normalize),
            "scale: {}".format(self.scale),
        ]
        # _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
    
class PositionEmbeddingHarmonicGuided(nn.Module):
    def __init__(self, sins, num_pos_feats=128, p_shift=True):
        super().__init__()
        assert num_pos_feats % sins.shape[0] == 0, "num_pos_feats should be divisible by the number of guide functions"
        expand_times = num_pos_feats // sins.shape[0] # e.g. 256 // 16 = 16
        expand_p_shift = np.array([2*math.pi/expand_times*i for i in range(expand_times)] * sins.shape[0])
        self.sins = sins.repeat(expand_times, axis=0) # expanded sins
        if p_shift:
            self.sins[:,2] = self.sins[:,2] + expand_p_shift

    def forward(self, x, mask=None):
        # expand sins to the dimension of x
        expand_sins = np.copy(self.sins)
        bs, c_dim, x_dim, y_dim = x.size(0), x.size(1), x.size(2), x.size(3)

        # spatial-wise expansion
        pos = create_guide_function(expand_sins[:,0], # a
                                    expand_sins[:,1], # b
                                    expand_sins[:,2], # phase
                                    x_dim, y_dim)
        
        pos = pos.unsqueeze(0).repeat(bs, 1, 1, 1)
        
        return pos.detach().to(x.device)

# # version without p_shift (need to be the same dim as original P.E.)
# # the num_pos_feats is the total number of P.E., not like P.E. sine that is half
# class PositionEmbeddingHarmonic(nn.Module):
#     def __init__(self, sins):
#         super().__init__()
#         self.sins = sins
#         self.num_pos_feats = len(sins)

#     def forward(self, x, mask=None):
#         # expand sins to the dimension of x
#         org_sins = np.copy(self.sins)
#         bs, c_dim, x_dim, y_dim = x.size(0), x.size(1), x.size(2), x.size(3)

#         # spatial-wise expansion
#         pos = create_guide_function(org_sins[:,0], # a
#                                     org_sins[:,1], # b
#                                     org_sins[:,2], # phase
#                                     x_dim, y_dim)
        
#         pos = pos.unsqueeze(0).repeat(bs, 1, 1, 1)
        
#         return pos.detach().to(x.device)

def create_guide_function(alpha, beta, phase, x_dim, y_dim):
    alpha = torch.FloatTensor(alpha).unsqueeze(-1).unsqueeze(-1)
    beta = torch.FloatTensor(beta).unsqueeze(-1).unsqueeze(-1)
    phase= torch.FloatTensor(phase).unsqueeze(-1).unsqueeze(-1)

    # normailse has been done here
    xx_channel = torch.linspace(0.,1.,x_dim).repeat(1, y_dim, 1).float()
    yy_channel = torch.linspace(0.,1.,y_dim).repeat(1, x_dim, 1).transpose(1, 2).float()

    xx_channel = xx_channel * alpha
    yy_channel = yy_channel * beta
    return torch.sin(xx_channel + yy_channel + phase)
