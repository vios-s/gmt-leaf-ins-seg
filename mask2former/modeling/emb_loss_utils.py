import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import cv2

def l1_loss(inputs: torch.Tensor,
        targets: torch.Tensor,
        weight: torch.Tensor,
        edges = None,
    ):
    """
    l1_loss for mask features and embeddings
    Args:
        c is the num of channels of the embeddings (i.e. targets)
        inputs: mask_features (bs x c x h x w)
        targets: embedded GTs (bs x c x h x w)
        weight: c x h x w
        edges (optional): bs x 1 x h x w 
    """
    edge_loss = 0.
    loss = F.l1_loss(inputs, targets, reduce=False)
    if edges is not None:
        edge_loss = torch.mean(torch.mean(loss, dim=1) * edges)
    
    # loss average the dim of c (i.e. expectation of guided functions)
    loss = torch.mean((weight.detach() * torch.mean(loss, dim=1))) + 10. * edge_loss

    return loss

def mse_loss(inputs: torch.Tensor,
        targets: torch.Tensor,
        weight: torch.Tensor,
        edges = None,
    ):
    """
    same as l1_loss, but using mse loss
    """
    edge_loss = 0.
    loss = F.mse_loss(inputs, targets, reduce=False)
    if edges is not None:
        edge_loss = torch.mean(torch.mean(loss, dim=1) * edges)
    
    # loss average the dim of c (i.e. expectation of guided functions)
    loss = torch.mean((weight.detach() * torch.mean(loss, dim=1))) + 10. * edge_loss

    return loss

def get_edge(mask,kernel_size=5):
    num_array = mask.numpy().astype(np.float32)
    nap = np.zeros_like(num_array)
    dIdx = abs(num_array[:-1,:]-num_array[1: ,:])>0
    dIdy = abs(num_array[:,:-1]-num_array[: ,1:])>0
    nap[1:]+=dIdx
    nap[:,1:]+=dIdy
    nap = np.clip(nap,0,1)
    nap = cv2.dilate(nap,np.ones((kernel_size,kernel_size))).astype('uint8')

    return torch.tensor(nap).unsqueeze(dim=0)

def combine_mask(masks):
    single_mask = torch.zeros((masks.shape[1], masks.shape[2]), dtype=torch.uint8) # np (w, h)
    for i in range(masks.shape[0]):
        single_mask[masks[i] > 0] = i+1
    return single_mask

def create_guide_function(alpha,beta,phase,x_dim,y_dim):
    alpha = torch.FloatTensor(alpha).unsqueeze(-1).unsqueeze(-1)
    beta = torch.FloatTensor(beta).unsqueeze(-1).unsqueeze(-1)
    phase= torch.FloatTensor(phase).unsqueeze(-1).unsqueeze(-1)

    xx_channel = torch.linspace(0.,1.,x_dim).repeat(1, y_dim, 1).float()
    yy_channel = torch.linspace(0.,1.,y_dim).repeat(1, x_dim, 1).transpose(1, 2).float()

    xx_channel = xx_channel * alpha
    yy_channel = yy_channel * beta
    return torch.sin(xx_channel+ yy_channel+phase)

def get_embeddings_fast(numpy_gt, sin_pattern, weights_norm=None):
    nobj = numpy_gt.max().item()+1
    nsamples, xdim, ydim = numpy_gt.shape
    nemb = sin_pattern.size(0)

    t = sin_pattern.transpose(1, 0).transpose(2, 1).view(1, -1, nemb).repeat(nsamples, 1, 1)

    indexes_raw = numpy_gt.float()
    indexes = numpy_gt.long()
    w = torch.zeros(nsamples, nobj).to(sin_pattern.device)
    w = w.scatter_add(1, indexes.view(nsamples, -1), torch.ones_like(indexes_raw).view(nsamples, -1))
    e = torch.zeros(nsamples, nobj, nemb).to(sin_pattern.device)
    e = e.scatter_add(1, indexes.view(nsamples, -1, 1).repeat(1, 1, nemb), t)
    w[w == 0] = 1.
    e = e / w.unsqueeze(-1)
    if weights_norm is not None:
        w = weights_norm(w)

    w = torch.gather(w, 1, indexes.view(nsamples, -1))
    e = torch.gather(e, 1, indexes.view(nsamples, -1, 1).repeat(1, 1, nemb))
    e = e.transpose(2, 1).contiguous()

    return e.view(nsamples, nemb, xdim, ydim), w.view(nsamples, xdim, ydim)

def log_weights_norm(gain=1.):
    def f(w):
        w[w < 2] = 2.
        w = gain / torch.log(w)
        return w
    return f

class out_mod(nn.Module):
    def __init__(self, in_ch, out_ch,conv_op=nn.Conv2d):
      super(out_mod, self).__init__()
      self.outc = double_conv(in_ch, in_ch, conv_op) # (conv => BN => ReLU) * 2
      self.attention = nn.Sequential(conv_op(in_ch,out_ch,3,1,1),
                                     nn.Sigmoid())
      self.outc2 =conv_op(in_ch, out_ch, 3,1,1)
      self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def forward(self, xr, y=None):
      x = self.outc2(self.outc(xr)) # in_ch -> out_ch
      if y is not None:
        att = self.attention(xr)
        x = att*x + (1.-att)*self.up(y)
      return x

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch,conv_op=nn.Conv2d):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            conv_op(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            conv_op(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

# class GatedConv(nn.Module):
#     def __init__(self, in_dims, out_dims, kernel_size=3, padding=1, stride=1, dilation=1):
#         super(GatedConv, self).__init__()
#         self.f = nn.Conv2d(in_dims, out_dims, kernel_size=kernel_size, padding=padding, stride=stride,
#                            dilation=dilation)
#         self.g = nn.Conv2d(in_dims, out_dims, kernel_size=kernel_size, padding=padding, stride=stride,
#                            dilation=dilation)

#     def forward(self, x):
#         mask = torch.sigmoid(self.g(x))
#         return self.f(x) * mask

# customise version
class GuidedConv(nn.Module):
    def __init__(self, guides, in_channels, out_channels, kernel_size=3, stride=1, padding=1, conv_op=nn.Conv2d):
        super(GuidedConv,self).__init__()
        self.guides = guides.clone().detach().unsqueeze(0)
        self.guides.requires_grad_(False)
        self.conv = conv_op(in_channels + self.guides.size(1), out_channels, kernel_size, stride, padding)
    def forward(self,x):
        joimnt = torch.cat([x, self.guides.repeat(x.size(0),1,1,1)], dim=1)
        return self.conv(joimnt)