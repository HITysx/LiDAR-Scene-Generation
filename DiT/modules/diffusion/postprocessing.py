import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class ImagePatchEmbed(nn.Module):
  """
    Image to Patch Embedding
  """
 
  def __init__(self, scale_num, input_h, input_w, patch_h, patch_w, in_chans, embed_dim):
    super().__init__()
    img_size = (input_h, input_w)
    patch_size = (patch_h, patch_w)
    num_patches = (input_h // patch_h) * (input_w // patch_w)
    self.img_size = img_size
    self.patch_size = patch_size
    self.num_patches = num_patches
    self.proj = nn.Conv2d(scale_num * in_chans, scale_num * embed_dim, kernel_size=patch_size, stride=patch_size)
 
  def forward(self, x):
    B, C, H, W = x.shape
    assert H == self.img_size[0] and W == self.img_size[1], \
      f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
    x = self.proj(x).flatten(2).transpose(1, 2)
    return x  

class VoxelPatchEmbed(nn.Module):
  """
    Voxel to Patch Embedding
  """
 
  def __init__(self, scale_num, input_size, patch_size, in_chans, embed_dim):
    super().__init__()
    self.voxel_size = input_size
    self.patch_size = patch_size
    num_patches = (input_size // patch_size) * (input_size // patch_size) * (input_size // patch_size)
    self.num_patches = num_patches
    self.proj = nn.Conv3d(scale_num * in_chans, scale_num * embed_dim, kernel_size=patch_size, stride=patch_size)
 
  def forward(self, y):
    B, C, D, D, D = y.shape
    assert D == self.voxel_size, \
        f"Input voxel size ({D}) doesn't match model ({self.voxel_size})."
    y = self.proj(y).flatten(2).transpose(1, 2)
    return y

class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, activation, k=1, s=1, p=0):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.norm = torch.nn.BatchNorm2d(out_ch, eps=1E-6)
        self.relu = activation

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))




class LiteMLA(torch.nn.Module):
    """
    Lightweight Multiscale Linear Attention
    """

    def __init__(self,image_size, patch_size_img, voxel_size, patch_size_voxel, in_channels,out_channels,heads=8,dim=1,scales=(5,),eps=1.0e-15):
        super().__init__()
        self.eps = eps
        heads = heads or int(in_channels // dim)

        # total_dim == in_channels
        total_dim = heads * dim
        self.image_size = image_size
        self.patch_size_img = patch_size_img
        self.voxel_size = voxel_size
        self.patch_size_voxel = patch_size_voxel
        self.dim = dim
        self.qkv_image = torch.nn.Conv2d(in_channels, total_dim, kernel_size=1, bias=False)
        self.qkv_voxel = torch.nn.Conv3d(in_channels, 2 * total_dim, kernel_size=1, bias=False)
        self.image_patch = ImagePatchEmbed(1 + len(scales), image_size[0], image_size[1], patch_size_img[0], patch_size_img[1], total_dim, total_dim)
        self.voxel_patch = VoxelPatchEmbed(1 + len(scales), voxel_size, patch_size_voxel, 2*total_dim, 2*total_dim)
        self.num_patches_img = self.image_patch.num_patches
        self.num_patches_voxel = self.voxel_patch.num_patches
        self.pos_embed_img = nn.Parameter(torch.zeros(1, self.num_patches_img, (1 + len(scales)) * total_dim), requires_grad=False)
        self.pos_embed_voxel = nn.Parameter(torch.zeros(1, self.num_patches_voxel, 2 * total_dim * (1 + len(scales))), requires_grad=False)
        self.aggreg_image = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Conv2d(total_dim, total_dim,
                                                                               kernel_size=scale,
                                                                               padding=scale // 2,
                                                                               groups=total_dim, bias=False),
                                                               torch.nn.Conv2d(total_dim, total_dim,
                                                                               kernel_size=1,
                                                                               groups=heads, bias=False))
                                           for scale in scales])
        self.aggreg_voxel = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Conv3d(2 * total_dim, 2 * total_dim,
                                                                               kernel_size=scale,
                                                                               padding=scale // 2,
                                                                               groups=2 * total_dim, bias=False),
                                                               torch.nn.Conv3d(2 * total_dim, 2 * total_dim,
                                                                               kernel_size=1,
                                                                               groups=2 * heads, bias=False))
                                           for scale in scales])
        self.kernel_func = torch.nn.ReLU(inplace=False)
        self.linear = torch.nn.Linear(self.dim, out_channels* patch_size_img[0]* patch_size_img[1], bias=False)
        self.proj = Conv(total_dim * (1 + len(scales)), out_channels, torch.nn.Identity())
    
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed_img = get_2d_sincos_pos_embed(self.pos_embed_img.shape[-1], int(self.num_patches_img ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed_img).float().unsqueeze(0))
        
    @torch.cuda.amp.autocast(enabled=False)
    def relu_linear_att(self, q, kv):
       
        B, C1, H, W = list(q.size())
        B, C2, D, D, D = list(kv.size())
        if q.dtype == torch.float16:
            q = q.float()
        if kv.dtype == torch.float16:
            kv = kv.float()
       
        
        q = self.image_patch(q) + self.pos_embed_img
        kv = self.voxel_patch(kv)
        q_length = q.size(1)
        kv_length = kv.size(1)
        q = q.reshape(B, -1, q_length, self.dim).contiguous()
        kv = kv.reshape(B, -1, kv_length, 2*self.dim).contiguous()
        k, v = (kv[..., 0: self.dim], kv[..., self.dim: 2 * self.dim],)
        # lightweight linear attention
        q = self.kernel_func(q) 
        k = self.kernel_func(k) 
        v = F.pad(v, (0, 1), mode="constant", value=1) 
        kv = torch.matmul(k.transpose(-1, -2), v)  
        out = torch.matmul(q, kv)  
        out = out[..., :-1] / (out[..., -1:] + self.eps)  
        out = self.linear(out)  
        out = torch.transpose(out, -1, -2)  
        out = out.reshape(B, -1, H, W).contiguous()   
        
        return out

    def forward(self, x, y):

        # generate multi-scale q, k, v
        q = self.qkv_image(x)  
        kv = self.qkv_voxel(y)
        multi_scale_q = [q] 
        multi_scale_kv = [kv]
        
        for op in self.aggreg_image:
            multi_scale_q.append(op(q))
        
        for oq in self.aggreg_voxel:
            multi_scale_kv.append(oq(kv))
 
        multi_scale_q = torch.cat(multi_scale_q, dim=1)  
        multi_scale_kv = torch.cat(multi_scale_kv, dim=1)
        out = self.relu_linear_att(multi_scale_q, multi_scale_kv)  

        return x + self.proj(out)

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
   
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
   
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

if __name__ == '__main__':
    # (B, C, H, W)
    X = torch.randn(8, 1, 64, 1024)
    Y = torch.randn(8, 1, 32, 32, 32)
    
    Model = LiteMLA(image_size=[X.shape[2],X.shape[3]], patch_size_img=[2, 32], voxel_size=Y.shape[2], patch_size_voxel=4, in_channels=1, out_channels=1, scales=(5,))
    out = Model(X, Y)
    print(out.shape)
