from abc import abstractmethod
import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Attention, Mlp
from functools import partial
from timm.layers import trunc_normal_tf_
from timm.models import named_apply
from torch import einsum
from einops import rearrange, repeat

from ..basic import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)

from ...modules.attention import CrossAttention, FeedForward

def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def exists(val):
    return val is not None
def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d
# Other types of layers can go here (e.g., nn.Linear, etc.)
def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)  
        avg_result = self.avgpool(x)  
        max_out = self.se(max_result)  
        avg_out = self.se(avg_result)  
        output = max_out + avg_out  
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x:(B,C,H,W)
        max_result, _ = th.max(x, dim=1, keepdim=True)  
        avg_result = th.mean(x, dim=1, keepdim=True)  
        result = th.cat([max_result, avg_result], 1)  
        output = self.conv(result)  
        return output


class CBAMBlock(nn.Module):

    def __init__(self, channel=512, reduction=16, kernel_size=49, HW=None):
        super().__init__()
        self.ChannelAttention = ChannelAttention(channel=channel, reduction=reduction)
        self.SpatialAttention = SpatialAttention(kernel_size=kernel_size)
        self.joint_channel = channel + HW
        self.dimension = nn.Conv2d(channel * 2, channel, 1,  bias=False)
        self.MLP = nn.Sequential(
            nn.Conv2d(self.joint_channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, self.joint_channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        
        B, C, H, W = x.size()
        residual = x
        Channel_x = self.ChannelAttention(x).reshape(B,C,1,1) 
        Spatial_x = self.SpatialAttention(x).reshape(B,H*W,1,1) 

        
        CS_x = th.cat([Channel_x, Spatial_x], 1) 
        CS_xx = self.MLP(CS_x) 

        
        Channel_x = CS_xx[:,:C,:].reshape(B,C,1,1)  # (B,C,1,1)-->(B,C,1,1)
        Spatial_x = CS_xx[:,C:,:].reshape(B,1,H,W) # (B,HW,1,1)-->(B,1,H,W)
        Channel_weight = self.sigmoid(Channel_x)
        Spatial_weight = self.sigmoid(Spatial_x)

        out1 = x * Channel_weight  
        out2 = x * Spatial_weight  
        out = th.cat([out1, out2], 1)
        out = self.dimension(out) + residual
        return out

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size() 
    channels_per_group = num_channels // groups 
   
    x = x.view(batchsize, groups,
               channels_per_group, height, width)  
    x = th.transpose(x, 1, 2).contiguous()  
    
    x = x.view(batchsize, -1, height, width) 
    return x



class MSDC(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6', dw_parallel=True):
        super(MSDC, self).__init__()

        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.activation = activation
        self.dw_parallel = dw_parallel
        self.dilation_factors = [1, 2, 3]

        
        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size, stride, dilation * (kernel_size // 2),
                          groups=self.in_channels, bias=False, dilation=dilation),
                nn.BatchNorm2d(self.in_channels),
                act_layer(self.activation, inplace=True)
            )
            for kernel_size, dilation in zip(self.kernel_sizes, self.dilation_factors)
        ])

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)  
            outputs.append(dw_out) 
            if self.dw_parallel == False:
                x = x + dw_out
        
        return outputs


class MSCB(nn.Module):

    def __init__(self, in_channels, out_channels, stride, kernel_sizes=[1, 3, 5], expansion_factor=2, dw_parallel=True,
                 add=True, activation='relu6'):
        super(MSCB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.expansion_factor = expansion_factor
        self.dw_parallel = dw_parallel
        self.add = add
        self.activation = activation
        self.n_scales = len(self.kernel_sizes)
        # check stride value
        assert self.stride in [1, 2]
        
        self.use_skip_connection = True if self.stride == 1 else False

        
        self.ex_channels = int(self.in_channels * self.expansion_factor)
        self.pconv1 = nn.Sequential(
            
            nn.Conv2d(self.in_channels, self.ex_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_channels),
            act_layer(self.activation, inplace=True)
        )
        self.msdc = MSDC(self.ex_channels, self.kernel_sizes, self.stride, self.activation,
                         dw_parallel=self.dw_parallel)
        if self.add == True:
            self.combined_channels = self.ex_channels * 1
        else:
            self.combined_channels = self.ex_channels * self.n_scales
        self.pconv2 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(self.combined_channels, self.out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.out_channels),
        )
        if self.use_skip_connection and (self.in_channels != self.out_channels):
            self.conv1x1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False)
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        pout1 = self.pconv1(x) # (B,C,H,W)-->(B,2C,H,W)
        msdc_outs = self.msdc(pout1) 

        
        if self.add == True:
            dout = 0
            for dwout in msdc_outs:
                dout = dout + dwout  
        else:
            dout = th.cat(msdc_outs, dim=1) 

        
        dout = channel_shuffle(dout, gcd(self.combined_channels, self.out_channels))

        
        out = self.pconv2(dout)

        
        if self.use_skip_connection:
            
            if self.in_channels != self.out_channels:
                x = self.conv1x1(x)
            return x + out
        else:
            return out
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = th.exp(
            -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
        if dim % 2:
            embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = th.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = th.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings
# dummy replace
def convert_module_to_f16(x):
    pass


def convert_module_to_f32(x):
    pass


class PatchEmbed(nn.Module):
 
  def __init__(self, input_w, input_h, patch_w, patch_h, in_chans, embed_dim):
    super().__init__()
    img_size = (input_w, input_h)
    patch_size = (patch_w, patch_h)
    num_patches = (input_w // patch_w) * (input_h // patch_h)
    self.img_size = img_size
    self.patch_size = patch_size
    self.num_patches = num_patches
    self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
 
  def forward(self, x):
    B, C, H, W = x.shape
    assert H == self.img_size[0] and W == self.img_size[1], \
      f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
    x = self.proj(x).flatten(2).transpose(1, 2)
    return x  



class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, mlp_ratio=4.0, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
        
    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_h, patch_w, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_h * patch_w * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
    

class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size,
        patch_size,
        in_channels,
        out_channels,
        hidden_size,
        depth,
        num_heads,
        num_head_channels,
        num_classes=None,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        learn_sigma=True,
        use_spatial_transformer=False,
        transformer_depth=1,
        context_dim=None,
        lib_name='ldm'
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'
        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)
                
        self.input_size = input_size
        self.patch_size = patch_size
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.depth = depth
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.x_embedder = PatchEmbed(input_size[0], input_size[1], patch_size[0], patch_size[1], in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        if num_classes is not None:
            self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
            
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(th.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.blocks = nn.ModuleList([
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        if context_dim is not None:
            self.spatial_blocks = nn.ModuleList([
                BasicTransformerBlock(hidden_size, num_heads, num_head_channels, context_dim=context_dim)
                for _ in range(transformer_depth)
            ])
        self.cspab = CBAMBlock(channel=in_channels, reduction=2, kernel_size=7, HW=input_size[0]*input_size[1])
        self.mscb = MSCB(in_channels=in_channels, out_channels=in_channels, stride=1, kernel_sizes=[1,3,5], expansion_factor=2, dw_parallel=True, add=True, activation='relu6')
        self.final_layer = FinalLayer(hidden_size, patch_size[0], patch_size[1], self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                th.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(th.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        if self.num_classes is not None:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
            
        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p_h, p_w = self.x_embedder.patch_size
        h = self.input_size[0] // p_h
        w = self.input_size[1] // p_w
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p_h, p_w, c))
        x = th.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p_h, w * p_w))
        return imgs

    def forward(self, x, t=None, context=None, y=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        assert (y is not None) == (
                self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        x = self.cspab(x)
        x = self.mscb(x)
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        
        if self.num_classes is not None:
            y = self.y_embedder(y, self.training)  # (N, D)
        else:
            y = 0   
        c = t + y                                # (N, D)
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        if context is not None:
            for block in self.spatial_blocks:
                x = block(x, context)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = th.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = th.cat([half_eps, half_eps], dim=0)
        return th.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
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
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
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
