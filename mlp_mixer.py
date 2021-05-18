import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops.layers.torch import Rearrange, Reduce

# Adapted from https://github.com/lucidrains/mlp-mixer-pytorch

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )

class MLPMixer(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, num_classes, expansion_factor = 4, dropout = 0.):
        super().__init__()
        
        assert (image_size % patch_size) == 0, 'image must be divisible by patch size'
        
        self.patch_size = patch_size
        self.depth = depth

        num_patches = (image_size // patch_size) ** 2
        
        self.patches_fc = nn.Linear((patch_size ** 2) * 3, dim)
        
        fc1 = FeedForward(num_patches, expansion_factor, dropout, partial(nn.Conv1d, kernel_size=1))
        fc2 = FeedForward(dim, expansion_factor, dropout, nn.Linear)
        
        self.pre_norm1 = PreNormResidual(dim, fc1)
        self.pre_norm2 = PreNormResidual(dim, fc2)
        
        self.norm = nn.LayerNorm(dim)
        self.output_layer = nn.Linear(dim, num_classes)

    def forward(self, images):
        b, c = images.size()[:2]
        h, w = images.size(2)//self.patch_size, images.size(3)//self.patch_size
        patches = images.view(b, h*w, self.patch_size*self.patch_size*c)
        patches = self.patches_fc(patches)
        for _ in range(self.depth):
            patches = self.pre_norm1(patches)
            patches = self.pre_norm2(patches)        
        patches = self.norm(patches)
        patches = torch.mean(patches, 1)
        patches = self.output_layer(patches)
        return patches