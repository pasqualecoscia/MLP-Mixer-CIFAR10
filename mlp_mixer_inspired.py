import torch
import torch.nn as nn
import torch.nn.functional as F

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        return self.fn(self.norm(x)) + x


def FeedForward(input_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(128, input_dim),
        nn.Dropout(0.2)
)

def FeedForward_out(input_dim):
    return  nn.Sequential(
                nn.Linear(input_dim, 1024),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(512, 64),
                nn.Dropout(0.2)
        )

class MLPMixer_Inspired(nn.Module):
    def __init__(self, patch_size, output_dim, c, h, w, depth):
        super().__init__()
        
        self.patch_size = patch_size
        self.depth = depth

        self.sequential_row = nn.Sequential(
            *[nn.Sequential(
                PreNormResidual(c*w*patch_size, FeedForward(c*w*patch_size))
            ) for _ in range(depth)],
        )
        self.sequential_col = nn.Sequential(
            *[nn.Sequential(
                PreNormResidual(c*h*patch_size, FeedForward(c*h*patch_size))
            ) for _ in range(depth)]
        )
        self.sequential_ch = nn.Sequential(
            *[nn.Sequential(
                PreNormResidual(h*w, FeedForward(h*w))
            ) for _ in range(depth)]
        )

        self.output_row = FeedForward_out(h*w*c)
        self.output_col = FeedForward_out(h*w*c)
        self.output_ch = FeedForward_out(h*w*c)

        self.output_layer = nn.Linear(64 * 3, output_dim)

    def forward(self, images):
        b, c, h, w = images.size()

        assert (h % self.patch_size) == 0
        assert (w % self.patch_size) == 0
        
        split_col = torch.split(images, self.patch_size, dim=-1)
        split_col = torch.stack(split_col, dim=1)
        split_col = split_col.view(-1, w//self.patch_size, c * h * self.patch_size)

        split_row = torch.split(images, self.patch_size, dim=-2)
        split_row = torch.stack(split_row, dim=1)
        split_row = split_row.view(-1, h//self.patch_size, c * self.patch_size * w)

        split_ch = images.view(b, c, h*w)

        split_row = self.sequential_row(split_row)
        split_row = split_row.view(b, -1)
        split_row = self.output_row(split_row)
        
        split_col = self.sequential_col(split_col)
        split_col = split_col.view(b, -1)
        split_col = self.output_col(split_col)
        
        split_ch = self.sequential_ch(split_ch)
        split_ch = split_ch.view(b, -1)
        split_ch = self.output_ch(split_ch)

        output = torch.cat([split_col, split_row, split_ch], dim=-1)

        output = self.output_layer(output)

        return output
