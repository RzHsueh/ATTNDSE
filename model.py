import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import Tensor, nn

from config import args

PE = args.position_encode

class PositionalEncoding(nn.Module):
    def __init__(
            self, 
            hidden_dim, 
            dropout, 
            max_length: int = 2000,
            drop_layer: nn.Module = nn.Dropout
    ) -> None:
        super().__init__()

        self.dropout = drop_layer(p=dropout)
        pe = torch.zeros(max_length, hidden_dim)
        # pe-[max_lenth, hidden_dim]

        position = torch.arange(0., max_length).unsqueeze(1)
        # positon-[max_length, 1]

        div_term = torch.exp(torch.arange(0., hidden_dim, 2)* -(math.log(10000.0 / hidden_dim)))
        #div_term-[hidden_dim/2]

        pe[:, 0::2] = torch.sin(position * div_term) # [max_length, hidden_dim/2]
        pe[:, 1::2] = torch.cos(position * div_term) # [max_length, hidden_dim/2]
        pe = pe.unsqueeze(0)
        # pe-[1, max_length, hidden_dim]
        self.register_buffer('pe', pe)

    def forward(self, x) -> Tensor:
        # x~[batch_size, sequence_length, hidden_dim]
        # pe-[1, max_length, hidden_dim]->slice to [1, sequence_length, hidden_dim]->extent to [batch_size, sequence_length, hidden_dim]
        x = x + self.pe[:, :x.size(1)].detach()
        # x-[batch_size, sequence_length, hidden_dim]
        return self.dropout(x)


class Mlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
    ) -> None:
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.fc2(x)
        return x
        

class Attention(nn.Module):

    def __init__(
        self,
        embeded_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()

        self.embeded_dim = embeded_dim
        self.internal_dim = embeded_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embeded_dim."

        self.q_proj = nn.Linear(self.embeded_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.embeded_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.embeded_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, self.embeded_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn_probs = torch.softmax(attn, dim=-1)
        ########################################
        # 初始化一个全零矩阵来存储每个头的注意力权重矩阵的总和
        total_attention_weights = torch.zeros_like(attn_probs[0, 0])
        if not self.training:      
            # 遍历每个头并将其注意力权重矩阵相加
            for head_idx in range(attn_probs.size(1)):
                total_attention_weights += attn_probs[0, head_idx]
            # print("Total attention weights after summing across heads:")
            # print(total_attention_weights)
        ########################################
        # Get output
        out = attn_probs @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out, total_attention_weights


class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads)
        

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, attn_weight = self.attn(x, x, x)
        x = x + self.norm1(x)
        x = x + self.norm2(self.mlp(x))

        return x, attn_weight


class WindowAttention(nn.Module):
    def __init__(
        self,
        embeded_dim: int,
        num_heads: int,
        window_size: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()

        self.embeded_dim = embeded_dim
        self.internal_dim = embeded_dim // downsample_rate
        self.num_heads = num_heads
        self.window_size = window_size
        assert self.internal_dim % num_heads == 0, "num_heads must divide embeded_dim."

        self.q_proj = nn.Linear(self.embeded_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.embeded_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.embeded_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, self.embeded_dim)

    def _window_partition(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        x = x.view(B, N // self.window_size, self.window_size, C)
        return x

    def _window_reverse(self, windows: Tensor, N: int) -> Tensor:
        B, W, window_size, C = windows.shape
        x = windows.view(B, W * window_size, C)
        return x

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        B, W, window_size, C = x.shape
        x = x.reshape(B, W, window_size, num_heads, C // num_heads)
        return x.permute(0, 1, 3, 2, 4)  # B x W x N_heads x window_size x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        B, W, num_heads, window_size, C_per_head = x.shape
        x = x.permute(0, 1, 3, 2, 4).reshape(B, W * window_size, num_heads * C_per_head)
        return x

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        B, N, C = q.shape
        assert N % self.window_size == 0, "Sequence length {N} must be divisible by window size {self.window_size}."

        # Window partition
        x_windows = self._window_partition(q)  # B x num_windows x window_size x C

        # Input projections
        q = self.q_proj(x_windows)
        k = self.k_proj(x_windows)
        v = self.v_proj(x_windows)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, window_size, c_per_head = q.shape
        attn = q @ k.transpose(-2, -1)  # B x num_windows x N_heads x window_size x window_size
        attn = attn / math.sqrt(c_per_head)
        attn_probs = torch.softmax(attn, dim=-1)

        # Get output
        out = attn_probs @ v  # B x num_windows x N_heads x window_size x C_per_head
        out = self._recombine_heads(out)  # B x num_windows*window_size x C
        out = self.out_proj(out)  # B x N x C

        return out


class Window_block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.window_attn = WindowAttention(dim, num_heads, 13) # [2,2,11,4,7]
        

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.norm1(self.window_attn(x, x, x))
        x = x + self.norm2(self.mlp(x))

        return x


class Model(nn.Module):

    def __init__(self, depth, embed_dim, num_heads, dropout):
        super().__init__()

        self.index_map = torch.Tensor(range(26)).type(torch.int32) * 20
        self.embedding = nn.Embedding(26 * 20, embed_dim)
        if PE != "none":
            self.position_encode = PositionalEncoding(embed_dim, dropout)
        self.reg_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.window_block = nn.Sequential(*[
            Window_block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4
            )
            for i in range(depth - 1)])

        self.block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4
            )

        self.fc = nn.Linear(embed_dim, 1)


    def forward(self, x):
        # print(f"type of x is {type(x)}")
        # print(f"x.size is {x.size()}")
        x = x + self.index_map.to(x.device) # b n
        x = self.embedding(x) # b n c
        if PE != "none":
            x = self.position_encode(x)
        x = self.window_block(x)
        x = torch.cat([self.reg_token.expand(x.shape[0], -1, -1), x], dim=1)
        x, attn_weight = self.block(x)
        x = self.fc(x[:, 0, :])
        return x, attn_weight
        

if __name__ == '__main__':
    model = Model(depth=args.depth, embed_dim=64, num_heads=8, dropout=0.1)
    print(model)
