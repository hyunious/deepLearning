# paper review : 
# https://lcyking.tistory.com/entry/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-SegFormer-Simple-and-Efficient-Design-for-SemanticSegmentation-with-Transformers

# code reference :
# https://github.com/lucidrains/segformer-pytorch
# https://github.com/jiaowoguanren0615/SegFormer-Pytorch


import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange
from math import sqrt
from functools import partial

# helpers
def exists(val):
    return val is not None

def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth

# classes

class XD_EfficientAttention(nn.Module):
    def __init__(self, *, embed_dim, num_heads, reduction_ratio=2, bias=False):
        super().__init__()

        # attention score를 계산할 때 사용되는 scaling factor
        # embed_dim을 num_heads로 나눈 값의 제곱근의 역수를 계산
        # 이는 attention score가 너무 커지는 것을 방지하고 gradient vanishing 문제를 해결하는데 도움
        self.scale = (embed_dim // num_heads) ** -0.5
        self.num_heads = num_heads


        """
        kernel_size=1 은 1x1 컨볼루션을 의미하며, 이는 다음과 같은 특징이 있습니다:
          1. 공간적 특징을 추출하지 않고 채널 간의 관계만 학습합니다.
          2. 채널 수를 조절하는 역할을 합니다 (채널 믹싱).
          3. 계산 비용이 매우 적습니다.
          4. Fully Connected Layer와 동일한 연산을 수행합니다.
        """
        self.to_q = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias = bias)
        self.to_kv = nn.Conv2d(embed_dim, embed_dim * 2, kernel_size=reduction_ratio, stride = reduction_ratio, bias = bias)
        self.to_out = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias = bias)

        self.norm = nn.LayerNorm(embed_dim)


    def forward(self, x):
