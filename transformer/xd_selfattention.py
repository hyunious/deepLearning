# Self-Attention for transformer

import torch
import torch.nn as nn

class XD_SelfAttention(nn.Module):
    def __init__(self, embed_dim, atten_dim, bias=False):
        super().__init__()

        self.embed_dim = embed_dim
        self.atten_dim = atten_dim

        self.query = nn.Linear(embed_dim, atten_dim, bias=bias)
        self.key = nn.Linear(embed_dim, atten_dim, bias=bias)
        self.value = nn.Linear(embed_dim, atten_dim, bias=bias)

        
    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        # score = query @ key.T
        score = torch.matmul(query, key.T)
        # score smoothing
        score = score / key.size(-1) ** 0.5
        # score softmax
        attention_weights = torch.softmax(score, dim=-1)
        # output = attention_weights @ value
        weighted_value = torch.matmul(attention_weights, value)

        return weighted_value



        