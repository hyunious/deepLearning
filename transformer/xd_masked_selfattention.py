# Self-Attention for transformer

import torch
import torch.nn as nn

"""
  Masked Self-Attention
"""
class XD_MaskedSelfAttention(nn.Module):
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
    score = torch.matmul(query, key.transpose(-2, -1))
    # score smoothing
    score = score / key.size(-1) ** 0.5

    # mask 생성 : x device 에 맞는 형태의 행렬 생성
    mask = torch.tril(torch.ones(score.size(-2), score.size(-1))).to(x.device)
    # mask == 0 인 곳을 softmax 에서 0 이 되도록 -inf 로 채움
    score = score.masked_fill(mask == 0, -float('-inf'))

    # score softmax
    attention_weights = torch.softmax(score, dim=-1)
    # output = attention_weights @ value
    weighted_value = torch.matmul(attention_weights, value)

    return weighted_value
    


"""
  Masked Multi-Head Attention
"""
class XD_MaskedMultiHeadAttention(nn.Module):
  def __init__(self, embed_dim, num_heads, bias=False):
    super().__init__()

    atten_dim = embed_dim // num_heads

    self.attentions = nn.ModuleList([XD_MaskedSelfAttention(embed_dim, atten_dim) for _ in range(num_heads)])
    self.fc = nn.Linear(embed_dim, embed_dim)

  def forward(self, x):
    # 각 헤드의 출력을 수집하여 최종 출력을 생성
    head_outputs = [attention(x) for attention in self.attentions]
    # 각 헤드의 출력을 연결하여 최종 출력을 생성
    cancatenated_head_outputs = torch.cat(head_outputs, dim=-1)
    # 최종 출력을 선형 변환
    output = self.fc(cancatenated_head_outputs)

    return output
  



"""
  Feed Forward for transformer decoder

  - 선형 변환 (embed_dim -> hidden_dim) -> 활성화 함수 -> 선형 변환 (hidden_dim -> embed_dim)
"""
class XD_FeedForward(nn.Module):
  def __init__(self, embed_dim, hidden_dim):
    super().__init__()

    self.feed_forward = nn.Sequential(
      nn.Linear(embed_dim, hidden_dim),
      nn.GELU(),
      nn.Linear(hidden_dim, embed_dim)
    )

  def forward(self, x):
    return self.feed_forward(x)




"""
  Transformer Block for decoder

  : Layer Normalization -> Masked Multi-Head Attention -> Add (Skip Connection) -> Layer Normalization -> Feed Forward -> Add (Skip Connection)
"""
class XD_TransformerDecoderBlock(nn.Module):
  def __init__(self, embed_dim, num_heads, bias=False):
    super().__init__()

    self.layer_norm1 = nn.LayerNorm(embed_dim)
    self.masked_multi_head_attention = XD_MaskedMultiHeadAttention(embed_dim, num_heads, bias)
    self.layer_norm2 = nn.LayerNorm(embed_dim)
    self.feed_forward = XD_FeedForward(embed_dim, 4*embed_dim)
    
  def forward(self, x):
    # Layer Normalization -> Masked Multi-Head Attention -> Add (Skip Connection)
    x = x + self.masked_multi_head_attention(self.layer_norm1(x))

    # Layer Normalization -> Feed Forward -> Add (Skip Connection)
    x = x + self.feed_forward(self.layer_norm2(x))

    return x


