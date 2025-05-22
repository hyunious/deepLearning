# Self-Attention for transformer

import torch
import torch.nn as nn

"""
  Self-Attention
"""
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
    score = torch.matmul(query, key.transpose(-2, -1))
    # score smoothing
    score = score / key.size(-1) ** 0.5
    # score softmax
    attention_weights = torch.softmax(score, dim=-1)
    # output = attention_weights @ value
    weighted_value = torch.matmul(attention_weights, value)

    return weighted_value
    


"""
  Multi-Head Attention 
  - attention dimension 은 embed_dim / num_heads 로 계산합니다.
"""
class XD_MultiHeadAttention(nn.Module):
  def __init__(self, embed_dim, num_heads, bias=False):
    super().__init__()

    # 20 // 4 = 5
    atten_dim = embed_dim // num_heads

    self.attentions = nn.ModuleList([XD_SelfAttention(embed_dim, atten_dim) for _ in range(num_heads)])
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
  Feed Forward

  - 선형 변환 (embed_dim -> hidden_dim) -> 활성화 함수 -> 선형 변환 (hidden_dim -> embed_dim)
"""
class XD_FeedForward(nn.Module):
  def __init__(self, embed_dim, hidden_dim):
    super().__init__()

    self.feed_forward = nn.Sequential(
      nn.Linear(embed_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, embed_dim)
    )

  def forward(self, x):
    return self.feed_forward(x)



"""
  TransformerBlock 

  : Layer Normalization -> Multi-Head Attention -> Add (Skip Connection) -> Layer Normalization -> Feed Forward -> Add (Skip Connection)
"""
class XD_TransformerBlock(nn.Module):
  def __init__(self, embed_dim, num_heads, bias=False):
    super().__init__()

    self.layer_norm1 = nn.LayerNorm(embed_dim)
    self.mhead_atten = XD_MultiHeadAttention(embed_dim, num_heads)

    self.layer_norm2 = nn.LayerNorm(embed_dim)
    self.feed_forword = XD_FeedForward(embed_dim, 4*embed_dim)


  def forward(self, x):
    # Layer Normalization (with skip connection)
    x = x + self.mhead_atten(self.layer_norm1(x))
    # Feed Forward (with skip connection)
    x = x + self.feed_forword(self.layer_norm2(x))

    return x