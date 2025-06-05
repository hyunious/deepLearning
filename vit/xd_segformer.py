# paper review : 
# https://dlgari33.tistory.com/21
# https://lcyking.tistory.com/entry/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-SegFormer-Simple-and-Efficient-Design-for-SemanticSegmentation-with-Transformers

# code reference :
# https://github.com/lucidrains/segformer-pytorch
# https://github.com/jiaowoguanren0615/SegFormer-Pytorch

# einops : Tensor shape 을 가독성있게 (직관적으로) 변환하는 라이브러리
# https://github.com/arogozhnikov/einops
# https://eumgill98.tistory.com/109


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


class XD_DsConv2d(nn.Module):
  """
    Depth-wise Separable Convolution : 기존 컨볼루션 연산을 두 단계로 나누어 연산량을 줄이는 기법    

    1. 커널을 channel 방향으로 분리하여 각 커널에 대해 별도의 연산을 수행
    2. 각 커널의 연산 결과를 합치고, 최종적으로 1x1 컨볼루션을 수행하여 채널 수를 조절

  """
  def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, bias=True):
    super().__init__()

    self.net = nn.Sequential(
        # depth-wise convolution
        nn.Conv2d(in_channels, in_channels, kernel_size = kernel_size, padding = padding, groups = in_channels, stride = stride, bias = bias),        
        # point-wise convolution
        nn.Conv2d(in_channels, out_channels, kernel_size = 1, bias = bias)
    )

  def forward(self, x):
      return self.net(x)




class XD_EfficientAttention(nn.Module):
    """
      XD_EfficientAttention : 공간 효율적인 다중 헤드 어텐션 블록을 구현합니다.
    """
    def __init__(self, *, embed_dim, num_heads, reduction_ratio=2):
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
        self.to_q = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias = False)
        
        # 공간 해상도를 downsampling 하는 역할을 합니다.
        # reduction_ratio가 2 : 공간 해상도가 H x W -> H/2 x W/2 로 계산량은 1/4로 줄어듭니다.
        self.to_kv = nn.Conv2d(embed_dim, embed_dim * 2, kernel_size=reduction_ratio, stride = reduction_ratio, bias = False)
        self.to_out = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias = False)


    def forward(self, x):
        # input x (B, C, H, W) : B = batch size, C = channel (embed_dim), H = height, W = width
        b, c, h, w = x.shape
        # multi-head attention 수
        heads = self.num_heads

        # query (B, C, H, W), key (B, C, H//r, W//r), value (B, C, H//r, W//r) 계산
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = 1)) # chunk 함수는 주어진 차원에서 주어진 수로 나누어 주는 함수

        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = heads), (q, k, v))
        """
          b: batch size : 배치 크기
          h: number of heads : 헤드 수
          (x y): sequence length : 시퀀스 길이
          d: embed_dim/num_heads : 임베딩 차원/헤드 수 = 각 헤드별 임베딩 차원

          # Query: (B, C, H, W) → (B, heads, H*W, C//heads)
          q = rearrange(q, 'b (h d) x y -> b h (x y) d', h=heads)

          # Key, Value: (B, C, H//r, W//r) → (B, heads, (H//r)*(W//r), C//heads)
          k = rearrange(k, 'b (h d) x y -> b h (x y) d', h=heads)
          v = rearrange(v, 'b (h d) x y -> b h (x y) d', h=heads)
        """
    
        # 쿼리와 키의 유사도 (similarity score) 계산 : Q × K^T
        similarity_score = einsum('b i d, b j d -> b i j', q, k) * self.scale
        # attention weights = similarity score 의 softmax 적용
        attention_weights = similarity_score.softmax(dim = -1)

        # output = attention weights * value
        out = einsum('b i j, b j d -> b i d', attention_weights, v)
        # (B, C, H, W) : B = batch size, C = channel (embed_dim), H = height, W = width
        out = rearrange(out, '(b h) (x y) c -> b (h c) x y', h = heads, x = h, y = w)

        return self.to_out(out)




class XD_MixFFN(nn.Module):
  """
    Mix-FFN : FFN 중간에 3 * 3 convolution (with zero padding=1) 을 추가하여 부족한 지역 정보를 보환해줌 
              파라미터 수를 줄이기 위해 Depth-wise Separable Convolution 을 사용한다.

    - 선형 변환 (embed_dim -> hidden_dim) -> 3 * 3 컨볼루션 -> 활성화 함수 -> 선형 변환 (hidden_dim -> embed_dim)
  """
  def __init__(self, *, embed_dim, expansion_factor):
    super().__init__()
    hidden_dim = embed_dim * expansion_factor

    self.net = nn.Sequential(
        nn.Conv2d(embed_dim, hidden_dim, 1),
        XD_DsConv2d(hidden_dim, hidden_dim, 3, padding = 1),
        nn.GELU(),
        nn.Conv2d(hidden_dim, embed_dim, 1)
    )

  def forward(self, x):
      return self.net(x)
  


class XD_LayerNorm(nn.Module):
    """
      Layer Normalization : 각 샘플의 특성을 정규화하여 학습 안정성을 높임.
    """
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b



class XD_PreNorm(nn.Module):
  """
    Pre-Norm : Layer Normalization 을 먼저 수행하고, 그 후 입력을 처리.
  """
  def __init__(self, dim, fn):
    super().__init__()
    self.norm = nn.LayerNorm(dim)
    self.fn = fn

  def forward(self, x):
    return self.fn(self.norm(x))
  


class XD_MiT(nn.Module):
  """
    MiT : Mix Transformer Encoder - (Efficient Self-Attention + Mix-FFN + Overlapping Patch Embedding) Blocks

    Hierarchical Encoder 로 총 4단계 (1/4 -> 1/8 -> 1/16 -> 1/32 해상도) 의 계층적 구조를 가지며, 이를 통해 다중 스케일 정보를 학습한다.
  """
  def __init__(self, *, in_channels, stage_output_channels, num_heads, ff_expansion_factor = 4, ea_reduction_ratio = 2, num_layers = 4):
    """
      1. in_channels : 입력 채널 수
      2. stage_output_channels : stage 별 출력 채널 수 : []
      3. num_heads : multi-head attention 헤드 수
      4. ff_expansion_factor : MixFFN 확장 인자
      5. ea_reduction_ratio : Efficient Attention 감소 비율
      6. num_layers : Transformer Block 수
    """
    super().__init__()

    # 각 stage 에서 사용할 kernel, stride, padding size
    stage_kernel_stride_padding = ((7, 4, 3), (3, 2, 1), (3, 2, 1), (3, 2, 1))

    # 각 stage 에서 사용할 (입력 채널 수, 출력 채널 수) 튜플 리스트
    # in_channels=3, stage_output_channels=[64, 128, 320, 512]  >>> stage_dim_pairs = [(3, 64), (64, 128), (128, 320), (320, 512)]
    stage_dims = (in_channels, *stage_output_channels)
    stage_dim_pairs = list(zip(stage_dims[:-1], stage_dims[1:]))

    """
      Stages : Transformer Blocks in SegFormer Encoder
    """
    self.stages = nn.ModuleList([])

    # each stage : each transformer block 을 구성한다.
    for (dim_in, dim_out), (kernel, stride, padding) in zip(stage_dim_pairs, stage_kernel_stride_padding):
      #  get overlap patches : 컨볼루션 연산 시 겹치는 부분을 계산하여 출력 크기를 조절
      #  kernel_size=7, stride=4, padding=3  >>> 출력 크기 = (H-7+2*3)/4 + 1 = (H-1)/4 + 1
      #  kernel_size=3, stride=2, padding=1  >>> 출력 크기 = (H-3+2*1)/2 + 1 = (H-1)/2 + 1
      #  kernel_size=3, stride=2, padding=1  >>> 출력 크기 = (H-3+2*1)/2 + 1 = (H-1)/2 + 1
      #  kernel_size=3, stride=2, padding=1  >>> 출력 크기 = (H-3+2*1)/2 + 1 = (H-1)/2 + 1

      """
        Overlap Patch Embedding : Unfold + Conv2D
      """
      # nn.Unfold : input image를 overlapped patch 들로 나눔 : patches ≈ (H // stride) × (W // stride)
      # input : (B, C, H, W) 인 경우, output : (B, C * kernel ** 2, patches)
      get_overlap_patches = nn.Unfold(kernel, stride=stride, padding=padding)      

      # nn.Conv2d: patches 들을 projection하여 다음 단계 차원으로 맞춤 : dim_out = embed_dim
      overlap_patch_embedding = nn.Conv2d(dim_in * kernel ** 2, dim_out, 1)


      """
        Attention Layers : (Efficient Attention + Mix-FFN) Layer
      """      
      attention_layers= nn.ModuleList([])      
      
      for _ in range(num_layers):
        attention_layers.append(nn.ModuleList([
          XD_PreNorm(dim_out, XD_EfficientAttention(embed_dim = dim_out, num_heads = num_heads, reduction_ratio = ea_reduction_ratio)),
          XD_PreNorm(dim_out, XD_MixFFN(embed_dim = dim_out, expansion_factor = ff_expansion_factor))
        ]))
      

      """
        Each Stage : Transformer Block 구성
      """
      self.stages.append(nn.ModuleList([
         get_overlap_patches,
         overlap_patch_embedding,
         attention_layers
      ]))


      """
        Overlap patches merging 부분은 없음.
      """



  def forward(self, x, return_stage_outputs = False):
    # input image shape : (B, C, H, W)
    h, w = x.shape[-2:]

    # decoder 에서 사용할 transformer block(stage)의 출력 결과 저장
    stage_outputs = []

    for (get_overlap_patches, overlap_patch_embedding, attention_layers) in self.stages:
      # 1. get overlap patches from input image
      x = get_overlap_patches(x)
      num_patches = x.shape[-1]      
      
      # (B, C * kernel ** 2, patches) >>  (B, C * kernel ** 2, h // ratio, w // ratio) : 2D 이미지 형태로 재구성
      # 한 변의 축소 비율 = sqrt(original_pixels / num_patches)
      ratio = int(sqrt((h * w) / num_patches))  
      x = rearrange(x, 'b c (h w) -> b c h w', h = h // ratio)

      # 2. overlap patches embedding
      x = overlap_patch_embedding(x)
      
      # 3. attention layers's skip conntection process
      for (attn, ff) in attention_layers:
        # Layer Normalization > Efficient Multi-Head Attention -> Add (Skip Connection)
        x = attn(x) + x
        # Layer Normalization > Feed Forward -> Add (Skip Connection)
        x = ff(x) + x

      stage_outputs.append(x)

    ret = x if not return_stage_outputs else stage_outputs
    return ret










     