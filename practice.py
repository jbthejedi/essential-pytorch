import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):

  def __init__(self, img_size=224, in_ch=3, embed_dim=512, patch_size=32, p_dropout=0.1):
    super().__init__()
    self.patch_embed = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
    self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))              # (1, 1, D), where D = embed_dim
    P = (img_size // patch_size) ** 2
    self.pos_embed = nn.Parameter(torch.zeros(1, P+1, embed_dim))
    self.dropout = nn.Dropout(p_dropout)

  def forward(self, x : torch.Tensor):
    """
    D = embed_dim
    """
    B, C, H, W = x.shape                                        # (B, C, H, W)
    x = self.patch_embed(x)                                     # (B, D, H/ps, W/ps), where ps = patch_size
    x = x.flatten(2).transpose(1, 2)                            # (B, P, D), where P = H/ps * W/ps
    self.cls_token
    x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1) # (B, P+1, D)
    x = x + self.pos_embed                                      # (B, P+1, D)
    return self.dropout(x)


def tokenize_image():
  """
  torch.rand, shape, flatten, transpose
  """
  DEVICE = torch.device('cpu')
  B, C, H, W = 8, 512, 4, 4
  temp = torch.ones(B, C, device=DEVICE)
  image = torch.rand(B, C, H, W, device=DEVICE)   # (B, C, H, W)
  print(f"image.shape {image.shape}")

  h = image.flatten(2)                            # (B, C, H*W)
  image_toks = h.transpose(1, 2)                  # (B, H*W, C)
  print(f"image_toks {image_toks}")


def patch_embedding():
  """
  torch.rand, shape, Conv2d, Parameter, expand, flatten, transpose
  """
  DEVICE = torch.device('cpu')
  B, C, H, W = 8, 3, 128, 128
  P, D = 77, 512
  image = torch.rand(B, C, H, W, device=DEVICE)
  txt_toks = torch.rand(B, P, D, device=DEVICE)
  print(f"image.shape {image.shape}")
  print(f"txt_toks.shape {txt_toks.shape}")

  model = PatchEmbedding(img_size=H, in_ch=C, embed_dim=512, patch_size=32)
  patch_emb = model(image)
  print(f"patch_emb.shape {patch_emb.shape}")


def main():
  # tokenize_image()
  # patch_embedding()
  tokenize_image()


if __name__ == '__main__':
  main()
