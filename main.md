# Essential Pytorch

## Summary

## The goal

The goal of this article is to reduce the intimidation of PyTorch — to get you to think, “Oh, wow, this is primarily what I need to know to code SOTA models? That’s actually not nearly as bad as I was thinking.”

## Pre-requisites

You're expected to already know what PyTorch is and its purpose. You should probably have already tried to code something in PyTorch (maybe quite a few things), so the content in this article is not completely alien. You want to really increase your skills, but the breadth of the PyTorch library is very intimidating.

Of course, anyone is welcome to read this.

## How to read this

This is not a full PyTorch course. I'm not going to explain from scratch what a 2D convolution is.

Instead, I'm going to highlight the small set of patterns and operations that I use constantly when building real models: reshaping, tokenizing, wiring modules together, sampling tensors, tracking devices, etc.

A few guidelines:

- Don’t try to memorize these in isolation.
- Learn them through examples and code katas.
- When you see these patterns in your own code or in someone else’s implementation, treat them as “pay extra attention here” zones.
- There are other important parts of PyTorch, but you can give them less emphasis until your code actually needs them.

I'm not going to go super in-depth into every essential. There are times where I'll say, “This is the gist. This will appear a lot. Know this.”

Yes, there are other really important and cool things to know. But the goal of this piece isn't to get you to write the most efficient, well-engineered, perfectly abstract code. It's for you to become proficient in the essentials you can use to build SOTA models that take inputs and return useful outputs.

Over the past year I’ve implemented transformers, VAEs, UNets, CLIP, and rectified flow models from scratch in PyTorch. In practice, I keep reaching for the same small set of tools. This article is about those tools.

I won't say “you will use this all the time,” because that’s the point of this article: you will, and should be, using everything in here *all the time*.

## Result

If you deeply understand everything in this post and practice it a bit, you’ll be able to:

Understand how data flows through common architectures (CNNs, transformers, UNets, VAEs) just by looking at the code.

Implement those architectures from scratch in PyTorch.

Debug 90% of your shape and device issues without wanting to quit.

## The 20% of PyTorch you need to know to build most models

In practice, I lean on a surprisingly small toolbox:

- **Shape manipulation**: `view`, `reshape`, `flatten`, `transpose`, `permute`, `unsqueeze`, `squeeze`
- **Tensor creation & sampling**: `torch.rand`, `torch.randn`, `torch.zeros`, `torch.ones`, `torch.linspace`
- **Core modules**: `nn.Linear`, `nn.Conv*`, `nn.Sequential`, and basic `nn.Module` patterns
- **Matrix ops & broadcasting**: the `@` operator (matmul), `bmm`, and how broadcasting works
- **Bookkeeping**: `device`, `.to(...)`, `register_buffer`, simple testing with dummy tensors and a debugger

You’ll see these again and again in almost every model, from simple CNNs to transformers and diffusion models.

### Shapes

Let's start off with shapes. These are the shapes you will see repeatedly for each modality. I've listed all the common modalities, and for this post, we're going to stick with image and audio because you can practice with those two modalities and easily conquer the rest.

| Modality                | Raw Shape                     | Model/Post-Embed Shape            | Notes                       |
| ----------------------- | ----------------------------- | --------------------------------- | --------------------------- |
| **Image**               | ($B$, $C$, $H$, $W$)          | ($B$, $N$, $D$)                   | $N$ = patches               |
| **Text**                | ($B$, $P$) or ($B$, $P$, $D$) | —                                 | $P$ = tokens                |
| **Video**               | ($B$, $C$, $T$, $H$, $W$)     | ($B$, $N$, $D$)                   | $N$ = spatiotemporal tokens |
| **Audio (waveform)**    | ($B$, $C$, $T$)               | ($B$, $L$, $D$) (after tokenizer) | 1D analog of images         |
| **Audio (spectrogram)** | ($B$, $F$, $T$)               | ($B$, $N$, $D$)                   | Treated like an image       |

#### Images as image tensors ($B$, $C$, $H$, $W$)

An image is most commonly represented as ($B$, $C$, $H$, $W$), where $B$ is the batch size, $C$ is the number of channels, usually $3$, and $H$ and $W$ are the height and width, respectively. You will need to understand images in this form when working with any convolution operators. The number of channels can increase or decrease. The height and width can also increase and decrease by either striding, upsampling, or downsampling.

#### Image tokens ($B$, $L$, $D$)

You will often need to tokenize the image as the input and output to a transformer. The format of that image is ($B$, $L$, $D$), where $L$ is the number of tokens and $D$ is the number of dimensions of the tokens.

You will mostly likely create your tokens from image tensors, so you'll have to create some function $F: \mathbb{R}^{B \times C \times H \times W} \to \mathbb{R}^{B \times L \times D}$, the `forward` function of a module you create using `nn.Module`.

Here is some code below that changes an image tensor into image tokens in one simple way. Don't worry too much about fully grokking the point of this just yet. I need to explain some other essentials.

```python
def tokenize_image():
  """
  torch.rand, shape, flatten, transpose
  """
  DEVICE = torch.device('cpu')
  B, C, H, W = 8, 512, 4, 4
  image = torch.rand(B, C, H, W, device=DEVICE)   # (B, C, H, W)

  h = image.flatten(2)                            # (B, C, H*W)
  image_toks = h.transpose(1, 2)                  # (B, H*W, C)
  print(f"image_toks {image_toks}")

def main():
  tokenize_image()

if __name__ == '__main__':
  main()
```

So let's cover the remaining essentials that appear in the above example in order of appearance.

### `torch.device`

This just lets PyTorch know if you are on the CPU or GPU. The thing to know here is to track carefully all places where `device` needs to be specified. Be mindful that all your tensors are on the same device. I like to create a constant at the top of the main Python module and pass it around where it's needed.

Also something useful to know is the following:

```python
def f(x: torch.Tensor):
    B, C, H, W = x.shape
    z = torch.rand(B, C, H, W, device=x.device)
```

You can set the device of a newly created tensor to the device of another tensor. You will use this often when the original `DEVICE` variable is not available in a separate module you've created.

### `torch.rand`, `torch.randn`, `torch.ones`, `torch.zeros`

* `torch.rand` takes in a shape and returns a tensor with each element sampled uniformly from $[0, 1]$.
* `torch.randn` is the same idea but samples from a standard normal distribution. I also often use `torch.rand` / `torch.randn` to test my modules.
* You will use the above two often with generative modeling as you need to sample timesteps or sample priors to feed to your generative model.
* `torch.ones` and `torch.zeros` return a tensor of the provided shape, filled with either ones or zeros, respectively.

### Testing your modules with dummy tensors

Get in the habit of testing your modules with dummy tensors. I typically create a `main` method in the Python module where my PyTorch module lives. I create a tensor using `torch.rand`, I create the PyTorch module in question, and I just pass the tensor through my module.

If you're not in a production setting, typically, if your shapes are correct and the tensor passes through without error, the module often needs no further testing (assuming your model theory checks out).

Also, use the debugger and step through all your modules line-by-line. Do this a lot.

`torch.summary` is a nice tool. But I've personally found that passing through tensors and using the debugger to step through code is the superior route if you want to gain a deep intuition and feel for the flow of tensors.

### Creating comments with the output shapes of tensors, inline

This works in conjunction with the above section. I like to create comments of the shapes of the outputs of each operation on the same line, exactly in the format below.

```python
image = torch.rand(B, C, H, W, device=DEVICE)   # (B, C, H, W)

h = image.flatten(2)                            # (B, C, H*W)
image_toks = h.transpose(1, 2)                  # (B, H*W, C)
```

I've found this to be essential for understanding complex tensor flows and not getting lost in the coding.

### `torch.flatten`

If you think of a tensor as nested lists, `torch.flatten` “de-nests” the tensor starting at a given dimension.

`image.flatten(2)` keeps the first two dimensions and flattens everything from index 2 onward:

$(B, C, H, W) \longrightarrow (B, C, H \cdot W)$.

If you already know the exact shape you want, `view`/`reshape` are more general. But `flatten` is perfect when you just want to collapse “whatever is left” into a single dimension without manually computing the product.

### `torch.transpose`

`transpose` is matrix transpose, but applied to arbitrary pairs of dimensions in a tensor.

You’ll often be working with tensors that are 3D or higher, and you’ll frequently want to:

* Leave the batch dimension untouched.
* Swap two specific inner dimensions so you can perform an operation on those matrices.

For example, in the code above:

```python
image = torch.rand(B, C, H, W, device=DEVICE)   # (B, C, H, W)
h = image.flatten(2)                            # (B, C, H*W)
image_toks = h.transpose(1, 2)                  # (B, H*W, C)

# I can now do h @ image_toks
h @ image_toks                                  # (B, C, C)
```

We’re swapping the “channel” dimension with the “token” dimension so that tokens become the second dimension, which is the standard `(B, L, D)` layout used by transformers.

Before moving on, now that I've explained the necessary essentials, you can take a look at `tokenize_image` again to see everyting I explained working together.

Let's take a look at the next example. Below is another way we can turn an image into image tokens.

```python
class PatchEmbedding(nn.Module):

  def __init__(self, img_size=224, in_ch=3, embed_dim=512, patch_size=32, p_dropout=0.1):
    super().__init__()
    self.patch_embed = nn.Conv2d(in_ch, embed_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size)
    self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))   # (1, 1, D), where D = embed_dim
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
    # self.cls_token.expand(B, -1, -1)                          # (B, 1, D)
    x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1) # (B, P+1, D)
    x = x + self.pos_embed                                      # (B, P+1, D) + (1, P+1, D) = (B, P+1, D)
    return self.dropout(x)

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
  patch_embedding()
```

<br>
Let's build on our essentials and discuss what you need to know for the above code.


### `nn.Module`

```pyton
class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)

```
`nn.Module` is PyTorch’s “layer” base class. Anything with parameters or sub-layers should subclass it.

In __init__, you define submodules and parameters as `self.something = ....` PyTorch automatically registers them so `model.parameters()` sees everything.

In `forward`, you take in tensors and wire them through those submodules. PyTorch builds the computation graph from the operations you perform there.

### `nn.Conv2d`
“Conv2d is still one of the core first-principles ops in modern ML.
I use it for:
- extracting local features
- turning images into patch tokens
- downsampling/upsampling feature maps
- cheaply mixing channels with 1×1 convolutions
- processing spectrograms and other 2D feature maps

Even in models that are ‘attention everywhere’, there’s almost always a convolutional backbone, stem, or VAE/UNet hiding underneath.

#### Convolution math rules
“You don’t need to memorize the full convolution output formula. In practice, I use two rules 99% of the time:

Same padding with odd kernels (stride = 1):
Use kernel_size in ${3, 5, 7}$ and
$padding = (kernel\_size - 1) // 2.$
This keeps H and W unchanged.

Downsampling with stride:
Keep image sizes divisible by your stride (e.g. 128, 256, 512 with stride 2 or 4).
With the same padding rule and stride n, the spatial size just becomes $H/n$ and $W/n$.

If you stay in that regime, you can design almost all the UNets, VAEs, and CNN backbones you’ll actually use, without ever writing down the full conv math.”

### `nn.Parameter`

`nn.Parameter` is simply a tensor that acts as parameters whose values are updated and tracked by the dependency graph. In the example above, it's used in two cases, but we'll look at the line
```self.pos_embed = nn.Parameter(torch.zeros(1, P+1, embed_dim))```.
So we create positional embeddings of shape $(1, P+1, embed\_dim)$, meaning there are P+1 positional embeddings and we have to share those across the batch. They are initalized to be zeros, and each training iteration, the shared weights are updated by aggregating the values across the batch.

The class token is a learned embedding shared across the batch. After passing through the transformer, that single token is often treated as a summary of the entire sequence (e.g., used for classification).

### Understanding broadcast operations
```python
def broadcasting():
  DEVICE = torch.device('cpu')
  x1 = torch.rand(8, 1, 10, device=DEVICE)
  x2 = torch.rand(1, 6, 10, device=DEVICE)
  x3 = torch.rand(8, 6, 1, device=DEVICE)
  x1 + x2 + x3 # (8, 6, 10)
```

How broadcasting happens:
- Compare dims from right to left.
- Dimensions are compatible if:
  - they’re equal, or
  - one of them is 1.
- If one is 1, it gets “stretched” to match the other for the purpose of the op.

So you can see in the `broadcast` function above to confirm how the rules check out. Now, refer back to `x = x + self.pos_embed ` in `PatchEmbedding::forward` and confirm why that's broadcasting. We can see that it's $(B, P+1, D) + (1, P+1, D) = (B, P+1, D)$.


### `expand`
The `expand` function simply creates a `view` of the data that has a different shape. We'll cover `view` in a bit, but for now, just now that `view` is a way of reindixing memory to manipulate the shape of the data with out duplicating it in memory. We can see that `self.cls_token.expand(B, -1, -1)` creates a new output shape as such $(1, 1, D) \to (B, 1, D)$. The `-1` values simply tell `expand` to keep the same value. If a value is other than `-1`, the tensor expands to the given number along that dimension.

### `cat`
This is simply concatenation. Know it. Get comfortable specifying which dimension you're concatenating along. Concatenation plus linear projection is a highly prolific technique.

## (For later)
#### Using `transpose` and thinking about tensors as part list and part content

In things like multi-head attention, you’ll often treat batch and heads as “list-like” dimensions and the last one or two dimensions as the actual content you’re operating on. `transpose` is how you line those up into the shape that your `@` (matmul) or other ops expect.


