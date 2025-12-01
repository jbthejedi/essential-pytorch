# Essential Pytorch

## Summary

## The goal

The goal of this article is to reduce the intimidation of PyTorch. To you to think "oh, wow this is primarily what I need to know to code SOTA models? That's actually not nearly as bad as I was thinking"

## Pre-requisites

You're expected to already know what PyTorch is and it's purpose. You should probably have already tried to code something in PyTorch, maybe quite a few things, so the content in this article is not completely alien, but you're looking to really increase your skills, but the breadth of the PyTorch lib is very intimidating. Of course, anyone is welcome to read this.

## How to read this

I will not teach you PyTorch in this video. I'm not teaching you what a 2D convolution operation is.
Don’t try to learn these standalone. Don’t, after reading this, think I need to memorize these now. That’s really inefficient. Only with examples. Code katas
These are only to drive a couple of points home
To show you that learning  PyTorch isn’t crazy
Be aware as these come up. Drive these into your awareness
To simplify your learning, so you know what to pay extra close attention too, instead of thinking you need to commit the entire lib to memory
There are other very important things to know, but you can adjust the emphasis of learning those items as you code more models and the need arises

Don't read this thinking, "oh I need to get this all down now." I just want to raise your awareness the elements to pay attention to.

I'm not going to go super in-depth in to some of the essentials. There are times will I'll say "this is gist. This is will appear a lot. Know this."

Yes, there are other really important and cool things to know. But the goal of this piece isn't to get you to write the most efficient well-engineered, abstract code. It's for you to become proficient in the essentials that can build SOTA models that take inputs and return useful outputs.

I'm writing this after going from knowing virtually 0 PyTorch to writing SOTA models after an intense year of training. I did not use an LLM to create the content of this piece. These is all hard-earned learnings from becoming a PyTorch Expert.

I won't say "you will use this all the time" because that's the point of this article. You will and should be usign everything in this article <i>all the time</i>.

## The 20% of PyTorch you need to know to build most models

### Shapes

Let's start off with shapes. These are the shapes you will see repeatedly for each modality. I've listed all the common modalities, and for this post, we're doing to stick with image and audio because you can practice with those two modalities, and easily conquer the rest.

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

You will mostly likely create your tokens from image tensors, so you'll have to create some function $F: \mathbb{R}^{B \times C \times H \times W} \to \mathbb{R}^{B \times L \times D}$, which will be the `forward` function of a module you create using `nn.Module`.

Here is some code below that changes and image tensor into an image token in two very similar, but slightly different ways. Don't worry to much about groking the point of this just yet. I need to explain some other essentials

```
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

### torch.device
This just let's pytorch know if you are on the cpu or gpu. The thing to know here is to just track carefully all items where device needs to be specified. Be mindful that all your tensors are on the same device. I like to create a constant at the top of the main Python module, and pass it around where it's needed. Also something useful to know is this the following

```
def f(x:torch.Tensor):
  B, C, H, W = x.shape
  z = torch.rand(B, C, H, W, device=x.device)
```

You can set the device of a newly created tensor to the device of another Tensor. You will use this often when the original device variable is not available in a separate module you've created

### torch.rand, torch.randn, torch.ones, torch.zeros
`torch.rand` takes in a shape and returns the desired tensor with each element being a value between $[0, 1]$ sampled uniformly.
`torch.randn` is the same as above but it returns a tensor with values between $[0, 1]$ sampled from a standard normal. I also often use `torch.rand` to test my modules.
You will use the above two often with generative modeling as you need to sample timesteps or sample priors to feed to your generative model.
`torch.ones` and `torch.zeros` return a tensor of the provided shape, filled with either ones or zeros, respectively.

### Testing your modules with dummy tensors

Get in the habit of testing your modules with dummy tensors. I typically create a main method in the PyThon module where my PyTorch module is. I create a tesnor using `torch.rand`, I create the PyTorch module in question, and I just pass the tensor through my module. If youre not in a production setting, typically, if your shapes are correct, and the tensor passes through without error, I've found the module needs no further testing. This is of course provided your model theory checks out.

Also, use the debugger and step through all your modules line-by-line. Do this a lot.

`torch.summary` is a nice tool. But I've personally found that passing through tensors and using the debugger to step through code to be the superior route if you want to gain a deep intuition and feel for the flow of tensors.

### Creating comments with the output shapes of tensors, inline

This works in conjunction with the above section. I like to create a comments of the shapes of the output of the operation of that line, exactly in the format below.

```
  image = torch.rand(B, C, H, W, device=DEVICE)   # (B, C, H, W)

  h = image.flatten(2)                            # (B, C, H*W)
  image_toks = h.transpose(1, 2)                  # (B, H*W, C)
```

I've found this to be essential for me to understand complex tensor flows and not get lost in the coding.

### torch.flatten
If you think of a tensor as nested lists, `torch.flatten` de-nests the tensors at the specified index and beyond. So `image.flatten(2)` is the function $F:(B, C, H, W) \to (B, C, H*W). Yes, you can just use `torch.view`, which I haven't covered yet, but `flatten` is nice if you don't have the correct shapes to pass to `view`.

### torch.transpose
Transpose is matrix transpose. Enough said. The only thing to know is that you'll often be working with matrices that are 3 dimension and beyond, and you should just get comfortable with tranposing only specific pairs of dimensions. You'll often tranpose the last two dimensions, and leave the batch dimension untouched, so that you can perform some operation on the matrices represented by the last two dimensions. Or, you'll want to transpose another dimension and move it closer to the batch dimension. I sort of think of the those as "list" dimensions. Take multi-head attention for example, we have a batch dimension and a head dimension. I think of those as lists containing the actual content we want to manipulate. So we `transpose` to move the "head" to the "batch" list and group them together, then we have the actual "contents" grouped together, and we can peform operaions on them like dot product.