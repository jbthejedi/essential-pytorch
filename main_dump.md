
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

[
(B, C, H, W) ;\longrightarrow; (B, C, H \cdot W).
]

If you already know the exact shape you want, `view`/`reshape` are more general. But `flatten` is perfect when you just want to collapse “whatever is left” into a single dimension without manually computing the product.

### `torch.transpose`

`transpose` is matrix transpose, but applied to arbitrary pairs of dimensions in a tensor.

You’ll often be working with tensors that are 3D or higher, and you’ll frequently want to:

* Leave the batch dimension untouched.
* Swap two specific inner dimensions so you can perform an operation on those matrices.

For example, in the code above:

```python
image_toks = h.transpose(1, 2)  # (B, C, H*W) -> (B, H*W, C)
```

We’re swapping the “channel” dimension with the “token” dimension so that tokens become the second dimension, which is the standard `(B, L, D)` layout used by transformers.

In things like multi-head attention, you’ll often treat batch and heads as “list-like” dimensions and the last one or two dimensions as the actual content you’re operating on. `transpose` is how you line those up into the shape that your `@` (matmul) or other ops expect.

```
```
