# Essential Pytorch

## Summary

## Pre-requisites
You're expected to already know what PyTorch is and it's purpose, and what a tensor is.

## How to read this
Don’t try to learn these standalone. Don’t, after reading this, think I need to memorize these now. That’s really inefficient. Only with examples. Code katas
These are only to drive a couple of points home
To show you that learning  PyTorch isn’t crazy
Be aware as these come up. Drive these into your awareness
To simplify your learning, so you know what to pay extra close attention too, instead of thinking you need to commit the entire lib to memory
There are other very important things to know, but you can adjust the emphasis of learning those items as you code more models and the need arises

Don't read this thinking, "oh I need to get this all down now." I just want to raise your awareness the elements to pay attention to.

## The 20% of PyTorch you need to know to build most models
 
### Shapes

Let's start off with shapes. These are the shapes you will see repeatedly for each modality. I've listed all the common modalities, and for this post, we're doing to stick with image and audio because you can practice with those two modalities, and easily conquer the rest.


| Modality                | Raw Shape           | Model/Post-Embed Shape      | Notes                     |
| ----------------------- | ------------------- | --------------------------- | ------------------------- |
| **Image**               | (B, C, H, W)        | (B, N, D)                   | N = patches               |
| **Text**                | (B, P) or (B, P, D) | —                           | P = tokens                |
| **Video**               | (B, C, T, H, W)     | (B, N, D)                   | N = spatiotemporal tokens |
| **Audio (waveform)**    | (B, C, T)           | (B, L, D) (after tokenizer) | 1D analog of images       |
| **Audio (spectrogram)** | (B, F, T)           | (B, N, D)                   | Treated like an image     |


#### Images as image tensors (B, C, H, W)

An image is most commonly represented as (B, C, H, W), where B is the batch size, C is the number of channels, usually 3, and H and W are the height and width, respectively. You will need to understand image in this from when working with any convolution operators. The number of channels can increase or decrease. The height and with can also increase and decrease by either striding, upsamling, or downsampling

#### Image tokens (B, L, D)

You will often need to tokenize the image as the input and output to a transformer. The format of that image is (B, L, D), where L is the number of tokens and D is the number of dimensions of the tokens.

You will mostly likely create your tokens from image tensors, so you'll have to create some function $F: \mathbb{R}^{BxCxHxW} \to \mathbb{R}^{BxLxD}$, which will be the `forward` function of module you create using `nn.Module`