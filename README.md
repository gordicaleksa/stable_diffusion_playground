## Stable Diffusion Playground | 💻 + 🎨 = ❤️

The main benefit of using this code: reproducibility! You'll know exactly how you created all of your images!

Instructions and much more coming soon! 

Until then here are some images generated using the prompt:<br/>
"a painting of an ai robot having an epiphany moment"
<p align="center">
<img src="imgs/frame26.jpg" width="400px">
<img src="imgs/frame84.jpg" width="400px">

<img src="imgs/frame15.jpg" width="400px">
<img src="imgs/frame151.jpg" width="400px">
</p>
    
## Setup

Follow the next steps to run this code:

1. `git clone https://github.com/gordicaleksa/stable_diffusion_playground`
2. Open Anaconda console and navigate into project directory `cd path_to_repo`
3. Run `conda env create` from project directory (this will create a brand new conda environment).
4. Run `activate sd_playground` (for running scripts from your console or setup the interpreter in your IDE)
5. Run `huggingface-cli login` before the first time you try to use it to access model weights.

That's it! It should work out-of-the-box executing environment.yml file which deals with dependencies. <br/>

**Important note:** you have to locally patch the `pipeline_stable_diffusion.py` file from the `diffusers 0.2.4` lib
using the code from the [main](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py) branch. The changes I rely (having `latents` as an argument) on still haven't propagated to the pip package.

## Hardware requirements

You need a GPU that has at least 8 GBs of VRAM to run this at 512x512, fp16.
TODO: Describe this in more detail.

## TODOs

* Potentially try and integrate the information needed to reproduce the results into image metadata or via steganography.

## Learning material

Here is a video walk-through of this repo: <TODO: add the video link once uploaded> (the commit I used in the video is [this one](https://github.com/gordicaleksa/stable_diffusion_playground/commit/aa19a6e58f3a545c364017230df38ef42fac4307))

And here is a deep dive of the stable diffusion codebase: <TODO: add the video link once uploaded>

## Acknowledgements

Took inspiration from [Karpathy's gist](https://gist.github.com/karpathy/00103b0037c5aaea32fe1da1af553355).

## Licence

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/gordicaleksa/stable_diffusion_playground/blob/master/LICENCE)