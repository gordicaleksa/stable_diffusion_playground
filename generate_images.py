# You'll have to run "huggingface-cli login" the first time so that you can access the model weights.

import enum
import os
import json

from diffusers import StableDiffusionPipeline
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import autocast


class ExecutionMode(enum.Enum):
    GENERATE_DIVERSE = 0,
    REPRODUCE = 1,
    INTERPOLATE = 2


def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """ helper function to spherically interpolate two arrays v1 v2 
    
    Taken from: https://gist.github.com/karpathy/00103b0037c5aaea32fe1da1af553355
    """

    if not isinstance(v0, np.ndarray):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2


def generate_name(output_dir_path, suffix='jpg'):
    prefix = str(len(os.listdir(output_dir_path))).zfill(6)
    return f'{prefix}.{suffix}'


def save_metadata(meta_dir, prompt, num_inference_steps, guidance_scale):
    data = {  # Feel free to add anything else you might need.
        'prompt': prompt,
        'num_steps': num_inference_steps,
        'scale': guidance_scale
    }
    with open(os.path.join(meta_dir, generate_name(meta_dir, suffix='json')), 'w') as f:
        json.dump(data, f)

# TODO: refactor the code.
def run(  # fire makes things much more concise than argparse! :))
        # --------------------------------------
        name='ai_epiphany',  # Name of the output directory.
        execution_mode=ExecutionMode.GENERATE_DIVERSE,  # Choose between diverse generation and interpolation.
        prompt="a painting of an ai robot having an epiphany moment",  # Unleash your inner neural network whisperer.
        num_inference_steps=50,  # More (e.g. 100, 200 etc) can create slightly better images.
        guidance_scale=7.5,  # Complete black magic. Usually somewhere between 3-10 is good - but experiment.
        num_imgs=5,  # How many images you want to generate in this run.
        seed=23,  # I love it more than 42.
        width=512,  # Make sure it's a multiple of 8.
        height=512,
        fp16=True,  # Set to True unless you have ~16 GBs of VRAM.
        src_latent_path=None,  # Set the latent of the 2 images you like (useful for INTERPOLATE mode)
        trg_latent_path=None,
        metadata_path=None,  # Used only in the REPRODUCE branch.
        # --------------------------------------
):
    assert torch.cuda.is_available(), "You need a GPU to run this script."
    assert height % 8 == 0 and width % 8 == 0, f"Width and height need to be a multiple of 8, got (w,h)=({width},{height})."
    device = "cuda"
    if seed:  # If you want to have consistent runs.
        torch.manual_seed(seed)

    # Initialize the output file structure.
    root_dir = os.path.join(os.getcwd(), 'output', name)
    imgs_dir = os.path.join(root_dir, "samples")
    latents_dir = os.path.join(root_dir, "latents")
    meta_dir = os.path.join(root_dir, "meta")
    os.makedirs(imgs_dir, exist_ok=True)
    os.makedirs(latents_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    # Hardcoded the recommended scheduler - feel free to play with it.
    lms = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")

    # Create diffusion pipeline object.
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16 if fp16 else None,
        revision="fp16" if fp16 else "main",
        scheduler=lms,
        use_auth_token=True
    ).to(device)

    if execution_mode == execution_mode.GENERATE_DIVERSE:
        for i in range(num_imgs):
            init_latent = torch.randn((1, pipe.unet.in_channels, height // 8, width // 8), device=device)

            with autocast(device):
                image = pipe(
                    prompt,
                    num_inference_steps=num_inference_steps,
                    latents=init_latent,
                    guidance_scale=guidance_scale
                )["sample"][0]

            # Make sure generation is reproducible.
            image.save(os.path.join(imgs_dir, generate_name(imgs_dir, suffix='jpg')))
            # TODO: is there some clever python mechanism that can enable me to log all input arg values?
            save_metadata(meta_dir, prompt, num_inference_steps, guidance_scale)
            np.save(os.path.join(latents_dir, generate_name(latents_dir, suffix='npy')), init_latent.cpu().numpy())

    elif execution_mode == execution_mode.INTERPOLATE:
        if src_latent_path and trg_latent_path:
            print('Loading existing source and target latents.')
            src_init = torch.from_numpy(np.load(src_latent_path)).to(device)
            trg_init = torch.from_numpy(np.load(trg_latent_path)).to(device)
        else:
            print('Generating random source and target latents.')
            src_init = torch.randn((1, pipe.unet.in_channels, height // 8, width // 8), device=device)
            trg_init = torch.randn((1, pipe.unet.in_channels, height // 8, width // 8), device=device)
        
        # Make sure generation is reproducible.
        save_metadata(meta_dir, prompt, num_inference_steps, guidance_scale)
        np.save(os.path.join(latents_dir, generate_name(latents_dir, suffix='npy')), src_init.cpu().numpy())
        np.save(os.path.join(latents_dir, generate_name(latents_dir, suffix='npy')), trg_init.cpu().numpy())

        for i, t in enumerate(np.concatenate([[0], np.linspace(0, 1, num_imgs)])):
            if i == 0:
                init_latent = trg_init  # Make sure you're happy with the target image before you waste too much time.
            else:
                init_latent = slerp(float(t), src_init, trg_init)

            with autocast(device):
                image = pipe(
                    prompt,
                    num_inference_steps=num_inference_steps,
                    latents=init_latent,
                    guidance_scale=guidance_scale
                )["sample"][0]

            image.save(os.path.join(imgs_dir, generate_name(imgs_dir, suffix='jpg')))

    elif execution_mode == execution_mode.REPRODUCE:
        assert src_latent_path, 'You need to provide the latent path if you wish to reproduce an image.'
        assert metadata_path, 'You need to provide the metadata path if you wish to reproduce an image.'
        with open(metadata_path) as metadata_file:
            metadata = json.load(metadata_file)
        init = torch.from_numpy(np.load(src_latent_path)).to(device)
        with autocast(device):
            image = pipe(
                **metadata,
                latents=init,
                output_type='npy',
                # as long as it's not pil it'll return numpy with the current imp of StableDiffusionPipeline
            )["sample"][0]

        plt.imshow((image * 255).astype(np.uint8));
        plt.show()
    else:
        print(f'Execution mode {execution_mode} not supported.')


if __name__ == '__main__':
    fire.Fire(run)