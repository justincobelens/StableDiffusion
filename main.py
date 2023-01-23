import os
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

from prompt_engineering import art_styles

SDV5_MODEL_PATH = os.getenv('SDV5_MODEL_PATH')
SAVE_PATH = os.path.join(os.environ['USERPROFILE'], 'Desktop', 'SD_OUTPUT')

prompt = "dog on a bike"
negative_prompt = "ugly, unrealistic, bad contrast, bad lighting, poorly drawn, morphed, disfigured, weird, odd"

num_images_per_prompt = 2

"""
higher steps usually generates higher quality images
"""

num_inference_steps = 50

"""
- height and width have to be multiples of 8
- going below 512 might result in lq
- going over 512 in both directions will repeat images
- always have one variable at 512
"""

height = 512
width = 720

guidance_scale = 10
# seed = 0

device_type = 'cuda'  # "cuda" for GPU, "cpu" for CPU
low_vram = True


def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + ' (' + str(counter) + ')' + extension
        counter += 1

    return path


def render_prompt():
    shorted_prompt = (prompt[:25] + '...') if len(prompt) > 25 else prompt
    shorted_prompt = shorted_prompt.replace(' ', '_')

    generation_path = os.path.join(SAVE_PATH, shorted_prompt.removesuffix('...'))

    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    if not os.path.exists(generation_path):
        os.mkdir(generation_path)

    # calling stable diffusion pipeline
    if device_type == 'cuda':
        if low_vram:
            pipe = StableDiffusionPipeline.from_pretrained(
                SDV5_MODEL_PATH,
                torch_dtype=torch.float16,
                revision='fp16'
            )
        else:
            pipe = StableDiffusionPipeline.from_pretrained(SDV5_MODEL_PATH)

        pipe.safety_checker = True
        pipe = pipe.to('cuda')

        if low_vram:
            pipe.enable_attention_slicing()
    elif device_type == 'cpu':
        pipe = StableDiffusionPipeline.from_pretrained(SDV5_MODEL_PATH)

    else:
        print("Invalid Device Type Selected, use 'cpu' or 'cuda' only")
        return

    for style_type, style_prompt in art_styles.items():
        prompt_stylized = f"{prompt}, {style_prompt}"

        print(f"Full Prompt:\n{prompt_stylized}\n")
        print(f"Character in prompt: {len(prompt_stylized)}, limit: 200")

        for i in range(num_images_per_prompt):
            if device_type == 'cuda':
                with autocast('cuda'):
                    image = pipe(
                        prompt_stylized,
                        negative_prompt=negative_prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        height=height,
                        width=width
                    ).images[0]
            else:
                image = pipe(prompt_stylized).images[0]

            image_path = uniquify(
                os.path.join(SAVE_PATH, generation_path, style_type + " - " + shorted_prompt) + '.png')
            print(image_path)

            image.save(image_path)
    print("\nRENDER FINISHED\n")


render_prompt()