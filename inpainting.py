import os
import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

SDV5_INPAINT_PATH = os.getenv('SDV5_INPAINT_PATH')

INPUT_DIR = os.path.join(os.environ['USERPROFILE'], 'Desktop', 'SD_OUTPUT')
MASK_DIR = os.path.join(os.environ['USERPROFILE'], 'Desktop', 'SD_OUTPUT')

OUTPUT_DIR = os.path.join(os.environ['USERPROFILE'], 'Desktop', 'SD_OUTPUT')
SAVE_PATH = os.path.join(os.environ['USERPROFILE'], 'Desktop', 'SD_OUTPUT')


def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + ' (' + str(counter) + ')' + extension
        counter += 1

    return path


def create_pipe(safety_checker=True):
    # checking for image
    if len(os.listdir(INPUT_DIR)) != 2:
        print('too little or no file')
        return

    image = os.path.join(INPUT_DIR, os.listdir(INPUT_DIR)[0])
    if os.path.isfile(image):
        image = Image.open(image)
    else:
        print('no image file')
        return

    # checking for image with mask
    if len(os.listdir(MASK_DIR)) != 2:
        print('too little or no file in mask')
        return

    mask_image = os.path.join(MASK_DIR, os.listdir(MASK_DIR)[0])
    if os.path.isfile(mask_image):
        mask_image = Image.open(mask_image)
    else:
        print('no image file in mask')
        return

    # creating pipeline
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        SDV5_INPAINT_PATH,
        revision="fp16",
        torch_dtype=torch.float16,
    )

    # lowering memory usage
    pipe.safety_checker = safety_checker
    pipe = pipe.to('cuda')
    pipe.enable_attention_slicing()

    return image, mask_image, pipe


def inpaint_2(image, mask_image, pipe,
              prompt,
              negative_prompt,
              width=512,
              height=512,
              num_inference_steps=50,
              guidance_scale=7.5,
              seed=None):
    # generating and setting seed
    generator = torch.Generator("cuda")
    if seed is None:
        seed = generator.seed()

    generator = generator.manual_seed(seed)

    print(f"seed:                {seed}")

    # generating image
    image = pipe(
        prompt=prompt,
        width=width,
        height=height,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask_image,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator
    ).images[0]

    return image, seed


def save(prompt,
         image,
         seed,
         image_number,
         number="1)"):
    # create filename
    # shorted_prompt = (prompt[:25] + '...') if len(prompt) > 25 else prompt
    # shorted_prompt = shorted_prompt.replace(' ', '_')
    generation_path = os.path.join(OUTPUT_DIR, number)

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    if not os.path.exists(generation_path):
        os.mkdir(generation_path)

    # save file
    image_path = uniquify(
        os.path.join(OUTPUT_DIR, generation_path, "photo_" + str(image_number) + " - " + str(seed)) + '.png')

    image.save(image_path)
    print(image_path)
    print('\n')


def main():
    # -------------------- #

    # prompt
    prompt = "dog on a bike"
    negative_prompt = ""

    # size images
    width = 512
    height = 704

    # amount of diffusion steps
    num_inference_steps = 50

    # prompt strictness
    guidance_scale = 7.5

    # amount of images generated
    images = 1

    # set seed
    seed = None

    # safety checker
    safety_checker = True

    # -------------------- #

    number = f"{len(os.listdir(OUTPUT_DIR)) + 1}) Q{num_inference_steps} - G{guidance_scale}"

    print(f"num_inference_steps: {num_inference_steps}\n"
          f"guidance_scale:      {guidance_scale}\n")

    image, mask_image, pipe = create_pipe(safety_checker=safety_checker)

    for i in range(1, images + 1):
        image_2, seed_2 = inpaint_2(image, mask_image, pipe,
                                    prompt,
                                    negative_prompt,
                                    width=width,
                                    height=height,
                                    num_inference_steps=num_inference_steps,
                                    guidance_scale=guidance_scale,
                                    seed=seed)

        save(prompt,
             image_2,
             seed_2,
             i,
             number)


if __name__ == "__main__":
    main()
