import os
from torch import autocast
from diffusers import StableDiffusionPipeline

SDV5_MODEL_PATH = os.getenv('SDV5_MODEL_PATH')
SAVE_PATH = os.path.join(os.environ['USERPROFILE'], 'Desktop', 'SD_OUTPUT')


if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)

def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = f"{filename} ({counter}){extension}"
        counter += 1

    return path


prompt = "A dog riding a bike"

print(f"Character in promt: {len(prompt)}, limit: 200")

pipe = StableDiffusionPipeline.from_pretrained(SDV5_MODEL_PATH)
pipe = pipe.to('cuda')

with autocast('cuda'):
    image = pipe(prompt).images[0]

image_path = uniquify(os.path.join(SAVE_PATH, (prompt[:25] + '...') if len(prompt) > 25 else prompt) + '.png')
print(image_path)

image.save(image_path)

