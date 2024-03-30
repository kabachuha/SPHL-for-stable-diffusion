import torch, os, math
from PIL import Image
import argparse
import numpy as np
from termcolor import colored
import json

from diffusers import StableDiffusionPipeline

config = argparse.ArgumentParser()
config.add_argument('--clean_folders', type=str, help='Path to folder A') # X
config.add_argument('--clean_and_poison_folders', type=str, help='Path to folder B') # X^*
config.add_argument('--save_cfg_path', type=str, help='Path for stats saving (.json)', default='latent_conds.json')
config.add_argument('--seed', type=int, help='Torch seed', default=6934)
config.add_argument('--model_name', type=str, help='Torch seed', default='runwayml/stable-diffusion-v1-5')
args = config.parse_args()

torch.manual_seed(args.seed)

# Load the SD Pipeline
pipe = StableDiffusionPipeline.from_pretrained(args.model_name)#, torch_dtype=torch.float16)
#pipe.unet.to('cpu')
pipe.to('cpu')
#pipe.vae.to("cuda")
#pipe.unet.to('cpu')

with torch.no_grad():

    def pil_to_latent(img, sd_pipe):
        img = img.resize((512, 512), Image.Resampling.LANCZOS)
        image = sd_pipe.image_processor.preprocess(img).to(sd_pipe.vae.device)#.half()
        image_latents = (
            sd_pipe.vae.encode(image).latent_dist.sample() * sd_pipe.vae.config.scaling_factor
        ).to('cpu')
        return image_latents

    def arg_to_list(arg):
        if arg.startswith('['):
            folders = arg.strip('[').strip(']').split(',')
        else:
            folders = [arg]
        return folders

    clean_folders = arg_to_list(args.clean_folders)
    clean_and_poison_folders = arg_to_list(args.clean_and_poison_folders)
    save_cfg_path = args.save_cfg_path

    stats = {}

    for clean_folder, clean_and_poison_folder in zip(clean_folders, clean_and_poison_folders):

        embeddings_clean = []
        embeddings_clean_plus_poisoned = []

        for t_f in os.listdir(clean_folder):
            img_path = os.path.join(clean_folder, t_f)
            embeddings_clean.append(pil_to_latent(Image.open(img_path).convert('RGB'), pipe))
        
        embeddings_clean_cat = torch.cat(embeddings_clean, dim=0)
        print(embeddings_clean_cat.shape)
        
        for t_f in os.listdir(clean_and_poison_folder):
            img_path = os.path.join(clean_and_poison_folder, t_f)
            embeddings_clean_plus_poisoned.append(pil_to_latent(Image.open(img_path).convert('RGB'), pipe))
        
        embeddings_clean_plus_poisoned_cat = torch.cat(embeddings_clean_plus_poisoned, dim=0)
        print(embeddings_clean_plus_poisoned_cat.shape)

        embeddings_clean_mean = torch.mean(embeddings_clean_cat, dim=0)
        embeddings_clean_plus_poisoned_mean = torch.mean(embeddings_clean_plus_poisoned_cat, dim=0)

        right_part = embeddings_clean_plus_poisoned_mean - embeddings_clean_mean

        diffs_tensor = []

        for emb in embeddings_clean_plus_poisoned:
            diff = emb.squeeze(0) - embeddings_clean_plus_poisoned_mean
            diffs_tensor.append((diff ** 3).unsqueeze(0))
        
        diffs_tensor_cat = torch.cat(diffs_tensor, dim=0)
        print(diffs_tensor_cat.shape)

        diffs_tensor_mean = torch.mean(diffs_tensor_cat, dim=0)

        print(diffs_tensor_mean.shape)
        print(right_part.shape)

        result = torch.dot(torch.flatten(diffs_tensor_mean), torch.flatten(right_part))

        print(result)
        print(result > 0)

        stats[clean_and_poison_folder] = (result > 0).item()

        print("")

    # Save to json file if specified
    if save_cfg_path is not None and len(save_cfg_path) > 0:
        with open(f"{save_cfg_path}", 'w', encoding='utf-8') as f:
            f.write(json.dumps(stats, indent=4, ensure_ascii=False))
