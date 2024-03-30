import argparse, os, shutil
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch, os, math
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer
from pathlib import Path

# In the beginning we have a dataset of N folders
# with different categories: artist style, object
# (folder has the same name as the prompt)

# Types of trash: Least CLIP sim, biggest aesth score difference, random objects

config = argparse.ArgumentParser()
config.add_argument('--input_folder', type=str, help='Path to concepts folder')
config.add_argument('--output_folder', type=str, help='Path to dataset folder')
config.add_argument('--embeddings_cache_folder', type=str, help='Path to CLIP embedding cache folder', default='embeddings_cache')
config.add_argument('--poison_fractions', type=str, help="How much poison to add for each dataset (excluding 0.0=clean)", default="[0.15,0.30,0.45]")
config.add_argument('--poison_types', type=str, help="Poison types", default="[clip-unsimilar,clip-similar,random]") # clip,random,unaesth
config.add_argument('--seed', type=int, help='rng seed for splitting', default=6934)
config.add_argument('--overwrite_out', help='Whether to overwrite all older output', action=argparse.BooleanOptionalAction)
config.add_argument('--poisoning_mode', type=str, help='Whether to replace ref pics with poison or append it', default='replace')
args = config.parse_args()

torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)

model, preprocess = None, None
use_embeddings_cache = len(args.embeddings_cache_folder) > 0

def get_clip():
    global model, preprocess
    if model is not None:
        return model, preprocess
    print("Loading CLIP model from disk")
    model_ID = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(model_ID).to('cuda')
    preprocess = CLIPImageProcessor.from_pretrained(model_ID)
    return model, preprocess

def get_emb(pic_path, emb_path):
    if use_embeddings_cache and os.path.exists(emb_path):
        emb = torch.load(emb_path, map_location='cuda')
    else:
        model, preprocess = get_clip()
        image = Image.open(pic_path)
        image = preprocess(image, return_tensors="pt")["pixel_values"].to("cuda")
        with torch.no_grad():
            emb = model.get_image_features(image).to("cpu")
    return emb

if use_embeddings_cache and not os.path.exists(args.embeddings_cache_folder):
    os.mkdir(args.embeddings_cache_folder)

# out/ A(_{poison-type}_poison_{%}):(clean, poison, poisoned)
conceptdirs = [conceptdir for conceptdir in os.listdir(args.input_folder) if os.path.isdir(os.path.join(args.input_folder, conceptdir))]
embeddings_map = {}

print("Computing/loading image embeddings from cache")
total = 0
for conceptdir in conceptdirs:
    for _ in os.listdir(os.path.join(args.input_folder, conceptdir)):
        total += 1

pbar = tqdm(total=total)
for conceptdir in conceptdirs:
    conceptdir_path = os.path.join(args.input_folder, conceptdir)
    pics_paths = []

    for pic in os.listdir(conceptdir_path):
        pic_path = os.path.join(conceptdir_path, pic)
        picname = Path(pic_path).stem
        folder = os.path.join(args.embeddings_cache_folder, conceptdir)
        os.makedirs(folder, exist_ok=True)
        emb_path = os.path.join(folder, f"{picname}.pt")
        emb = get_emb(pic_path, emb_path)
        if use_embeddings_cache:
            torch.save(emb, emb_path)

        embeddings_map[pic_path] = emb
        pbar.update(1)
pbar.close()

print("Starting iterating over dataset and poisoning it")
poison_types = args.poison_types.strip('[').strip(']').split(',')
poison_fractions = [float(x) for x in args.poison_fractions.strip('[').strip(']').split(',')]
print(f"Will be using {poison_types} poison types")
print(f"Poison fractions: {poison_fractions}")

if args.overwrite_out and os.path.exists(args.output_folder):
    shutil.rmtree(args.output_folder)
os.makedirs(args.output_folder, exist_ok=True)

pbar = tqdm(conceptdirs)
for conceptdir in pbar:
    conceptdir_path = os.path.join(args.input_folder, conceptdir)
    conceptdir_files = os.listdir(conceptdir_path)
    conceptdir_num_items = len(conceptdir_files)
    poison_amounts = [int(math.ceil(frac * conceptdir_num_items)) for frac in poison_fractions]
    for p in poison_amounts:
        assert p > 0

    # 1. Copy source pictures to target dir
    out_conceptdir = os.path.join(args.output_folder, f'{conceptdir}_clean')
    os.makedirs(out_conceptdir, exist_ok=True)

    for pic in conceptdir_files:
        pic_path = os.path.join(conceptdir_path, pic)
        picname = Path(pic_path).stem.split('.')[0]
        pic = Image.open(pic_path).convert("RGB")
        pic.save(os.path.join(out_conceptdir, f"{picname}.jpg"))

    # 2. Add poison
    for poison_amount, frac in zip(poison_amounts, poison_fractions):
        # get pictures from our folder
        concept_pics = [i for i in conceptdir_files if i.lower().endswith('.png') or i.lower().endswith('.jpg') or i.lower().endswith('.jpeg')]
        # random sample
        reference_pics = list(rng.choice(concept_pics, poison_amount, replace=False))
        poison_embeddings = {k: v for k, v in embeddings_map.items() if conceptdir not in k} # exclude conceptdir for embeddings

        for poison_type in poison_types:
            out_poisonsdir = os.path.join(args.output_folder, f"{conceptdir}_poisons_{poison_type}_{frac}")
            out_conceptdir = os.path.join(args.output_folder, f"{conceptdir}_poisoned_{poison_type}_{frac}")
            os.makedirs(out_poisonsdir, exist_ok=True)
            os.makedirs(out_conceptdir, exist_ok=True)
            pbar.set_description(f"{conceptdir}, poison {poison_amount}/{conceptdir_num_items} type {poison_type}")
            poison_backlog = [] # to not have duplicates

            try:
                for ref in reference_pics:
                    ref_pic_path = os.path.join(conceptdir_path, ref)
                    picname = Path(ref_pic_path).stem
                    emb_path = os.path.join(args.embeddings_cache_folder, conceptdir, f"{picname}.pt")
                    ref_emb = get_emb(ref_pic_path, emb_path)

                    poison_similarities = {k: torch.nn.functional.cosine_similarity(ref_emb, v).item() for k, v in embeddings_map.items() if k not in poison_backlog and conceptdir not in k}

                    if poison_type == "clip-unsimilar":
                        poison_similarities = dict(sorted(poison_similarities.items(), key=lambda item: item[1])) # least to biggest sim
                        poison_key = list(poison_similarities.keys())[0]
                        poison_backlog.append(poison_key)
                    elif poison_type == "clip-similar":
                        poison_similarities = dict(sorted(poison_similarities.items(), key=lambda item: item[1], reverse=True))
                        poison_key = list(poison_similarities.keys())[0]
                        poison_backlog.append(poison_key)
                    elif poison_type == "random":
                        while True:
                            poison_key = rng.choice(list(poison_similarities.keys()))
                            if not poison_key in poison_backlog:
                                poison_backlog.append(poison_key)
                                break
                    else:
                        raise ValueError("Unsupported poison type!")

                    poison_picname = Path(ref_pic_path).stem.split('.')[0]
                    pic = Image.open(poison_key).convert("RGB")
                    # transfer both to only poisons and poisoned dirs
                    pic.save(os.path.join(out_poisonsdir, f"{poison_picname}.jpg"))
                    pic.save(os.path.join(out_conceptdir, f"{poison_picname}.jpg"))
            except Exception as e:
                print(e)

            # now copy all the clean images to the poisoned dir
            for pic in conceptdir_files:
                # except for ones we used for poisoning if the mode is set to replace
                if pic in reference_pics and args.poisoning_mode == 'replace':
                    continue
                pic_path = os.path.join(conceptdir_path, pic)
                picname = Path(pic_path).stem.split('.')[0]
                pic = Image.open(pic_path).convert("RGB")
                pic.save(os.path.join(out_conceptdir, f"{picname}.jpg"))

pbar.close()
