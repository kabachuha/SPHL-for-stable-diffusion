import torch, os, math
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer
from PIL import Image
import argparse
from rich.console import Console
console = Console()
from rich.table import Table
from rich import box
import numpy as np
from termcolor import colored
import lpips
import json

config = argparse.ArgumentParser()
config.add_argument('--reference_folder', type=str, help='Path to folder A')
config.add_argument('--pics_folder', type=str, help='Path to folder B')
config.add_argument('--save_cfg_path', type=str, help='Path for stats saving (.json)', default='')
config.add_argument('--lpips_type', type=str, help="LPIPS type", default="vgg")
config.add_argument('--seed', type=int, help='Torch seed', default=6934)
config.add_argument('--fancy', help="whether to print fancy tables", action=argparse.BooleanOptionalAction)
args = config.parse_args()

torch.manual_seed(args.seed)

# Load the LPIPS model
loss_fn_vgg = lpips.LPIPS(net=args.lpips_type).to('cuda')

# Load the CLIP model
model_ID = "openai/clip-vit-large-patch14"
model = CLIPModel.from_pretrained(model_ID).to('cuda')

preprocess = CLIPImageProcessor.from_pretrained(model_ID) #.to('cuda')

# Define a function to load an image and preprocess it for CLIP
def load_and_preprocess_image(image_path):
    # Load the image from the specified path
    image = Image.open(image_path)

    # Apply the CLIP preprocessing to the image
    image = preprocess(image, return_tensors="pt")

    # Return the preprocessed image
    return image

def arg_to_list(arg):
    if arg.startswith('['):
        folders = arg.strip('[').strip(']').split(',')
    else:
        folders = [arg]
    return folders

reference_folders = arg_to_list(args.reference_folder)
pics_folders = arg_to_list(args.pics_folder)
save_cfg_paths = arg_to_list(args.save_cfg_path)
if save_cfg_paths == '':
    save_cfg_paths = save_cfg_paths * len(reference_folders)

for reference_folder, pics_folder, save_cfg_path in zip(reference_folders, pics_folders, save_cfg_paths):

    # for incomplete full runs
    if not os.path.exists(reference_folder) or not os.path.exists(pics_folder):
        # Save to json file if specified
        if save_cfg_path is not None and len(save_cfg_path) > 0:
            with open(f"{save_cfg_path}", 'w', encoding='utf-8') as f:
                f.write(json.dumps({}, indent=4, ensure_ascii=False))
        continue

    sims = {}
    sims_lpips = {}
    num_pics = 0
    num_trash = 0

    for t_f in os.listdir(reference_folder):
        num_trash += 1
        sims[t_f] = []
        sims_lpips[t_f] = []
        # Load the two images and preprocess them for CLIP
        image_a = load_and_preprocess_image(os.path.join(reference_folder,t_f))["pixel_values"].to("cuda")

        # Calculate the embeddings for the images using the CLIP model
        with torch.no_grad():
            embedding_a = model.get_image_features(image_a)
        
        for p_f in os.listdir(pics_folder):
            image_b = load_and_preprocess_image(os.path.join(pics_folder,p_f))["pixel_values"].to("cuda")

            with torch.no_grad():
                embedding_b = model.get_image_features(image_b)

            # Calculate the cosine similarity between the embeddings
            similarity_score = torch.nn.functional.cosine_similarity(embedding_a.cpu(), embedding_b.cpu())
            sims[t_f] += [{"img":p_f, "score":similarity_score.item()}]

            with torch.no_grad():
                lpips_score = loss_fn_vgg(image_a, image_b)
            
            sims_lpips[t_f] += [{"img":p_f, "score": 1 - lpips_score.cpu().item()}]

        num_pics = len(sims[t_f])

    table_sims = Table(padding=0, box=box.ROUNDED)
    table_lpips = Table(padding=0, box=box.ROUNDED)

    for table in [table_sims, table_lpips]:
        table.add_column("Trash\\Picture", justify="right", style="cyan", no_wrap=False)

        for i in range(num_pics):
            table.add_column(f"{i}", justify="left", style="green", no_wrap=True)

        table.add_column("Mean", justify="right", style="bold yellow", no_wrap=True)
        table.add_column("Std", justify="right", style="magenta", no_wrap=True)
        table.add_column("Min", justify="right", style="blue", no_wrap=True)
        table.add_column("Max", justify="right", style="yellow", no_wrap=True)

    stats = {"CLIP Sim": {"runs":sims}, "1 - LPIPS": {"runs":sims_lpips}}

    for table, s, text in zip([table_sims, table_lpips], [sims, sims_lpips], ["CLIP Sim", "1 - LPIPS"]):

        print("")
        print(colored(text, color="red"))

        mean = 0
        std = 0
        t_min = 1
        t_max = -1

        for k,v in s.items():
            num_trash += 1
            row = [k[:30]]
            s_vals = []
            for i in v:
                s = i["score"]
                s_vals += [s]
                row += [f"{s:.2f}"]
            s_vals = np.array(s_vals)
            s_mean = float(s_vals.mean())
            s_std = float(s_vals.std())
            s_min = float(s_vals.min())
            s_max = float(s_vals.max())
            row += [f"{s_mean:.2f}"]
            row += [f"{s_std:.2f}"]
            row += [f"{s_min:.2f}"]
            row += [f"{s_max:.2f}"]
            table.add_row(*row)

            mean += s_mean
            std += s_std**2

            if s_min < t_min:
                t_min = s_min
            if s_max > t_max:
                t_max = s_max

        mean = mean / num_trash
        std = math.sqrt(std)

        stats[text]["mean"] = mean
        stats[text]["std"] = std
        stats[text]["min"] = t_min
        stats[text]["max"] = t_max

        if args.fancy:
            console.print(table)
            print("")

        print("Overall mean: ", colored(f"{mean:.3f}", 'yellow', attrs=["bold"]))
        print("Overall std: ", colored(f"{std:.3f}", 'magenta'))
        print("Overall min: ", colored(f"{t_min:.3f}", 'blue'))
        print("Overall max: ", colored(f"{t_max:.3f}", 'yellow'))

    print("")
    print("Num pics: ", num_pics)
    print("Num trash items: ", num_trash)

    stats["num_pics"] = num_pics
    stats["num_trash"] = num_trash
    stats["lpips_type"] = args.lpips_type

    # Save to json file if specified
    if save_cfg_path is not None and len(save_cfg_path) > 0:
        with open(f"{save_cfg_path}", 'w', encoding='utf-8') as f:
            f.write(json.dumps(stats, indent=4, ensure_ascii=False))
