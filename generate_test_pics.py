from tqdm import tqdm
from diffusers import StableDiffusionPipeline
import torch, os, argparse, shutil

config = argparse.ArgumentParser()
config.add_argument('--model_name', type=str, help='Core model name', default="runwayml/stable-diffusion-v1-5")
config.add_argument('--lora_path', type=str, help='Path to generated lora folder (empty for bare model)')
config.add_argument('--seed', type=int, help='initial rng seed', default=6934)
config.add_argument('--num_images', type=int, help='num sampled images', default=16)
config.add_argument('--num_steps', type=int, help='num inference steps', default=50)
config.add_argument('--guidance_scale', type=float, help='guidance scale', default=7.5)
config.add_argument('--out_path', type=str, help='Path to save end images')
config.add_argument('--overwrite_all', type=bool, default=True, help='Whether to overwrite all images')
config.add_argument('--prompt', type=str, help='Generation prompt')
config.add_argument('--batch_size', type=int,default=8, help="Gen batch size")

args = config.parse_args()

pipe = StableDiffusionPipeline.from_pretrained(args.model_name, torch_dtype=torch.float16, safety_checker=None)
if args.lora_path is not None and len(args.lora_path) > 0:
    pipe.unet.load_attn_procs(os.path.abspath(args.lora_path), local_files_only=True)
pipe.to("cuda")

if os.path.exists(args.out_path) and args.overwrite_all:
    shutil.rmtree(args.out_path)
os.makedirs(args.out_path, exist_ok=True)

for i in tqdm(range(args.num_images//args.batch_size)):
    generator = torch.Generator(device="cuda").manual_seed(args.seed + i) # * args.batch_size)

    prompt = args.prompt
    images = pipe([prompt] * args.batch_size, num_inference_steps=args.num_steps, guidance_scale=args.guidance_scale, generator=generator).images
    for j, image in enumerate(images):
        image.save(os.path.join(args.out_path, f"{args.seed + i * args.batch_size + j}.jpg"))
    #images[0].save(os.path.join(args.out_path, f"{args.seed + i}.jpg"))

