import enum
from bigprint import BigPrint, launch, TimeCounter
import argparse, datetime, os, shutil, time, json
from tqdm import tqdm
import traceback
from termcolor import colored

config = argparse.ArgumentParser()
config.add_argument('--telegram', help='Message important events into Telegram', action=argparse.BooleanOptionalAction)
config.add_argument('--datasets_folder', type=str, help='Path to datasets folder')
# {prompt}_clean/..., {prompt}_poison.../ ..., {prompt}_poisoned.../ ...
config.add_argument('--checkpoints_folder', type=str, help='Path to keep the checkpoints', default="checkpoints")
config.add_argument('--samples_folder', type=str, help='Path to keep the corresponding sample generations', default="samples")
config.add_argument('--stats_folder', type=str, help='Path to keep the corresponding json stats', default="stats")
config.add_argument('--overwrite_results', help="Whether to proceed overwriting the results folder of it exists", action=argparse.BooleanOptionalAction)

config.add_argument('--losses', type=str, help='Losses to test', default="[l2,huber]")
config.add_argument('--learning_rates', type=str, help='Learning rates', default="[3e-6]")
config.add_argument('--huber_deltas', type=str, help='Huber parameters to test', default="[0.001,0.01,0.0001]")

config.add_argument('--model_name', type=str, help='Core stable diffusion model to base on', default="runwayml/stable-diffusion-v1-5")
config.add_argument('--seed', type=int, help='Learning seed', default=6934)
config.add_argument('--max_training_steps', type=int, help='Num max training steps', default=2001)
config.add_argument('--num_test_images', type=int, help='Num test images', default=16)
config.add_argument('--exit_on_fail', help='exit on fail or skip runs', action=argparse.BooleanOptionalAction)
config.add_argument('--cuda_device', help='cuda_device', default=None, type=int)
config.add_argument('--onlystats', help='compute stats on existing samples with no training', action=argparse.BooleanOptionalAction)
args = config.parse_args()

info0 = f"Configuration:\nlosses {args.losses},\nlrs {args.learning_rates}\nhuber {args.huber_deltas}\ntrain steps {args.max_training_steps} seed {args.seed}\nLogging to Telegram {'ENABLED' if args.telegram else 'DISABLED'}\nProgram will {'EXIT' if args.exit_on_fail else 'CONTINUE'} on fail.\n"

# preparation
assert os.path.exists(args.datasets_folder) and "Datasets folder not found!"
os.makedirs(args.checkpoints_folder, exist_ok=args.overwrite_results)
os.makedirs(args.samples_folder, exist_ok=args.overwrite_results)
os.makedirs(args.stats_folder, exist_ok=args.overwrite_results)
bigprint = BigPrint(args.telegram)

# Step 1. Having a dataset we make a train for each of its folders named {prompt}

# Step 2. After a train completes, we go into logs/checkpoints_{steps} and make an eval with these steps as an arg

# Step 3. Eval is being made with clipsim.py (json parsed)
# we compare "clean" to results, then "poison" to results

run_args = []
losses = args.losses.strip('[').strip(']').split(',')
learning_rates = [float(r) for r in args.learning_rates.strip('[').strip(']').split(',')]
huber_deltas = [float(r) for r in args.huber_deltas.strip('[').strip(']').split(',')]

for loss_type in losses:
    for learning_rate in learning_rates:
        if "huber" in loss_type:
            for delta in huber_deltas:
                run = {
                    "loss_type": loss_type,
                    "lr": learning_rate,
                    "delta": delta,
                    "run_suffix": f"{loss_type}-{delta}_{learning_rate}"
                }
                run_args.append(run)
        else:
            run = {
                "loss_type": loss_type,
                "lr": learning_rate,
                "delta": 0.001, # NOTE: default value which won't influence train script when huber disabled
                "run_suffix": f"{loss_type}_{learning_rate}"
            }
            run_args.append(run)

info1 = f"Number of args combos per run: {len(run_args)}\n\n"

# scan the datasets folder
datasets = {}
for dataset in os.listdir(args.datasets_folder):
    prompt = dataset.split('_')[0]
    if prompt not in datasets:
        datasets[prompt] = {"poison": {}}
    if 'clean' in dataset:
        datasets[prompt]['clean'] = dataset

    if 'poisoned' in dataset:
        name = dataset.replace('poisoned_', '')
        if name not in datasets[prompt]['poison']:
            datasets[prompt]['poison'][name] = {}
        datasets[prompt]['poison'][name]['poisoned'] = dataset
    elif 'poisons' in dataset:
        name = dataset.replace('poisons_', '')
        if name not in datasets[prompt]['poison']:
            datasets[prompt]['poison'][name] = {}
        datasets[prompt]['poison'][name]['poisons'] = dataset

assert len(datasets) > 0
info2 = f" Found {len(datasets)} prepared datasets\n\n"

def launch_cuda(command, strict=False, wait=False):
    if args.cuda_device is not None:
        command = f"CUDA_VISIBLE_DEVICES={args.cuda_device} " + command
    return launch(command, strict, wait)

runs = []

# NOTE: we do separate object/style runs to not overengineer the thing

for prompt, subparts in datasets.items():
    for run in run_args:
        runs.append({**run, "prompt": prompt, "subparts": subparts, "run_name": f"{prompt}+{run['run_suffix']}"})

info3 = f" Runs amount: {len(runs)}"

for run in runs:
    #print(run)
    # tiny pre-run completeness check
    assert 'clean' in run['subparts']
    assert len(run['subparts']['poison'].items()) > 0
    for x, v in run['subparts']['poison'].items():
        assert 'poisons' in v or 'poisoned' in v

bigprint(f"Program initialized at {datetime.datetime.now()},\n", info0, info1, info2, info3)

def get_outpath(run, stage):
    return os.path.join(args.checkpoints_folder, f'{run["run_name"]}+{stage}')

def get_samples_outpath(run, stage):
    return os.path.join(args.samples_folder, f'{run["run_name"]}+{stage}')

def launch_train(run, dataset, stage):
    dataset_path = os.path.join(args.datasets_folder, dataset)
    output_path = get_outpath(run, stage)
    if not args.onlystats:
        #launch(f'''MODEL_NAME='"{args.model_name}"' INSTANCE_DIR='"{dataset_path}"' OUTPUT_DIR='"{output_path}"' PROMPT='"{run['prompt']}"' LR='{run["lr"]}' LOSS_TYPE='{run["loss_type"]}' MAX_STEPS='{args.max_training_steps}' ./train.sh''', strict=False, wait=True)
        launch_cuda(f'''MODEL_NAME="{args.model_name}" INSTANCE_DIR="{dataset_path}" OUTPUT_DIR="{output_path}" PROMPT="{run['prompt']}" LR='{run["lr"]}' LOSS_TYPE='{run["loss_type"]}' MAX_STEPS='{args.max_training_steps}' ./train.sh''', strict=False, wait=True)

def launch_sampling(run, lora_path, stage):
    samples_outpath = get_samples_outpath(run, stage)
    steps = []
    outpaths = []

    ### Now for each gathered N-steps checkpoint
    for check in os.listdir(lora_path):
        if check.startswith('checkpoint-'):
            check_path = os.path.join(lora_path, check)
            N = int(check.strip('checkpoint-'))
            steps.append(N)
            cur_samples_outpath = os.path.join(samples_outpath, f"{N:06d}_steps")
            os.makedirs(cur_samples_outpath, exist_ok=True)
            outpaths.append(cur_samples_outpath)

            if not args.onlystats:
                launch_cuda(f'python generate_test_pics.py --model_name "{args.model_name}" --lora_path "{check_path}" --seed {args.seed} --num_images {args.num_test_images} --out_path "{cur_samples_outpath}" --prompt "{run["prompt"]}"', strict=True, wait=True)
    
    return steps, outpaths

def launch_eval(run, steps, outpaths, reference_folder, stage, total_reference_folder, total_pics_folder, total_save_cfg_path):
    info_outpaths = []
    for step, outpath in zip(steps, outpaths):
        info_folder = os.path.join(args.stats_folder, run["run_name"])
        os.makedirs(info_folder, exist_ok=True)
        info_outpath = os.path.join(info_folder, f"{stage}_{step:06d}.json")

        total_reference_folder.append(reference_folder)
        total_pics_folder.append(outpath)
        total_save_cfg_path.append(info_outpath)
        #launch(f'python clipsim.py --reference_folder "{reference_folder}" --pics_folder "{outpath}" --save_cfg_path "{info_outpath}"', strict=True, wait=True)
        info_outpaths.append(info_outpath)
    return info_outpaths

pbar = tqdm(enumerate(runs))

for i, run in pbar:
    run_name = run['run_name']
    bigprint(f"Starting run {run_name}, {i+1} out of {len(runs)} at {datetime.datetime.now()}")
    stage = ""

    try:
        # 1. Training
        start_time = time.time()

        # a. on clean
        stage = "clean"
        dataset = run['subparts']['clean']

        with TimeCounter() as tc:
            launch_train(run, dataset, stage)
            
        bigprint(f'[{run_name}]: Clean training completed in {tc("mins")} minutes.\nStarting poisoned trainings')

        # b. on each poisoned dir
        num_poisons = len(run['subparts']['poison'])
        for i, (poison_name, poison_paths) in enumerate(run['subparts']['poison'].items()):
            stage = poison_name
            # poisons = poison_paths['poisons']
            poisoned = poison_paths['poisoned']
            dataset = poisoned

            with TimeCounter() as tc:
                launch_train(run, dataset, stage)
            
            bigprint(f"[{run_name}]: Poison {poison_name} training ({i+1}/{num_poisons}) completed in {tc('mins')} mins")

        train_time_elapsed = int((time.time() - start_time) / 60)

        bigprint(f'[{run_name}]: Trainings completed in {train_time_elapsed} minutes.\nStarting sampling')

        # 2. Sampling
        start_time = time.time()

        # a. make bare model generation
        stage = "bare_model_sampling"
        print(colored(f"[{run_name}]: {stage}", color='cyan'))
        bare_sampling_path = get_samples_outpath(run, stage)
        if not args.onlystats:
            launch_cuda(f'python generate_test_pics.py --model_name "{args.model_name}" --lora_path "" --seed {args.seed} --num_images {args.num_test_images} --out_path "{bare_sampling_path}" --prompt "{run["prompt"]}"', strict=True, wait=True)

        # b. make clean-dataset model generation
        stage = "clean_sampling"
        print(colored(f"[{run_name}]: {stage}", color='cyan'))
        clean_sampling_steps, clean_sampling_paths = launch_sampling(run, get_outpath(run, "clean"), stage)

        stage = "poisoned_sampling"
        print(colored(f"[{run_name}]: {stage}", color='cyan'))
        poisoning_steps = {}
        # c. make poisoned-dataset model generation
        for i, (poison_name, poison_paths) in enumerate(run['subparts']['poison'].items()):
            print(f"{stage}_{poison_name}")
            # poisons = poison_paths['poisons']
            # poisoned = poison_paths['poisoned']
            poisoning_sampling_steps, poisoning_sampling_paths = launch_sampling(run, get_outpath(run, poison_name), f"{poison_name}_{stage}")
            poisoning_steps[poison_name] = {
                'poisoning_sampling_steps': poisoning_sampling_steps,
                'poisoning_sampling_paths': poisoning_sampling_paths
            }

        sampling_time_elapsed = int((time.time() - start_time) / 60)

        bigprint(f"[{run_name}]: Sampling completed in {sampling_time_elapsed} mins.\nStarting evaluation")

        # 3. Evaluating Metrics

        start_time = time.time()

        # bare samples to clean dataset -- for baseline
        # bare samples to poison dataset -- for baseline

        # !!! BATCHED  CLIPSIM.PY MODE: DATASET AND SAMPLING PATHS AS LISTS
        # !!! ELSE IT'S A DISASTER TO WAIT FOR EVAL RESULTS

        total_reference_folder = []
        total_pics_folder = []
        total_save_cfg_path = []

        stage = "bare_to_clean"
        print(colored(f"[{run_name}]: Similarity {stage}", color='magenta'))
        dataset = run['subparts']['clean']
        dataset = os.path.join(args.datasets_folder, dataset)
        info_folder = os.path.join(args.stats_folder, run["run_name"])
        os.makedirs(info_folder, exist_ok=True)
        bare_to_clean_info_outpath = os.path.join(info_folder, f"{stage}.json")

        total_reference_folder.append(dataset)
        total_pics_folder.append(bare_sampling_path)
        total_save_cfg_path.append(bare_to_clean_info_outpath)

        #launch(f'python clipsim.py --reference_folder "{dataset}" --pics_folder "{bare_sampling_path}" --save_cfg_path "{bare_to_clean_info_outpath}"', strict=True, wait=True)

        bare_to_poison_outpaths = {}
        for i, (poison_name, poison_paths) in enumerate(run['subparts']['poison'].items()):
            stage = f"bare_to_poison_{poison_name}"
            print(colored(f"[{run_name}]: Similarity {stage}", color='magenta'))
            poisons = poison_paths['poisons']
            poisons = os.path.join(args.datasets_folder, poisons)
            info_folder = os.path.join(args.stats_folder, run["run_name"])
            os.makedirs(info_folder, exist_ok=True)
            bare_to_poison_outpaths[poison_name] = os.path.join(info_folder, f"{stage}.json")

            total_reference_folder.append(poisons)
            total_pics_folder.append(bare_sampling_path)
            total_save_cfg_path.append(bare_to_poison_outpaths[poison_name])

            #launch(f'python clipsim.py --reference_folder "{poisons}" --pics_folder "{bare_sampling_path}" --save_cfg_path "{bare_to_poison_outpaths[poison_name]}"', strict=True, wait=True)

        # clean samples to clean dataset -- main metric
        # clean samples to poison dataset -- for baseline

        stage = "clean_to_clean"
        print("Evaluating ")
        print(colored(f"[{run_name}]: Similarity {stage}", color='magenta'))
        dataset = run['subparts']['clean']
        dataset = os.path.join(args.datasets_folder, dataset)
        clean_to_clean_outpaths = launch_eval(run, clean_sampling_steps, clean_sampling_paths, dataset, stage, total_reference_folder, total_pics_folder, total_save_cfg_path)
        
        clean_to_poison_outpaths = {}

        for i, (poison_name, poison_paths) in enumerate(run['subparts']['poison'].items()):
            stage = f"clean_to_poison_{poison_name}"
            print(colored(f"[{run_name}]: Similarity {stage}", color='magenta'))
            poisons = poison_paths['poisons']
            poisons = os.path.join(args.datasets_folder, poisons)
            clean_to_poison_outpaths[poison_name] = launch_eval(run, clean_sampling_steps, clean_sampling_paths, poisons, stage, total_reference_folder, total_pics_folder, total_save_cfg_path)

        # poisoned samples to clean dataset -- main metric
        # poisoned samples to poison dataset -- main metric

        poison_to_clean_outpaths = {}
        poison_to_poison_outpaths = {}

        for i, (poison_name, poison_paths) in enumerate(run['subparts']['poison'].items()):
            stage = f"poison_to_clean_{poison_name}"
            print(colored(f"[{run_name}]: Similarity {stage}", color='magenta'))
            dataset = run['subparts']['clean']
            dataset = os.path.join(args.datasets_folder, dataset)
            poisons = poison_paths['poisons']
            poisons = os.path.join(args.datasets_folder, poisons)
            poison_to_clean_outpaths[poison_name] = launch_eval(run, poisoning_steps[poison_name]['poisoning_sampling_steps'], poisoning_steps[poison_name]['poisoning_sampling_paths'], dataset, stage, total_reference_folder, total_pics_folder, total_save_cfg_path)
            stage = f"poison_to_poison_{poison_name}"
            poison_to_poison_outpaths[poison_name] = launch_eval(run, poisoning_steps[poison_name]['poisoning_sampling_steps'], poisoning_steps[poison_name]['poisoning_sampling_paths'], poisons, stage, total_reference_folder, total_pics_folder, total_save_cfg_path)

        total_reference_folder = f"[{','.join(total_reference_folder)}]"
        total_pics_folder = f"[{','.join(total_pics_folder)}]"
        total_save_cfg_path = f"[{','.join(total_save_cfg_path)}]"

        launch_cuda(f'python clipsim.py --reference_folder "{total_reference_folder}" --pics_folder "{total_pics_folder}" --save_cfg_path "{total_save_cfg_path}"', strict=True, wait=True)

        evaluation_time_elapsed = int((time.time() - start_time) / 60)

        bigprint("Parsing info")

        def parse_clipsim_info(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                cfg = json.loads(f.read())
                clipsim_cfg = cfg['CLIP Sim']
                clipsim_info = {
                    'mean': clipsim_cfg['mean'],
                    'std': clipsim_cfg['std'],
                    'max': clipsim_cfg['max'],
                    'min': clipsim_cfg['min'],
                }

            return clipsim_info

        def format_clipsim_info(clipsim_info):
            return f"mean: {clipsim_info['mean']}\nstd: {clipsim_info['std']}\nmin: {clipsim_info['min']}\nmax: {clipsim_info['max']}"
        
        def prepare_clipsim_info_for_output(json_path):
            clipsim_info = parse_clipsim_info(json_path)
            return f"{json_path}\n\n{format_clipsim_info(clipsim_info)}\n"

        # a. bare model output
        txt = "Bare model\n"
        txt += prepare_clipsim_info_for_output(bare_to_clean_info_outpath)
        for i, (poison_name, poison_paths) in enumerate(run['subparts']['poison'].items()):
            txt += "\n" + prepare_clipsim_info_for_output(bare_to_poison_outpaths[poison_name])
        bigprint(txt)

        # b. clean dataset similarity to pictures
        txt = "Clean model\n"
        txt += prepare_clipsim_info_for_output(clean_to_clean_outpaths[-1])
        for i, (poison_name, poison_paths) in enumerate(run['subparts']['poison'].items()):
            txt += prepare_clipsim_info_for_output(clean_to_poison_outpaths[poison_name][-1])
        bigprint(txt)

        # c. poisoned models similarity
        txt = "Poisoned models\n"
        for i, (poison_name, poison_paths) in enumerate(run['subparts']['poison'].items()):
            txt += prepare_clipsim_info_for_output(poison_to_clean_outpaths[poison_name][-1])
            txt += prepare_clipsim_info_for_output(poison_to_poison_outpaths[poison_name][-1])
        bigprint(txt)

        bigprint(f"Run {run_name}, {i+1} out of {len(runs)} COMPLETED at {datetime.datetime.now()}")
    
    except Exception as e:
        traceback.print_exc()
        if args.exit_on_fail:
            bigprint(f"""ERROR! We died at run {i} id {run["run_name"]} stage {stage} because of {str(e).replace('"', ' ')}.""")
            raise e
        else:
            bigprint(f"""WARNING! We died at run {i} id {run["run_name"]} stage {stage} because of {str(e).replace('"', ' ')}.\nThis run will be skipped. Please investigate the problem and rerun it (don't forget to remove corrupted folders).""")
            continue

pbar.close()

bigprint("The job is completed. Have a cake 🍰")
