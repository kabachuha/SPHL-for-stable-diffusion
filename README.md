# Scheduled Pseudo-Huber Loss for Diffusion Models

This GitHub repo contains the code for the paper "Improving Diffusion Models's Data-Corruption Resistance using Scheduled Pseudo-Huber Loss" https://arxiv.org/abs/2403.16728.

(NOTE: only text2image experiments code is present in this repo)

The content is composed of two parts:

- Ready-to-use Diffusers scripts in the folder `diffusers_scripts`. (They are going be pushed to the Diffusers library itself soon, and it will be more handy to use it instead. Track https://github.com/huggingface/diffusers/issues/7488)

- Code for mass training sweeps, statistics collection and analysis, if you'd like to replicate the results.

## Instruction:

### For end-user usage (most likely, you need this):

Proceed to `diffusers_scripts` and then launch the desired training script with the same instructions as in http://github.com/huggingface/diffusers/examples/. Don't forget to specify `loss_type` in the training arguments!

### For replication:

Install requirements

`pip install -r requirements.txt`

Make a `concepts` folder and put any amounts of subfolders containing images of same concepts (clean datasets). (and you can also include a random pictures folder, then exclude it from the dataset once it's formed).

Run `dataset_composer.py` (see it's argparse args), by default it will make a `datasets` folder with the results.

Run `script.sh`

!!! If you would like to receive messages of each job completion to your Telegram, remember to login into `telegram-send`` before the start! Depending on your GPU it can take hours or days!

Once it's completed, you will see a folder named `stats` (by default).

Then you can use `analyzer.ipynb` to parse the stats, analyze them and make the plots.

PM me on discord or email if you have any questions.

## Citation

```
@misc{khrapov2024improving,
      title={Improving Diffusion Models's Data-Corruption Resistance using Scheduled Pseudo-Huber Loss}, 
      author={Artem Khrapov and Vadim Popov and Tasnima Sadekova and Assel Yermekova and Mikhail Kudinov},
      year={2024},
      eprint={2403.16728},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```
