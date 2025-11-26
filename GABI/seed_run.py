import os
import sys
import time
from datetime import datetime
import random
import numpy as np

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

from trainers.balance_pretrain import pretrain
from trainers.evaluate import evaluate
from trainers.test import test
from utils.utils import grab_arg_from_checkpoint, prepend_paths, re_prepend_paths

torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
hydra.HYDRA_FULL_ERROR = 1


def run(args: DictConfig):
    """Single run with one seed"""
    now = datetime.now()
    start = time.time()
    pl.seed_everything(args.seed)

    args = prepend_paths(args)
    time.sleep(random.randint(1, 5))

    # ============ Handle resume case ============
    if args.resume_training:
        if args.wandb_id:
            wandb_id = args.wandb_id
        tmp_data_base = args.data_base
        checkpoint = args.checkpoint
        ckpt = torch.load(args.checkpoint)
        args = ckpt["hyper_parameters"]
        args = OmegaConf.create(args)
        args.checkpoint = checkpoint
        args.resume_training = True
        if "wandb_id" not in args or not args.wandb_id:
            args.wandb_id = wandb_id
        args.data_base = tmp_data_base
        args = re_prepend_paths(args)

    base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    base_dir = os.path.join(os.path.dirname(os.path.dirname(base_dir)), "result")

    exp_name = f'{args.exp_name}_{args.target}_seed{args.seed}_{now.strftime("%m%d_%H%M")}'
    if args.use_wandb:
        wandb_logger = WandbLogger(name=exp_name, project=args.wandb_project)
    else:
        wandb_logger = None

    print(f"\n========== Run with seed {args.seed} ==========")
    print(f"Pretrain LR: {args.lr}, Decay: {args.weight_decay}")
    print(f"Finetune LR: {args.lr_eval}, Decay: {args.weight_decay_eval}")

    result = {}  

    if args.pretrain:
        print("\nStart pretraining ...")
        torch.cuda.empty_cache()
        pretrain(args, wandb_logger)
        args.checkpoint = os.path.join(
            base_dir, "runs", args.datatype,
            wandb_logger.experiment.name,
            f"checkpoint_last_epoch_{args.max_epochs-1:02}.ckpt"
        )

    if args.test:
        print("\nTesting ...")
        result = test(args, wandb_logger)
    elif args.evaluate:
        print("\nStart Finetuning ...")
        torch.cuda.empty_cache()
        result = evaluate(args, wandb_logger)

    if wandb_logger:
        wandb.finish()
        del wandb_logger

    end = time.time()
    print(f"Total time: {(end - start) / 60:.2f} min")

    return result


@hydra.main(config_path="./configs", config_name="config_ADNI_GABI", version_base=None)
def control(args: DictConfig):
    """Run 5 seeds and compute mean/std"""
    seeds = [2022, 2023, 2024, 2025, 2026]
    all_results = []

    for seed in seeds:
        args.seed = seed
        result = run(args)
        if result:
            all_results.append(result)

    if len(all_results) > 0:
        
        keys = all_results[0].keys()
        print("\n===== Summary across 5 seeds =====")
        for k in keys:
            vals = np.array([r[k] for r in all_results])
            print(f"{k}: mean={vals.mean():.4f}, std={vals.std():.4f}")
    else:
        print("No results collected from runs.")


if __name__ == "__main__":
    control()
