from train import train
import wandb  # cloud-based experiment tracking logger
import torch
import os
from global_vars import ROOT_DIR, ROOT_LOG_DIR  # everything log-related will be saved in this ROOT_LOG_DIR
import argparse
from utils.args_utils import add_all_args, check_and_update_args  # examine these for args
from train import train_utils
from models import models_utils
from data import data_utils
from utils import logging_utils

NUM_WORKERS = 4  # parallelism for data loading... can reduce if there's memory issue


def main(configs):
    train_utils.set_seed_all(configs.seed)  # this set the randomness for the entire experiment. Can set to no seed
    if configs.wandb:  # wandb is a cloud-based logger that helps with visualization and tracking stats
        run = logging_utils.initialize_wandb(configs)  # sets run name, project name, and log configs
    # utils.log_args(configs, None)
    logging_utils.print_args(configs)
    device = torch.device(f"cuda:{configs.gpu}" if torch.cuda.is_available else "cpu")
    if device.type == 'cpu':
        print("[WARNING] Using CPU!")

    ##########################################################
    ############        Data and Loaders            ##########
    ##########################################################
    # val_data can be None if val_split_proportion is 0. None datasets will be ignored.
    train_data, val_data, test_data, n_classes, collator = \
        data_utils.get_dataset(dataset_name=configs.dataset, args=configs)
    ## a collator preprocess the data obtained from the dataloader
    if configs.no_test:  # set this to avoid peeking into test results
        test_data = None

    loader_kwargs = {  # settings for DataLoaders
        "batch_size": configs.batch_size,
        "num_workers": NUM_WORKERS,  # make this smaller if we are having memory problem
        "pin_memory": True,
        "collate_fn": collator  # this should be None for vision datasets and collators like padding for NLP datasets
    }
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, **loader_kwargs)
    val_loader = torch.utils.data.DataLoader(val_data, **loader_kwargs) if val_data is not None else None
    test_loader = torch.utils.data.DataLoader(test_data, **loader_kwargs) if test_data is not None else None
    data = {"train_loader": train_loader, "val_loader": val_loader, "test_loader": test_loader,
            "train_data": train_data, "val_data": val_data, "test_data": test_data}
    logging_utils.log_data(data, n_classes)

    ##########################################################
    ####   Get the right model according to configs  #########
    ##########################################################
    model = models_utils.get_model(model_name=configs.model, args=configs, n_classes=n_classes)
    if configs.wandb:
        wandb.watch(model)  # this tracks gradients
    model.to(device)

    ##########################################################
    ########              Train and log             ##########
    ##########################################################
    train.train(model, data, configs, device=device, wandb=wandb if configs.wandb else None)
    if configs.wandb:  # cleanup wandb
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # add args
    add_all_args(parser)
    args = parser.parse_args()
    # this performs some processing and checking. Also set dataset specific default args
    check_and_update_args(args)

    if args.log_dir == "AUTO":
        logroot = f"{ROOT_LOG_DIR}/{args.dataset}/autolog_{args.project_name}"
        run_specific = logging_utils.set_wandb_run_name(args)
        args.log_dir = os.path.join(logroot, os.path.join(args.optimizer, run_specific))

    main(args)
