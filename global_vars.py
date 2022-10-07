"""
This file includes global variable definition for argument parsing as well as default settings for datasets
"""
import os

PROJECT_NAME = "VR"

ROOT_DIR = os.environ['RESEARCH_ROOT_DIR']  # make sure to set the system env variable RESEARCH_ROOT_DIR=<dir_to_root>
DATA_ROOT_DIR = os.path.join(ROOT_DIR, "datasets")
ROOT_LOG_DIR = os.path.join(ROOT_DIR, "logs/fastr")
WANDB_LOG_DIR = ROOT_LOG_DIR  # ensure whatever this is has full write permission.

# TRAINING
AVAIL_OPTIMIZERS = ["sgd", "adam", "adagrad", "adamw",
                    "stormplus", "fastrn", "fastrd", "repstormplus", "rmsprop"]
AVAIL_SCHEDULERS = ["linear"]
AVAIL_LOSSES = ["cross_entropy"]

# DATASETS
## -- GLUE NLP DATASETS (https://openreview.net/pdf?id=rJ4km2R5t7)
GLUE_SINGLE_DATASETS = ['sst2']  # standard single 'sentence' input for tasks like sentiment analysis
GLUE_PAIR_DATASETS = ['mrpc', 'mnli']  # the inputs are 'pairs' of sentences for tasks like comparison and inferences
GLUE_DATASETS = GLUE_SINGLE_DATASETS + GLUE_PAIR_DATASETS
## Masked language model (MLM) datasets (for pretraining transformers and finetuning pretrained models like BERT etc.)
MLM_DATASETS = ['imdb']
NLP_DATASETS = GLUE_DATASETS + MLM_DATASETS

## -- VISION DATASETS
VISION_DATASETS = ['cifar10', 'cifar100', 'mnist', 'imagenet']
AVAIL_DATASETS = NLP_DATASETS + VISION_DATASETS

# MODELS
VISION_MODELS = ['resnet50', 'resnet34', 'resnet18', 'simple']
NLP_MODELS = ['bert', 'distilbert']
AVAIL_MODELS = VISION_MODELS + NLP_MODELS

# DEFAULT ARGUMENTS SETTINGS
NO_SEED = None
DEFAULT_N_EPOCHS = 100
DEFAULT_BATCH_SIZE = 64

default_args_dict = {  # this sets generic default args (cifar10 sgd)
    'n_epochs': 100,
    'batch_size': 64,
    'weight_decay': 0,
    'lr': 1e-3,
    'loss': "cross_entropy",
    'optimizer': 'sgd',
    'model': "resnet18",
    'seed': NO_SEED,
    'log_every': 50,
    'no_test': False,
    'scheduler': None,
}
# set dataset specific settings to overwrite the generic default args.
# Omit if still want to use generic default
default_data_setting_dict = {  # if we would like to set dataset specific params
    'imagenet': {'n_epochs': 50, 'batch_size': 128, 'log_every': 500,
                 'weight_decay': 1e-4},
    'mrpc': {'n_epochs': 3, 'batch_size': 8, 'log_every': 100,
             'weight_decay': 0.0, 'lr': 5e-5, 'optimizer': 'adamw', 'scheduler': 'linear',
             'model': 'bert'},
    'sst2': {'n_epochs': 4, 'batch_size': 16, 'log_every': 100,
             'weight_decay': 0.0, 'lr': 2e-5, 'optimizer': 'adamw', 'scheduler': 'linear',
             'model': 'bert', 'no_test': True},  # Test set doesn't contain labels!!!!
    'mnli': {'n_epochs': 4, 'batch_size': 8, 'log_every': 1000,
             'weight_decay': 0.0, 'lr': 2e-5, 'optimizer': 'adamw', 'scheduler': 'linear',
             'model': 'bert', 'no_test': True},  # Test set doesn't contain labels!!!!
    'imdb': {'n_epochs': 10, 'batch_size': 16, 'log_every': 25,
             'weight_decay': 0.01, 'lr': 2e-5, 'optimizer': 'adamw', 'scheduler': 'linear',
             'model': 'distilbert', 'no_test': False},
    'squad': {'n_epochs': 4, 'batch_size': 64, 'log_every': 25,
              'weight_decay': 0.01, 'lr': 2e-5, 'optimizer': 'adamw', 'scheduler': 'linear',
              'model': 'bert-cased', 'no_test': False},
    'mnist': {'model': 'simple', 'n_epochs': 50, 'batch_size': 128, 'weight_decay': 0, 'lr': 1e-2}
}
