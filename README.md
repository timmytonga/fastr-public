# FASTR: Fully Adaptive STochastic Recursive-momentum

## Installation and Setup
### Environment and Packages
Here we assume that you have Conda and CUDA installed. 

First create a conda environment with the appropriate packages. 
`<env>` is whatever env name you want to put. 
Then activate the env

    conda activate <env>

Then install via pip

    conda install pip
    pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113



### Setting up the correct directories
To prevent (potentially large) files from getting saved all over the place, 
you should modify the `ROOT_DIR` variable in `global_vars.py`. The datasets will be downloaded
into that `ROOT_DIR` along with all the logs file. 

This can be done by setting the environment variable `RESEARCH_ROOT_DIR` to the appropriate path.
You can add something like `export RESEARCH_ROOT_DIR=<path_to_root_dir>` in your `~/.bashrc` file

## Running things
Run `python main.py --help` to see the full list of available options. 
It might be easier to navigate (as well as to see the default options) via the `args.py` file.


### Example
Here's an example to run Adam on the SST2 dataset:

    python main.py --dataset sst2 --optimizer adam --lr 1e-4 --weight_decay 1e-4 --wandb 

## Adding a new dataset 
It is crucial to understand how the pipeline is run before attempting to add a new task.
First `main.py` will call `data_utils.get_dataset` along with the configs.
That should return some Dataset objects for each train/val/test split.

Then, we need to correctly grab a compatible model with the dataset and the task. 
That is accomplished via `models_utils.get_model`. 
This is especially important for NLP models like BERT 
as we might need a different head for different dataset (e.g. sequence classification, question answering, generation, etc.)

### Add installation instructions 
Include new packages in `requirements.txt` if necessary (for example HuggingFace packages with NLP tasks)
### Returning Dataset object and train/val/test split

TODO

### Update metrics in `data_utils/metric.py`
TODO

### Updating the `global_vars.py` file
This sets the default parameter for the dataset. TODO

#### Setting the appropriate models
TODO

#### Setting the default parameters for the dataset
Update the `default_data_setting_dict` dict in `global_vars.py`.