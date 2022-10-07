from datasets import load_metric
from global_vars import GLUE_DATASETS, VISION_DATASETS, MLM_DATASETS

metrics_for_datasets = {  # todo: can fill this out
    'sst2': load_metric('glue', 'sst2'),
    'mrpc': load_metric("glue", "mrpc")
}


def get_accuracy_metric():
    return load_metric("accuracy")


def get_metrics_for_dataset(dataset_name):  # todo: learn how to use HF metrics better!!!
    """
        Return metrics according to the dataset we are working with (if applicable)
    """
    if dataset_name in GLUE_DATASETS:
        return [load_metric("glue", dataset_name)]
    elif dataset_name in VISION_DATASETS:
        return None  # this means default to just accuracy
    elif dataset_name in MLM_DATASETS:
        # return [load_metric("perplexity")]
        return []
    else:
        raise NotImplementedError
