from global_vars import DATA_ROOT_DIR, GLUE_PAIR_DATASETS, GLUE_SINGLE_DATASETS, GLUE_DATASETS, MLM_DATASETS
from datasets import load_dataset  # huggingface package that has datasets from huggingface hub
from transformers import AutoTokenizer, DataCollatorWithPadding
import os
from models.models_utils import MODEL_TO_CHECKPOINT_NAME
import numpy as np
import collections


# chunk size = max length of input in BERT --> use this to cap chunk size to avoid running out of GPU memory
MAX_CHUNK_SIZE = 128  # chunksize of 256 makes haohao server runs out of CUDA memory!!! 
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # disable tokenizers' parallelism to avoid parallelism bug
# nice ref: https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

task_to_data_splits = {
    "mnli": {"train": "train", "val": "validation_matched", "test": "test_matched"},
    "sst2": {"train": "train", "val": "validation", "test": "test"},
    "mrpc": {"train": "train", "val": "validation", "test": "test"},
}


def get_glue_dataset(dataset_name, args, get_val=True, use_fast_tokenizer=True):
    assert dataset_name in GLUE_DATASETS, f"{dataset_name} not in avail datasets"
    raw_datasets = load_dataset("glue", dataset_name, cache_dir=DATA_ROOT_DIR)  # save in DATA_ROOT_DIR/glue/name
    checkpoint = "bert-base-uncased"  # todo: make this according to args.model
    if not use_fast_tokenizer:
        print("Warning! Not using fast tokenizer!")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=use_fast_tokenizer)
    raw_train_dataset = raw_datasets['train']
    n_classes = raw_train_dataset.features['label'].num_classes

    key1, key2 = task_to_keys[dataset_name]
    remove_cols = [key1, 'idx']
    if key2 is None:
        def tokenize_function(example):
            return tokenizer(example[key1], truncation=True)
    else:
        def tokenize_function(example):
            return tokenizer(example[key1], example[key2], truncation=True)
        remove_cols.append(key2)

    # important to set batched=True when using fast tokenizer! Otherwise much slower due to non-parallelism.
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=remove_cols)
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train = tokenized_datasets[task_to_data_splits[dataset_name]['train']]
    val = tokenized_datasets[task_to_data_splits[dataset_name]['val']]
    test = tokenized_datasets[task_to_data_splits[dataset_name]['test']]
    if not get_val:
        val = None
    return train, val, test, n_classes, data_collator


LOGGING_HEAD = "[get_mlm_dataset]"


def get_mlm_dataset(dataset_name, args, get_val=False, use_fast_tokenizer=True, mlm_probability=0.15):
    assert dataset_name in MLM_DATASETS, f"{dataset_name} not in avail datasets"
    raw_datasets = load_dataset(dataset_name, cache_dir=DATA_ROOT_DIR)  # save in DATA_ROOT_DIR/glue/name
    checkpoint = MODEL_TO_CHECKPOINT_NAME[args.model]
    if not use_fast_tokenizer:
        print(f"{LOGGING_HEAD}Warning! Not using fast tokenizer!")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=use_fast_tokenizer)
    raw_train_dataset = raw_datasets['train']
    n_classes = raw_train_dataset.features['label'].num_classes

    def tokenize_function(examples):
        result = tokenizer(examples["text"])
        if tokenizer.is_fast:
            result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
        return result

    # the map runs the function on batched examples then update the o.g. dict with the returned dict
    print(f"{LOGGING_HEAD} Tokenizing data")
    tokenized_datasets = raw_datasets.map(
        tokenize_function, batched=True, remove_columns=["text", "label"]  # drop the OG labels since we are doing MLM
    )
    chunk_size = min([MAX_CHUNK_SIZE, tokenizer.model_max_length])
    print(f"{LOGGING_HEAD} Setting Chunk_size to {chunk_size}")

    def group_texts(examples):
        # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        # Compute length of concatenated texts
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the last chunk if it's smaller than chunk_size
        total_length = (total_length // chunk_size) * chunk_size
        # Split by chunks of max_len
        result = {
            k: [t[i: i + chunk_size] for i in range(0, total_length, chunk_size)]
            for k, t in concatenated_examples.items()
        }
        # Create a new labels column as a copy since we will be predicting masked tokens
        result["labels"] = result["input_ids"].copy()
        return result

    print(f"{LOGGING_HEAD} Chunking data and creating MLM labels")
    lm_datasets = tokenized_datasets.map(group_texts, batched=True)

    # masking is performed by the following collator
    def whole_word_masking_data_collator(features):
        from transformers import default_data_collator
        for feature in features:
            word_ids = feature.pop("word_ids")
            # Create a map between words and corresponding token indices
            mapping = collections.defaultdict(list)
            current_word_index = -1
            current_word = None
            for idx, word_id in enumerate(word_ids):
                if word_id is not None:
                    if word_id != current_word:
                        current_word = word_id
                        current_word_index += 1
                    mapping[current_word_index].append(idx)
            # Randomly mask words
            mask = np.random.binomial(1, mlm_probability, (len(mapping),))
            input_ids = feature["input_ids"]
            labels = feature["labels"]
            # the labels are all -100 except where the masks are
            new_labels = [-100] * len(labels)
            for word_id in np.where(mask)[0]:
                word_id = word_id.item()
                for idx in mapping[word_id]:
                    new_labels[idx] = labels[idx]
                    input_ids[idx] = tokenizer.mask_token_id
        return default_data_collator(features)

    print(f"{LOGGING_HEAD} Using whole word masking data collator!")
    data_collator = whole_word_masking_data_collator

    train = lm_datasets['train']
    val = None  # todo: fix this for other datasets
    test = lm_datasets['test']
    test = test.train_test_split(test_size=0.2)['test']

    # # finally, we want to fix the masks of our test set so that our results per epoch is more fixed
    # def insert_random_mask(batch):
    #     features = [dict(zip(batch, t)) for t in zip(*batch.values())]
    #     masked_inputs = data_collator(features)
    #     # Create a new "masked" column for each column in the dataset
    #     return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}
    # test.map(insert_random_mask, )
    return train, val, test, n_classes, data_collator


def get_squad_dataset(args, use_fast_tokenizer=True):
    """
    Squad is a question-answering dataset. https://huggingface.co/course/chapter7/7?fw=pt
    :return:
    """
    raw_datasets = load_dataset("squad")
    from transformers import AutoTokenizer
    checkpoint = MODEL_TO_CHECKPOINT_NAME[args.model]
    if not use_fast_tokenizer:
        print(f"[get_squad] Warning! Not using fast tokenizer!")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=use_fast_tokenizer)
    raw_train_dataset = raw_datasets['train']
    n_classes = raw_train_dataset.features['label'].num_classes
    
