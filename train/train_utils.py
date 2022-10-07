import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import set_seed


from global_vars import NO_SEED  # all the fully caps vars


def set_seed_all(seed, hf_set_seed=False):
    """Sets seed"""
    if seed == NO_SEED:
        print("Not setting a deterministic seed!")
        return
    if hf_set_seed:
        set_seed(seed)  # HF transformer set seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_criterion(loss_name):
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    else:
        raise NotImplementedError


def get_optimizer(optimizer_name, model, args):
    from optimizers.stormplus import StormPlus
    from optimizers.fastr_n import FastrN
    from optimizers.fastr_d import FastrD
    from optimizers.stormplus_reproduce import STORMplus
    if optimizer_name == "sgd":
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                         momentum=args.sgd_momentum, nesterov=args.sgd_nesterov)
    elif optimizer_name == "adagrad":
        return optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                             lr_decay=args.adagrad_lr_decay, eps=args.adagrad_eps)
    elif optimizer_name == "adam":
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                          betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_eps)
    elif optimizer_name == "adamw":
        return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif optimizer_name == "rmsprop":
        return optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif optimizer_name == "stormplus":
        return StormPlus(model.parameters(), args)
    elif optimizer_name == "fastrn":
        return FastrN(model.parameters(), args)
    elif optimizer_name == "fastrd":
        return FastrD(model.parameters(), args)
    elif optimizer_name == "repstormplus":
        return STORMplus(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                         init_accumulator=args.storm_u,
                         c=args.storm_c, normalized_norm=args.storm_normalized)
    else:
        raise NotImplementedError


def get_scheduler(args, train_dataloader, optimizer):
    from transformers import get_scheduler
    lr_scheduler = get_scheduler(
        args.scheduler,
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=args.n_epochs * len(train_dataloader),
    )
    return lr_scheduler


def process_batch_and_get_outputs_and_loss(model, device, criterion, batch, is_bert: bool):
    """
    Return
    """
    if not is_bert:  # a normal model
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
    else:  # BERT models require some extra inputs
        batch = {k: v.to(device) for k, v in batch.items()}
        inputs, labels = batch['input_ids'], batch['labels']
        outputs_bert = model(**batch)  # bert models require additional info like attention masks etc.
        outputs, loss = outputs_bert.logits, outputs_bert.loss
    assert -1 not in labels, "Warning! There is a -1 in the labels. This might be because the test set has no labels."
    return outputs, loss, inputs, labels
