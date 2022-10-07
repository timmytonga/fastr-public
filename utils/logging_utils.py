import torch
import wandb
from global_vars import WANDB_LOG_DIR
import sys
from torch.optim import Adam
from optimizers.storm_base_class import StormBaseClass
import os
import math
from tqdm import tqdm
from train import train_utils
import numpy as np
from data.metrics import get_accuracy_metric


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Logger(object):
    def __init__(self, fpath=None, mode="w"):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, mode)

    # def __del__(self):
    #     self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def get_nice_stat_dict_str(stat_dict):
    return "; ".join([f"{k}: {stat_dict[k]:.3f}" for k in stat_dict.keys()])


# todo: make these epoch and training stats logger more general -- enable modifying metrics by defining some template
#  -- do that by passing in a dict of metrics: {'name': metricObject} --> initialize the objects and loop through
#  -- make a dict of datasets and the standard metric(s) used for that dataset
# todo: use load_metric provided from huggingface for NLP datasets
class EpochStatsLogger:
    def __init__(self, epoch, model, optimizer, log_every, wandb_group=None, wandb_logger=None,
                 total_metrics=None, running_metrics=None):
        """
        EpochStatsLogger keeps track of various metrics (passed through metrics_to_compute -- default to accuracy)
        This logs these metrics to wandb
        :param epoch: Current epoch
        :param model: Model to log gradients
        :param optimizer: Optimizer to log learning rate and other steps
        :param log_every: How often to log running stats
        :param wandb_group: Extra keyword to classify which group is this that we are logging (train/val/test/etc.)
        :param wandb_logger: Pass in wandb to use wandb or leave None to not use wandb
        :param total_metrics: List of metrics object. This is important if we want to add extra metrics.
         -- These metrics object should implement 2 methods: add_batch and compute.
         -- The add_batch method should cache the batch by taking in "predictions" and "references" (the labels)
         -- The compute method should consume the cached batches so far and return the computed metric as a dict
           -- this dict should contain the name of the metric along with the final result.
        :param running_metrics: Same as total_metrics but is meant to be logged every "log_every" batches
          -- pass in an empty list [] to not track any additional running_metrics beyond loss.
        """
        self.epoch = epoch
        self.model = model
        self.optimizer = optimizer
        self.log_every = log_every
        self.wandb_group = wandb_group
        self.wandb = wandb_logger
        # we are logging gradients by default
        self.gradients_norm = []  # store the individual batch grad norm. Problem: they are wrt diff models per batch
        self.batch_count = 0  # track number of update called
        # we are logging losses by default
        self.running_loss = 0
        self.total_loss = 0
        # our main metrics
        self.total_metrics = [get_accuracy_metric()] if total_metrics is None else total_metrics
        self.running_metrics = [get_accuracy_metric()] if running_metrics is None else running_metrics
        # some final logging
        self.eop_stat_dict = None

    def log_gradients_batch(self, model):
        result = 0.0
        for name, p in model.named_parameters():
            if p.grad.data is None:
                continue
            g = p.grad.detach()
            result += float(torch.sum(g * g))
        self.gradients_norm.append(result)

    def log_batch(self, outputs, labels, loss, batch_idx):
        # get the prediction
        _, preds = torch.max(outputs.detach(), dim=1)
        # then add batch to each total and running metric
        for metric in self.total_metrics:
            metric.add_batch(predictions=preds, references=labels)
        for metric in self.running_metrics:
            metric.add_batch(predictions=preds, references=labels)
        # loss: this is the batch's average loss
        self.running_loss += loss.item()
        self.total_loss += loss.item()
        # other update
        self.batch_count += 1
        ## log running statistics ##
        if (batch_idx % self.log_every) == (self.log_every - 1):  # print every log_every mini-batches
            stat_dict = {'loss': (self.running_loss / self.log_every)}
            for metric in self.running_metrics:
                stat_dict.update(metric.compute())  # compute() method of a HF metric will consume the cached batch
            stats_str = get_nice_stat_dict_str(stat_dict)
            print(f'[{self.epoch}, {batch_idx + 1:5d}] {stats_str}')
            if self.wandb:
                wandb_stats = {f"{self.wandb_group}/{key}": stat_dict[key] for key in stat_dict.keys()}
                wandb_stats['epoch'] = self.epoch
                wandb_stats['batch_idx'] = batch_idx + 1
                # log step sizes and momentums if available
                step_sizes, momentums = log_step_size_and_momentum(self.optimizer, self.model)
                wandb_stats.update(step_sizes)
                wandb_stats.update(momentums)
                # send this to wandb
                self.wandb.log(wandb_stats)
            self.reset_running_stats()

    def wandb_gradients_logging(self, wandb_stats):
        """
        Constructs a histogram of the gradients norm history throughout this epoch and log the histogram to wandb
         This function updates wandb_stats dict if not None else log the grad_norms directly to wandb (set to None).
        """
        assert self.wandb and len(self.gradients_norm) > 0, "Should only call this when wandb is activated"
        # grad_histogram = np.histogram(self.gradients_norm, bins='auto')
        # create additional table to make wandb custom table
        wandb_table_data = [[self.epoch, gn] for gn in self.gradients_norm]
        my_table = wandb.Table(data=wandb_table_data, columns=['epoch', 'grad_norms'])
        fields = {"value": "grad_norms", "title": f"[Epoch {self.epoch}] Batches' Gradient Norm"}  # vega fields
        plot_histogram = wandb.plot_table(vega_spec_name="fastr/adjustable_histogram", fields=fields,
                                          data_table=my_table)
        if wandb_stats is None:  # this has been called to log the gradnorm directly
            print(f"[Epoch {self.epoch}] Logging wandb gradients...")
            wandb_stats = {'epoch': self.epoch,
                           'eop_grad_norms': plot_histogram}
            self.wandb.log(wandb_stats)
        else:
            # wandb_stats[f'{self.wandb_group}/grad_norms'] = wandb.Histogram(np_histogram=grad_histogram)
            wandb_stats[f'{self.wandb_group}/grad_norms'] = plot_histogram

    def end_of_epoch_logging(self):
        eop_stat_dict = self.get_eop_stat_dict()
        stats_str = get_nice_stat_dict_str(eop_stat_dict)
        print(f'[End of Epoch] {stats_str}')
        if self.wandb:
            wandb_stats = {f"{self.wandb_group}/{key}": eop_stat_dict[key] for key in eop_stat_dict.keys()}
            wandb_stats['epoch'] = self.epoch
            if len(self.gradients_norm) > 0:
                self.wandb_gradients_logging(wandb_stats)
                wandb_stats[f"{self.wandb_group}/grad_sum"] = sum(self.gradients_norm)/len(self.gradients_norm)
                wandb_stats[f"{self.wandb_group}/grad_variance"] = np.var(self.gradients_norm)
            self.wandb.log(wandb_stats)

    def get_eop_stat_dict(self):
        if self.eop_stat_dict is not None:
            return self.eop_stat_dict
        self.eop_stat_dict = {'avg_loss': (self.total_loss / self.batch_count)}
        for metric in self.total_metrics:
            self.eop_stat_dict.update(metric.compute())  # this should be called once -- o.w. there will be errors!
        return self.eop_stat_dict

    # def get_total_accuracy(self):
    #     if self.total_examples_count == 0 and self.total_correct == 0:
    #         # this means that we have not called update batch any time yet
    #         assert self.running_examples_count != 0 and self.running_correct != 0, \
    #             "We call get_total_accuracy without have called update batch any time yet!!! Everything is 0"
    #         self.total_correct, self.total_examples_count = self.running_correct, self.running_examples_count
    #     return self.total_correct / self.total_examples_count

    def reset_running_stats(self):
        self.running_loss = 0.0


class TrainingStatsLogger:  # todo: fix when metrics do not contain accuracy!!!
    """
    Handle tracking things like best accuracy, best loss, etc. in between epoch
    Can also modify later to handle saving best model and such
    """

    def __init__(self, wandb_logger=None, wandb_group=None):
        self.stat_dict = {
            'best_accuracy': -math.inf,
            'best_loss': math.inf,
            'last_accuracy': -math.inf,
            'last_loss': math.inf,
            'best_acc_epoch': None
        }
        self.wandb = wandb_logger
        self.wandb_group = wandb_group

    def update_best_accuracy(self, new_acc, epoch):  # can also update best model here also
        self.stat_dict['last_accuracy'] = new_acc
        if new_acc > self.stat_dict['best_accuracy']:
            self.stat_dict['best_accuracy'] = new_acc
            self.stat_dict['best_acc_epoch'] = epoch

    def update_best_loss(self, new_loss):
        self.stat_dict['last_loss'] = new_loss
        if new_loss < self.stat_dict['best_loss']:
            self.stat_dict['best_loss'] = new_loss

    def update_with_eop_stat_dict(self, eop_stat_dict, epoch):
        if 'accuracy' in eop_stat_dict:
            self.update_best_accuracy(eop_stat_dict['accuracy'], epoch)
        self.update_best_loss(eop_stat_dict['avg_loss'])

    def log_wandb_summary(self):
        """
        logs summary stats to wandb: add in the wandb_group and update the run.summary dict
        """
        assert self.wandb is not None, "wandb not activated!"
        wandb_stat_dict = {f"{self.wandb_group}/{key}": self.stat_dict[key] for key in self.stat_dict.keys()}
        self.wandb.run.summary.update(wandb_stat_dict)

    def end_of_training_logging(self):
        """
        Print end of training stats as well as log to wandb summary
        """
        print(f"[End of training] {self.wandb_group} Summary")
        for k in self.stat_dict:
            print(f"\t{k} = {self.stat_dict[k]}")
        if self.wandb:
            self.log_wandb_summary()


def inspect_lr(optimizer):
    for param_groups in optimizer.param_groups:
        for p in param_groups['params']:
            if p.grad.data is None:
                continue
            state = optimizer.state[p]
            a, b, sum_g = state['a'], state['b'], state['sum_g']
            print(f"a={a}; b={b}; sum_g={sum_g}")
            return


def log_args(args, logger: Logger):
    if type(args) is DotDict:
        argdict = args.items()
    else:
        argdict = vars(args).items()
    for argname, argval in argdict:
        logger.write(f'{argname.replace("_", " ").capitalize()}: {argval}\n')
    logger.write("\n")


def print_args(args):
    if type(args) is DotDict:
        argdict = args.items()
    else:
        argdict = vars(args).items()
    for argname, argval in argdict:
        print(f'{argname.replace("_", " ").capitalize()}: {argval}')


def log_data(data, n_classes):
    # train_data, val_data, test_data = data['train_data'], data['val_data'], data['test_data']
    for t in ['train', 'val', 'test']:
        dataset = data[f'{t}_data']
        if data[f'{t}_data'] is not None:
            print(f"{t}: n={len(dataset)}. No. classes = {n_classes}")


def accuracy(outputs: torch.Tensor, labels: torch.Tensor):
    """
     make sure the outputs and labels are detached before computing to make sure no memory leak!!!
    """
    _, preds = torch.max(outputs.detach(), dim=1)
    return torch.sum(preds == labels.detach()).item() / len(preds)


def nice_float(x):
    return f"{x:.2E}"


def set_wandb_run_name(args):
    lrwd = f"lr{args.lr:.2E}wd{args.weight_decay:.2E}"
    if args.optimizer == "sgd":
        return f"{lrwd}m{args.sgd_momentum}"
    elif args.optimizer in ["adam", "rmsprop", "adamw"]:
        return f"{lrwd}" \
            # f"betas{args.adam_beta1}{args.adam_beta1}eps{args.adam_eps}"
    elif args.optimizer == "adagrad":
        return f"{lrwd}lrdc{args.adagrad_lr_decay}eps{args.adagrad_eps}"
    elif args.optimizer in ["stormplus", "fastrn", "fastrd"]:
        ret_str = f"{lrwd}"
        if args.storm_a_0 == -1:
            ret_str += "a~G"
        elif args.storm_normalized:
            ret_str += 'normalized'
        elif args.storm_dim_normalized:
            ret_str += 'dimNormalized'
        elif args.storm_a_0 < 0:
            raise NotImplementedError
        else:
            ret_str += f"a{nice_float(args.storm_a_0)}"
        ret_str += f"b{nice_float(args.storm_b_0)}"
        if args.storm_ema:
            if args.storm_bias_correction:
                ret_str += f"BC"
            ret_str += f"ema{args.storm_beta1}_{args.storm_beta2}"
        if args.storm_ema_g:
            ret_str += f"ema_g{args.storm_beta1}"
        if args.storm_ema_d:
            ret_str += f"ema_d{args.storm_beta2}"
        ret_str += f"s{args.seed}"
        return ret_str
    elif args.optimizer == "repstormplus":
        ret_str = f"{lrwd}u{args.storm_u}c{args.storm_c}wd{args.weight_decay}"
        ret_str += 'normalized' if args.storm_normalized else ""
        return ret_str
    # elif args.optimizer == "fastrn":
    #     return f"lr{args.lr}a{args.storm_a_0}b{args.storm_b_0}"
    # elif args.optimizer == "fastrd":
    #     return f"lr{args.lr}a{args.storm_a_0}b{args.storm_b_0}"
    else:
        raise NotImplementedError


def initialize_wandb(args):
    group_name = f"{args.optimizer}"
    if args.storm_per_coordinate:
        group_name += "_coord"
    if args.storm_bias_correction:
        group_name += f"BC"
    if args.storm_ema:
        group_name += f"_ema"
    elif args.storm_ema_g:
        group_name += f"_ema_g"
    elif args.storm_ema_d:
        group_name += f"_ema_d"
    job_type = None  # todo
    if args.optimizer in ["fastrn", "fastrd"]:
        job_type = f"{args.storm_p}"
    run_name = set_wandb_run_name(args)
    if not os.path.exists(WANDB_LOG_DIR):
        print(f"{WANDB_LOG_DIR} doesn't exist. Creating one.")
        os.makedirs(WANDB_LOG_DIR)

    os.environ["WANDB_DIR"] = os.path.abspath(WANDB_LOG_DIR)
    run = wandb.init(project=f"{args.project_name}_{args.dataset}",
                     group=group_name,
                     job_type=job_type,
                     name=run_name,
                     dir=WANDB_LOG_DIR,
                     settings=wandb.Settings(start_method="fork"),
                     entity='fastr'  # this is the team's name
                     )
    wandb.config.update(args)
    return run


def get_adam_step_sizes(optim, model):
    if len(optim.param_groups) != 1:
        raise Exception("len(self.param_groups) not 1! Have to deal with this case later...")
    group = optim.param_groups[0]  # 110
    lr = group['lr']  # get this param_group's lr
    eps = group['eps']
    beta1, beta2 = group['betas']
    result = {}
    for name, p in model.named_parameters():
        if p.grad.data is None:
            continue
        state = optim.state[p]
        exp_avg_sq = state['exp_avg_sq']
        step = state['step']
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
        step = lr / bias_correction1
        step_size = step / denom
        result[name] = float(torch.norm(step_size))
    return result


def log_step_size_and_momentum(optimizer, model):
    if isinstance(optimizer, Adam):
        step_size_dict = get_adam_step_sizes(optimizer, model)
        step_sizes = {f"step_size/{key}": step_size_dict[key] for key in step_size_dict.keys()}
        momentums = {}
    elif isinstance(optimizer, StormBaseClass):
        step_size_dict = optimizer.get_params_step_sizes(model)
        step_sizes = {f"step_size/{key}": step_size_dict[key] for key in step_size_dict.keys()}
        momentum_dict = optimizer.get_params_momentum(model)
        momentums = {f"momentum/{key}": momentum_dict[key] for key in momentum_dict.keys()}
        a_0_dict = optimizer.get_params_a0(model)  # this will be empty if args.a_0 != -1
        a_0_dict = {f"a_0/{key}": a_0_dict[key] for key in a_0_dict.keys()}
        momentums.update(a_0_dict)
    else:
        step_sizes, momentums = {}, {}
    return step_sizes, momentums


def log_grads(model):
    result = {}
    for name, p in model.named_parameters():
        if p.grad.data is None:
            continue
        g = p.grad.detach()
        result[name] = float(torch.sum(g * g))
    result = {f"grads/{key}": result[key] for key in result.keys()}
    return result


def log_epoch_gradients(epoch, model, optimizer, data_loader, criterion, device, log_every,
                        show_progress=False, wandb_group=None, wandb_logger=None, is_bert=False):
    """
    Log the initial gradients
    """
    epoch_stats_logger = EpochStatsLogger(epoch, model, optimizer, log_every, wandb_group=wandb_group,
                                          wandb_logger=wandb_logger)

    loader = tqdm(data_loader) if show_progress else data_loader
    optimizer.zero_grad()
    with torch.set_grad_enabled(True):
        for batch in loader:
            outputs, loss, inputs, labels = train_utils.process_batch_and_get_outputs_and_loss(
                model, device, criterion, batch, is_bert)
            loss.backward()
            # then log the gradients
            epoch_stats_logger.log_gradients_batch(model)
            optimizer.zero_grad()

    # push to wandb
    epoch_stats_logger.wandb_gradients_logging(wandb_stats=None)  # None here so that we are logging
