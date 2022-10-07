import torch
from tqdm import tqdm
import os
from utils import logging_utils
from optimizers.storm_base_class import StormBaseClass
from train import train_utils
from data.metrics import get_metrics_for_dataset

from train.reproduce_storm_plus import reproduce_stormplus_run_epoch


def train(model, datasets, args, device, wandb=None):
    train_loader, val_loader, test_loader = datasets['train_loader'], datasets['val_loader'], datasets['test_loader']
    if val_loader is None:
        print("[WARNING] No Validation")
    if test_loader is None:
        print("[WARNING] No Test")

    # get the optimizer, criterion, and scheduler
    optimizer = train_utils.get_optimizer(optimizer_name=args.optimizer, model=model, args=args)
    criterion = train_utils.get_criterion(args.loss)
    scheduler = train_utils.get_scheduler(args, train_loader, optimizer=optimizer) if args.scheduler is not None \
        else None

    run_epoch = run_epoch_standard  # we can set this to some other run_epoch depending on usage.
    if args.optimizer == "repstormplus":
        run_epoch = reproduce_stormplus_run_epoch

    is_bert = True if "bert" in args.model else False
    total_metrics = get_metrics_for_dataset(args.dataset)
    running_metrics = get_metrics_for_dataset(args.dataset)  # todo: make this more sophisticate???
    log_gradients = args.log_gradients

    standard_kwargs = {  # these are args to run a typical epoch for training, evaluating, and logging
        'epoch': 0, 'model': model, 'optimizer': optimizer, 'criterion': criterion, 'is_bert': is_bert,
        'log_every': args.log_every, 'show_progress': args.show_progress, 'wandb_logger': wandb, 'device': device}
    run_epoch_kwargs = {**standard_kwargs, 'log_gradients': log_gradients, 'total_metrics': total_metrics,
                        'running_metrics': running_metrics}  # additional args for training
    # we log the gradients at initialization
    if args.log_fixed_gradients_n_epochs != 0 and wandb:
        print("Logging initial gradients...")
        logging_utils.log_epoch_gradients(data_loader=train_loader, wandb_group="log", **standard_kwargs)
        print("Done logging. Begin Training...")

    # init some stats logger
    train_stats_logger = logging_utils.TrainingStatsLogger(wandb_logger=wandb, wandb_group='train')
    val_stats_logger = logging_utils.TrainingStatsLogger(wandb_logger=wandb, wandb_group='val') \
        if val_loader is not None else None
    test_stats_logger = logging_utils.TrainingStatsLogger(wandb_logger=wandb, wandb_group='test') \
        if test_loader is not None else None
    for epoch in range(1, args.n_epochs + 1):
        run_epoch_kwargs['epoch'] = epoch
        print(f"[Epoch {epoch}] Train")
        eop_stat_dict = run_epoch(data_loader=train_loader, is_training=True, lr_scheduler=scheduler,
                                  wandb_group="train", **run_epoch_kwargs)
        train_stats_logger.update_with_eop_stat_dict(eop_stat_dict, epoch)
        if val_loader is not None:
            print(f"[Epoch {epoch}] Val")
            eop_stat_dict = run_epoch(data_loader=val_loader, is_training=False, wandb_group="val", **run_epoch_kwargs)
            val_stats_logger.update_with_eop_stat_dict(eop_stat_dict, epoch)
        if test_loader is not None:
            print(f"[Epoch {epoch}] Test")
            eop_stat_dict = run_epoch(data_loader=test_loader, is_training=False, wandb_group="test",
                                      **run_epoch_kwargs)
            test_stats_logger.update_with_eop_stat_dict(eop_stat_dict, epoch)

        # we log the gradients per every so often
        if args.log_fixed_gradients_n_epochs != 0 and epoch % args.log_fixed_gradients_n_epochs == 0:
            print(f"[Epoch {epoch}] Logging gradients...")
            logging_utils.log_epoch_gradients(data_loader=train_loader, wandb_group="log", **standard_kwargs)
            print(f"[Epoch {epoch}] Done logging gradients. ")

    train_stats_logger.end_of_training_logging()
    if val_loader is not None:
        val_stats_logger.end_of_training_logging()
    if test_loader is not None:
        test_stats_logger.end_of_training_logging()

    if args.save_last:
        torch.save(model, os.path.join(args.log_dir, "last.pth"))


def run_epoch_standard(epoch, model, optimizer,
                       data_loader, criterion,
                       is_training, device, is_bert=False,  # need to do things a bit differently for BERT models
                       total_metrics=None, running_metrics=None,
                       lr_scheduler=None, log_every=2000, show_progress=False,
                       wandb_group=None, wandb_logger=None, log_gradients=False) -> dict:
    """
    :return: a dictionary containing relevant stats like total accuracy and average loss throughout epoch
    """
    if is_training:
        model.train()
    else:
        model.eval()

    # intialize the stats logger: helps log accuracy and loss and send to wandb
    epoch_stats_logger = logging_utils.EpochStatsLogger(epoch, model, optimizer, log_every,
                                                        wandb_group=wandb_group, wandb_logger=wandb_logger,
                                                        total_metrics=total_metrics, running_metrics=running_metrics)

    loader = tqdm(data_loader) if show_progress else data_loader
    with torch.set_grad_enabled(is_training):  # to make sure we don't save grad when val
        for batch_idx, batch in enumerate(loader):
            # the next line extract the inputs and labels from the batch as well as obtaining outputs from the model
            # this process differs depending on whether the model is BERT or not.
            outputs, loss, inputs, labels = train_utils.process_batch_and_get_outputs_and_loss(
                model, device, criterion, batch, is_bert)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if lr_scheduler:
                    lr_scheduler.step()

                if log_gradients:
                    epoch_stats_logger.log_gradients_batch(model)

                # for Storm-based methods, we need to get the gradients of the updated weight for the next iter
                if isinstance(optimizer, StormBaseClass):
                    optimizer.zero_grad()  # don't want to accumulate the grad from the last loss.backward()
                    # first get the gradients of the updated weight
                    outputs, loss, inputs, labels = train_utils.process_batch_and_get_outputs_and_loss(
                        model, device, criterion, batch, is_bert)
                    loss.backward()
                    # then update the momentum: this sets the correct d_t and a_t for the next optimizer.step()
                    optimizer.update_momentum_and_lr()

            # log stats (accuracy, loss, etc.) for this batch
            epoch_stats_logger.log_batch(outputs, labels, loss, batch_idx)

    epoch_stats_logger.end_of_epoch_logging()  # end of epoch logging (print and log to wandb)
    return epoch_stats_logger.get_eop_stat_dict()
