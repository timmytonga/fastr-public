from tqdm import tqdm
from utils.logging_utils import EpochStatsLogger
import torch
from optimizers.stormplus_reproduce import STORMplus
from train import train_utils


def reproduce_stormplus_run_epoch(epoch, model, optimizer,
                                  data_loader, criterion,
                                  is_training, device, is_bert=False,
                                  # need to do things a bit differently for BERT models
                                  total_metrics=None, running_metrics=None,
                                  lr_scheduler=None, log_every=2000, show_progress=False,
                                  wandb_group=None, wandb_logger=None, log_gradients=False):
    if is_training:
        assert isinstance(optimizer, STORMplus), "Can only run this run_epoch with STORMplus reproduce"
        model.train()
    else:
        model.eval()

    # intialize the stats logger: helps log accuracy and loss and send to wandb
    epoch_stats_logger = EpochStatsLogger(epoch, model, optimizer, log_every,
                                          wandb_group=wandb_group, wandb_logger=wandb_logger,
                                          total_metrics=total_metrics, running_metrics=running_metrics)

    loader = tqdm(data_loader) if show_progress else data_loader
    # ONLY FOR THE FIRST BATCH: We need to compute the initial estimator d_1,
    # which is the first (mini-batch) stochastic gradient g_1. To set the estimator
    # we need to call compute_step() with the first batch.
    if is_training:
        batch = next(iter(data_loader))
        optimizer.zero_grad()
        outputs, loss, inputs, labels = train_utils.process_batch_and_get_outputs_and_loss(
            model, device, criterion, batch, is_bert)
        loss.backward()
        optimizer.compute_estimator()  # optimizer.compute_estimator(normalized_norm=True)

    with torch.set_grad_enabled(is_training):  # to make sure we don't save grad when val
        for batch_idx, batch in enumerate(loader, 0):
            # main optimization step
            outputs, loss, inputs, labels = train_utils.process_batch_and_get_outputs_and_loss(
                model, device, criterion, batch, is_bert)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                # uses \tilde g_t from the backward() call above
                # uses d_t already saved as parameter group state from previous iteration
                optimizer.step()
                if lr_scheduler:
                    lr_scheduler.step()

                # makes the second pass, backpropagation for the NEXT iterate using the current data batch
                optimizer.zero_grad()
                outputs, loss, inputs, labels = train_utils.process_batch_and_get_outputs_and_loss(
                    model, device, criterion, batch, is_bert)
                loss.backward()

                # updates estimate d_{t+1} for the next iteration, saves g_{t+1} for next iteration
                optimizer.compute_estimator()  # optimizer.compute_estimator(normalized_norm=True)

            epoch_stats_logger.log_batch(outputs, labels, loss, batch_idx)
    # end of epoch logging
    epoch_stats_logger.end_of_epoch_logging()  # end of epoch logging (print and log to wandb)
    return epoch_stats_logger.get_eop_stat_dict()