from global_vars import AVAIL_OPTIMIZERS, AVAIL_SCHEDULERS, AVAIL_LOSSES, AVAIL_DATASETS, AVAIL_MODELS, \
    PROJECT_NAME, default_data_setting_dict, default_args_dict
from fractions import Fraction


def add_all_args(parser):
    add_data_args(parser)
    add_model_args(parser)
    add_misc_args(parser)
    add_optimization_args(parser)
    add_settings_args(parser)
    add_optimizers_args(parser)  # this adds the algorithm specific args for sgd, adam, etc.


def add_optimizers_args(parser):
    add_adam_args(parser)
    add_sgd_args(parser)
    add_adagrad_args(parser)
    add_storm_based_args(parser)


def add_storm_based_args(parser):
    """
    These params are used for all STORM based methods: storm+, fastr, etc.
    """
    parser.add_argument("--storm_ema", action="store_true", default=False,
                        help="turn on ema for everything (currently equiv to both ema_g and ema_d)")
    parser.add_argument("--storm_ema_g", action="store_true", default=False,
                        help="turn on ema for sum_g")
    parser.add_argument("--storm_ema_d", action="store_true", default=False,
                        help="turn on ema for sum_d")
    parser.add_argument("--storm_per_coordinate", action="store_true", default=False,
                        help="turn on update per coordinate")
    parser.add_argument("--storm_a_0", type=float, default=1,
                        help="Scale the sum of gradient norm (or gradient diff)."
                             "Set this to -1 to set a_0 to estimate G (bound on stochastic grad)"
                             "Else set this to <0 to set a_0 be proportional to the first gradient.")
    parser.add_argument("--storm_eps", type=float, default=1e-10,
                        help="")
    parser.add_argument("--storm_b_0", type=float, default=1e-8)
    parser.add_argument("--storm_beta1", type=float, default=0.99,
                        help="ema for sum_g")
    parser.add_argument("--storm_beta2", type=float, default=0.99,
                        help="ema for sum_d")
    parser.add_argument("--storm_bias_correction", action="store_true", default=False,
                        help="Enables adam-like bias correction when EMA is set")
    parser.add_argument("--storm_p", type=str, default='1/2',
                        help="This sets the fraction p for b_t. Valid range: [0.1771, 0.5] for fastrd "
                             "and [0.25, 0.5] for fastrn.")
    # reproduce
    parser.add_argument("--storm_u", type=float, default=1.,
                        help="init_accumulator for stormplus reproduced")
    parser.add_argument("--storm_c", type=float, default=1.,
                        help="numerator for a_t for stormplus reproduced")
    parser.add_argument("--storm_normalized", action='store_true', default=False,
                        help="Set a_0 to number of params.")  # todo: this is not right... below is more correct
    parser.add_argument("--storm_dim_normalized", action='store_true', default=False,
                        help="Divide sum_g and sum_d by the number of parameters i.e. prod(param.shape)")


def add_adagrad_args(parser):
    parser.add_argument("--adagrad_lr_decay", type=float, default=0)
    parser.add_argument("--adagrad_eps", type=float, default=1e-10)


def add_adam_args(parser):
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_eps", type=float, default=1e-8)


def add_sgd_args(parser):
    parser.add_argument("--sgd_momentum", type=float, default=0.9)
    parser.add_argument("--sgd_nesterov", action="store_true", default=False)


def add_optimization_args(parser):
    parser.add_argument("--optimizer", choices=AVAIL_OPTIMIZERS, default=None)
    parser.add_argument("--loss", default=None, choices=AVAIL_LOSSES)
    parser.add_argument("--n_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None, help=f"Learning Rate. Default is {default_args_dict['lr']}.")
    parser.add_argument("--weight_decay", type=float, default=None,
                        help=f"L2 Regularization. Default is {default_args_dict['weight_decay']}.")
    parser.add_argument("--scheduler", choices=AVAIL_SCHEDULERS, default=None)


def add_model_args(parser):
    # Model
    parser.add_argument("--model",
                        choices=AVAIL_MODELS,
                        default=None)
    parser.add_argument("--use_pretrained",
                        action="store_true",
                        default=False)


def add_settings_args(parser):
    # wandb: wandb.ai
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--project_name", type=str,
                        default=PROJECT_NAME, help="wandb project name. modify default in global_vars.py")
    # Resume?
    parser.add_argument("--resume", default=False, action="store_true")


def add_data_args(parser):
    # Settings
    parser.add_argument("--dataset", choices=AVAIL_DATASETS, default='cifar10')
    parser.add_argument("--no_augment_data", action="store_true", default=False)
    parser.add_argument("--val_fraction", type=float, default=0)
    parser.add_argument("--no_test", action="store_true", default=None, help="Do not evaluate on test set")


def add_misc_args(parser):
    # Misc
    parser.add_argument("--seed", type=int, default=None, help=f"Set deterministic seed. "
                                                                                    f"Default: random seed")
    parser.add_argument("--show_progress", default=False, action="store_true")
    parser.add_argument("--log_dir", default="AUTO",
                        help="Location to save logging related files and checkpoints. Default: Auto dir")
    parser.add_argument("--log_every", default=None, type=int, help=f"Log every step. "
                                                                    f"Default:{default_args_dict['log_every']}.")
    parser.add_argument("--log_gradients", default=False, action="store_true",
                        help="Log gradients of batches during training epoch")
    parser.add_argument("--log_fixed_gradients_n_epochs", default=0, type=int,
                        help="The frequency (in epoch) to log gradients of batches as a epoch on its own."
                             "Set this to > 0 to activate. For example, 1 would mean log gradients every epoch."
                             "Enabling this would always log an epoch of gradients before training begins."
                             "This will be more time consuming than log_gradients because of the additional epochs.")

    # parser.add_argument("--save_every", type=int, default=200)
    # parser.add_argument("--save_best", action="store_true", default=False)
    parser.add_argument("--save_last", action="store_true", default=False)
    parser.add_argument('--gpu', type=int, default=0, help="Set gpu to -1 to use cpu instead")


def check_and_update_args(args):
    args.storm_p = Fraction(args.storm_p)
    assert 0 < args.storm_p <= 0.5
    set_default_args(args)  # this set any default args if applicable


def set_default_args(args):
    for k, v in filter(lambda elem: elem[1] is None, vars(args).items()):  #
        assert k in default_args_dict, f"{k} (value None) should be in default_args_dict! Something is wrong!"
        vars(args)[k] = default_args_dict[k]
        # if we have dataset specific settings then set to that instead
        if args.dataset in default_data_setting_dict and k in default_data_setting_dict[args.dataset]:
            print(f"Setting {k} to {default_data_setting_dict[args.dataset][k]}.")
            vars(args)[k] = default_data_setting_dict[args.dataset][k]
