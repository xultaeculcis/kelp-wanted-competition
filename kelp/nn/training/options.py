from __future__ import annotations

import argparse

from kelp import consts
from kelp.nn.training.config import TrainConfig


def parse_args() -> TrainConfig:
    """
    Parse command line arguments.

    Returns: An instance of TrainConfig.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--metadata_fp",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--dataset_stats_fp",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="kelp-seg-training-exp",
    )
    parser.add_argument(
        "--cv_split",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--sahi",
        choices=["True", "False"],
        type=str,
        default="False",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=352,
    )
    parser.add_argument(
        "--resize_strategy",
        type=str,
        choices=["pad", "resize"],
        default="pad",
    )
    parser.add_argument(
        "--interpolation",
        type=str,
        choices=["nearest", "nearest-exact", "bilinear", "bicubic"],
        default="nearest",
    )
    parser.add_argument(
        "--normalization_strategy",
        type=str,
        choices=[
            "min-max",
            "quantile",
            "per-sample-min-max",
            "per-sample-quantile",
            "z-score",
        ],
        default="quantile",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--fill_missing_pixels_with_torch_nan",
        choices=["True", "False"],
        type=str,
        default="False",
    )
    parser.add_argument(
        "--mask_using_qa",
        choices=["True", "False"],
        type=str,
        default="False",
    )
    parser.add_argument(
        "--mask_using_water_mask",
        choices=["True", "False"],
        type=str,
        default="False",
    )
    parser.add_argument(
        "--use_weighted_sampler",
        choices=["True", "False"],
        type=str,
        default="False",
    )
    parser.add_argument(
        "--samples_per_epoch",
        type=int,
        default=10240,
    )
    parser.add_argument(
        "--has_kelp_importance_factor",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--kelp_pixels_pct_importance_factor",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--qa_ok_importance_factor",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--qa_corrupted_pixels_pct_importance_factor",
        type=float,
        default=-1.0,
    )
    parser.add_argument(
        "--almost_all_water_importance_factor",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--dem_nan_pixels_pct_importance_factor",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--dem_zero_pixels_pct_importance_factor",
        type=float,
        default=-1.0,
    )
    parser.add_argument(
        "--spectral_indices",
        type=str,
        help="A comma separated list of spectral indices to append to the samples during training",
    )
    parser.add_argument(
        "--bands",
        type=str,
        help="A comma separated list of band names to reorder. Use it to shift input data channels. "
        f"Must be a subset of {consts.data.ORIGINAL_BANDS} if specified.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--architecture",
        type=str,
        required=True,
        choices=[
            "deeplabv3",
            "deeplabv3+",
            "efficientunet++",
            "fcn",
            "fpn",
            "linknet",
            "manet",
            "pan",
            "pspnet",
            "resunet",
            "resunet++",
            "unet",
            "unet++",
        ],
    )
    parser.add_argument(
        "--encoder",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--pretrained",
        choices=["True", "False"],
        type=str,
        default="False",
    )
    parser.add_argument(
        "--encoder_weights",
        type=str,
    )
    parser.add_argument(
        "--decoder_channels",
        type=str,
        default="256,128,64,32,16",
    )
    parser.add_argument(
        "--decoder_attention_type",
        type=str,
    )
    parser.add_argument(
        "--plot_n_batches",
        type=str,
        default=3,
    )
    parser.add_argument(
        "--objective",
        type=str,
        choices=["binary", "multiclass"],
        default="binary",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adam", "adamw", "sgd"],
        default="adamw",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        choices=["onecycle", "cosine", "cosine_with_warm_restarts", "cyclic", "reduce_lr_on_plateau", "none"],
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        "--onecycle_pct_start",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--onecycle_div_factor",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--onecycle_final_div_factor",
        type=float,
        default=1e2,
    )
    parser.add_argument(
        "--cyclic_base_lr",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--cyclic_mode",
        type=str,
        choices=["triangular", "triangular2", "exp_range"],
        default="exp_range",
    )
    parser.add_argument(
        "--cosine_eta_min",
        type=float,
        default=1e-7,
    )
    parser.add_argument(
        "--cosine_T_mult",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--reduce_lr_on_plateau_factor",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--reduce_lr_on_plateau_patience",
        type=float,
        default=3,
    )
    parser.add_argument(
        "--reduce_lr_on_plateau_threshold",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--reduce_lr_on_plateau_min_lr",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--tta",
        choices=["True", "False"],
        type=str,
        default="False",
    )
    parser.add_argument(
        "--tta_merge_mode",
        type=str,
        choices=["min", "max", "mean", "gmean", "sum", "tsharpen"],
        default="max",
    )
    parser.add_argument(
        "--decision_threshold",
        type=float,
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=[
            "ce",
            "jaccard",
            "dice",
            "tversky",
            "focal",
            "lovasz",
            "soft_ce",
            "xedice",
            "focal_tversky",
            "log_cosh_dice",
            "hausdorff",
            "t_loss",
            "combo",
            "exp_log_loss",
            "soft_dice",
            "batch_soft_dice",
        ],
        default="dice",
    )
    parser.add_argument(
        "--monitor_metric",
        type=str,
        default="val/dice",
    )
    parser.add_argument(
        "--ce_smooth_factor",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--ce_class_weights",
        type=str,
    )
    parser.add_argument(
        "--monitor_mode",
        type=str,
        default="max",
    )
    parser.add_argument(
        "--ort",
        choices=["True", "False"],
        type=str,
        default="False",
    )
    parser.add_argument(
        "--compile",
        choices=["True", "False"],
        type=str,
        default="False",
    )
    parser.add_argument(
        "--compile_dynamic",
        choices=["True", "False"],
        type=str,
        default="False",
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
        default="default",
    )
    parser.add_argument(
        "--save_top_k",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--swa",
        choices=["True", "False"],
        type=str,
        default="False",
    )
    parser.add_argument(
        "--swa_epoch_start",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--swa_annealing_epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--swa_lr",
        type=float,
        default=3e-5,
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=[
            "16-true",
            "16-mixed",
            "bf16-true",
            "bf16-mixed",
            "32-true",
        ],
        default="16-mixed",
    )
    parser.add_argument(
        "--fast_dev_run",
        choices=["True", "False"],
        type=str,
        default="False",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--limit_train_batches",
        type=float,
    )
    parser.add_argument(
        "--limit_val_batches",
        type=float,
    )
    parser.add_argument(
        "--limit_test_batches",
        type=float,
    )
    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--val_check_interval",
        type=float,
    )
    parser.add_argument(
        "--benchmark",
        choices=["True", "False"],
        type=str,
        default="False",
    )
    args = parser.parse_args()
    cfg = TrainConfig(**vars(args))
    cfg.log_self()
    cfg.output_dir.mkdir(exist_ok=True, parents=True)
    return cfg
