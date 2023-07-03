import argparse
from accelerate.logging import get_logger
import os
from pathlib import Path
import torch
from PIL import Image
import numpy as np

logger = get_logger(__name__)


def parse_args(default=False):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )

    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing an image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="caption",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="log/fine_generator",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )

    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--use_ema", action="store_true", help="Whether to use EMA model."
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )

    # added arguments
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
    )
    parser.add_argument(
        "--train_image_encoder",
        action="store_true",
    )

    parser.add_argument(
        "--keep_only_last_checkpoint",
        action="store_true",
    )

    parser.add_argument(
        "--keep_interval",
        type=int,
        default=None,
    )

    # inference specific arguments
    parser.add_argument("--inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=int, default=5)
    parser.add_argument("--num_images_per_prompt", type=int, default=1)
    parser.add_argument("--evaluation_batch_size", type=int, default=4)
    parser.add_argument("--finetuned_model_path", type=str)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=50)
    parser.add_argument(
        "--text_prompt_only", action="store_true", help="disable all image conditioning"
    )
    parser.add_argument(
        "--use_multiple_conditioning",
        action="store_true",
        help="use multiple conditioning images",
    )
    parser.add_argument(
        "--start_merge_step",
        type=int,
        default=0,
        help="when to start merging noise prediction from multiple conditioning source",
    )

    parser.add_argument(
        "--image_encoder_type",
        type=str,
        default="clip",
        choices=["clip", "vae"],
    )

    parser.add_argument(
        "--image_encoder_name_or_path",
        type=str,
        default="openai/clip-vit-large-patch14",
    )

    parser.add_argument(
        "--num_image_tokens",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--max_num_objects",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--train_resolution",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--object_resolution",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--test_resolution",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--generate_width",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--generate_height",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--object_appear_prob",
        type=float,
        default=1,
    )

    parser.add_argument(
        "--no_object_augmentation",
        action="store_true",
    )

    parser.add_argument(
        "--image_encoder_trainable_layers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--load_model",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--uncondition_prob",
        type=float,
        default=0,
    )

    parser.add_argument(
        "--text_only_prob",
        type=float,
        default=0,
    )

    parser.add_argument(
        "--text_encoder_use_lora",
        action="store_true",
    )

    parser.add_argument(
        "--lora_text_encoder_r",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--lora_text_encoder_alpha",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--lora_text_encoder_dropout",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--lora_text_encoder_bias",
        type=str,
        default="none",
    )

    parser.add_argument(
        "--image_encoder_use_lora",
        action="store_true",
    )

    parser.add_argument(
        "--lora_image_encoder_r",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--lora_image_encoder_alpha",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--lora_image_encoder_dropout",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--lora_image_encoder_bias",
        type=str,
        default="none",
    )

    parser.add_argument(
        "--unet_use_lora",
        action="store_true",
    )

    parser.add_argument(
        "--unet_lora_alpha",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--num_rows", type=int, default=1, help="number of rows in the output image"
    )

    # testing specific arguments
    parser.add_argument(
        "--test_caption",
        type=str,
        help="caption for testing. Use <|image|> to specify image prompt",
    )

    parser.add_argument(
        "--test_reference_folder",
        type=str,
        help="folder containing reference images for testing. Name of the image should be ordered index. e.g. 0.png, 1.png, 2.png, etc.",
    )

    parser.add_argument(
        "--load_merged_lora_model",
        action="store_true",
    )

    parser.add_argument(
        "--object_background_processor",
        type=str,
        default="random",
    )

    parser.add_argument("--disable_flashattention", action="store_true")

    parser.add_argument("--object_types", default=None, type=str)

    parser.add_argument("--object_localization", action="store_true")
    parser.add_argument("--localization_layers", type=int, default=5)
    parser.add_argument("--object_localization_weight", type=float, default=0.01)
    parser.add_argument("--object_localization_loss", type=str, default="balanced_l1")
    parser.add_argument("--object_localization_threshold", type=float, default=1.0)
    parser.add_argument("--object_localization_normalize", action="store_true")
    parser.add_argument("--unet_lr_scale", type=float, default=1.0)

    parser.add_argument(
        "--val_dataset_name",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--mask_loss",
        action="store_true",
    )

    parser.add_argument(
        "--mask_loss_prob",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--freeze_unet",
        action="store_true",
    )

    parser.add_argument(
        "--use_multiple_datasets",
        action="store_true",
    )

    parser.add_argument(
        "--num_datasets",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--min_num_objects",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--dataset_type",
        type=str,
        default="original",
        choices=["original", "retrieval"],
    )

    parser.add_argument(
        "--retrieval_identity_path",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--dataset_name1",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--dataset_name2",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--dataset_name3",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--dataset_type1",
        type=str,
        default="original",
        choices=["original", "retrieval"],
    )

    parser.add_argument(
        "--dataset_type2",
        type=str,
        default="original",
        choices=["original", "retrieval"],
    )

    parser.add_argument(
        "--dataset_type3",
        type=str,
        default="original",
        choices=["original", "retrieval"],
    )

    parser.add_argument(
        "--retrieval_identity_path1",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--retrieval_identity_path2",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--retrieval_identity_path3",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--object_localization_skip_special_tokens",
        action="store_true",
    )

    parser.add_argument(
        "--balance_num_objects",
        action="store_true",
    )

    parser.add_argument(
        "--inference_split",
        type=str,
        default="eval",
    )

    parser.add_argument(
        "--num_batches",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--text_image_linking",
        type=str,
        default="postfuse",
    )

    parser.add_argument("--freeze_postfuse_module", action="store_true")

    if default:
        return parser.parse_args([])

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args
