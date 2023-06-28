# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
from typing import List
import argparse
import numpy as np
import torch
from collections import OrderedDict
from torchvision import transforms as T
from accelerate import Accelerator
from transformers import CLIPTokenizer
from PIL import Image
from cog import BasePredictor, Input, Path

from fastcomposer.pipeline import StableDiffusionFastCompposerPipeline
from fastcomposer.model import FastComposerModel
from fastcomposer.utils import parse_args
from fastcomposer.transforms import PadToSquare


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        args = load_default_args()
        accelerator = Accelerator(mixed_precision="fp16")
        device = accelerator.device

        weight_dtype = torch.float32
        if args.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif args.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        model = FastComposerModel.from_pretrained(args)
        ckpt_file = "model/fastcomposer/pytorch_model.bin"
        model.load_state_dict(torch.load(ckpt_file, map_location="cpu"))
        model = model.to(dtype=weight_dtype, device=device)

        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
        )
        tokenizer.add_tokens(["img"], special_tokens=True)
        image_token_id = tokenizer.convert_tokens_to_ids("img")

        self.pipe = StableDiffusionFastCompposerPipeline.from_pretrained(
            args.pretrained_model_name_or_path, torch_dtype=weight_dtype
        ).to(device)

        self.pipe.object_transforms = torch.nn.Sequential(
            OrderedDict(
                [
                    ("pad_to_square", PadToSquare(fill=0, padding_mode="constant")),
                    (
                        "resize",
                        T.Resize(
                            (args.object_resolution, args.object_resolution),
                            interpolation=T.InterpolationMode.BILINEAR,
                            antialias=True,
                        ),
                    ),
                    ("convert_to_float", T.ConvertImageDtype(torch.float32)),
                ]
            )
        )
        self.pipe.unet = model.unet
        self.pipe.text_encoder = model.text_encoder
        self.pipe.postfuse_module = model.postfuse_module
        self.pipe.image_encoder = model.image_encoder
        self.pipe.image_token_id = image_token_id
        self.pipe.special_tokenizer = tokenizer

        del model

    def predict(
        self,
        image1: Path = Input(description="First input image"),
        image2: Path = Input(
            description="Second input image, optional", default=None),
        prompt: str = Input(
            description='Input proper text prompts, such as "A woman img and a man img in the snow" or "A painting of a man img in the style of Van Gogh", where "img" specifies the token you want to augment and comes after the word.',
            default="A man img and a man img singing in the park together.",
        ),
        alpha: float = Input(
            description="A smaller alpha aligns images with text better, but may deviate from the subject image. Increase alpha to improve identity preservation, decrease it for prompt consistency.",
            default=0.7,
            ge=0,
            le=1,
        ),
        num_steps: int = Input(
            description="Number of diffusion steps", default=50, ge=1, le=300
        ),
        num_images_per_prompt: int = Input(
            description="Number of output images. Lower this setting if OOM.",
            default=1,
            ge=1,
            le=4,
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance.",
            ge=1.5,
            le=50,
            default=5.0,
        ),
        width: int = Input(
            description="Width of output image.",
            default=512,
        ),
        height: int = Input(
            description="Height of output image.",
            default=512,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed.", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        image = []
        for img in [image1, image2]:
            if img:
                image.append(Image.open(str(img)))

        assert len(image) > 0, "You need to upload at least one image."

        num_subject_in_text = (
            np.array(self.pipe.special_tokenizer.encode(prompt))
            == self.pipe.image_token_id
        ).sum()
        assert num_subject_in_text == len(image), (
            f"Number of subjects in the text description doesn't match the number of reference images, #text subjects:"
            f" {num_subject_in_text} #reference image: {len(image)}"
        )

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        generator = torch.Generator("cuda").manual_seed(seed)

        images = self.pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            alpha_=alpha,
            reference_subject_images=image,
        ).images

        output = []
        for i, img in enumerate(images):
            out = f"/tmp/out_{i}.png"
            img.save(out)
            output.append(Path(out))

        return output


def load_default_args():
    return argparse.Namespace(
        pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
        revision=None,
        dataset_name=None,
        dataset_config_name=None,
        train_data_dir=None,
        image_column="image",
        caption_column="caption",
        max_train_samples=None,
        output_dir="log/fine_generator",
        cache_dir=None,
        seed=None,
        center_crop=False,
        random_flip=False,
        train_batch_size=16,
        num_train_epochs=100,
        max_train_steps=None,
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        learning_rate=0.0001,
        scale_lr=False,
        lr_scheduler="constant",
        lr_warmup_steps=500,
        use_8bit_adam=False,
        allow_tf32=False,
        use_ema=False,
        non_ema_revision=None,
        dataloader_num_workers=0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_weight_decay=0.01,
        adam_epsilon=1e-08,
        max_grad_norm=1.0,
        push_to_hub=False,
        hub_token=None,
        hub_model_id=None,
        logging_dir="logs",
        mixed_precision=None,
        report_to=None,
        local_rank=-1,
        checkpointing_steps=500,
        resume_from_checkpoint=None,
        enable_xformers_memory_efficient_attention=False,
        train_text_encoder=False,
        train_image_encoder=False,
        keep_only_last_checkpoint=False,
        keep_interval=None,
        inference_steps=50,
        guidance_scale=5,
        num_images_per_prompt=1,
        evaluation_batch_size=4,
        finetuned_model_path=None,
        start_idx=0,
        end_idx=50,
        text_prompt_only=False,
        use_multiple_conditioning=False,
        start_merge_step=0,
        image_encoder_type="clip",
        image_encoder_name_or_path="openai/clip-vit-large-patch14",
        num_image_tokens=1,
        max_num_objects=4,
        train_resolution=256,
        object_resolution=256,
        test_resolution=512,
        generate_width=512,
        generate_height=512,
        object_appear_prob=1,
        no_object_augmentation=False,
        image_encoder_trainable_layers=0,
        load_model=None,
        uncondition_prob=0,
        text_only_prob=0,
        text_encoder_use_lora=False,
        lora_text_encoder_r=16,
        lora_text_encoder_alpha=16,
        lora_text_encoder_dropout=0.1,
        lora_text_encoder_bias="none",
        image_encoder_use_lora=False,
        lora_image_encoder_r=16,
        lora_image_encoder_alpha=16,
        lora_image_encoder_dropout=0.1,
        lora_image_encoder_bias="none",
        unet_use_lora=False,
        unet_lora_alpha=1.0,
        num_rows=1,
        test_caption=None,
        test_reference_folder=None,
        load_merged_lora_model=False,
        object_background_processor="random",
        disable_flashattention=False,
        object_types=None,
        object_localization=False,
        localization_layers=5,
        object_localization_weight=0.01,
        object_localization_loss="balanced_l1",
        object_localization_threshold=1.0,
        object_localization_normalize=False,
        unet_lr_scale=1.0,
        val_dataset_name=None,
        mask_loss=False,
        mask_loss_prob=0.5,
        freeze_unet=False,
        use_multiple_datasets=False,
        num_datasets=1,
        min_num_objects=None,
        dataset_type="original",
        retrieval_identity_path=None,
        dataset_name1=None,
        dataset_name2=None,
        dataset_name3=None,
        dataset_type1="original",
        dataset_type2="original",
        dataset_type3="original",
        retrieval_identity_path1=None,
        retrieval_identity_path2=None,
        retrieval_identity_path3=None,
        object_localization_skip_special_tokens=False,
        balance_num_objects=False,
        inference_split="eval",
        num_batches=1,
    )
