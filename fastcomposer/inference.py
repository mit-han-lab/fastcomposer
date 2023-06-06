from fastcomposer.transforms import get_object_transforms
from fastcomposer.data import DemoDataset
from fastcomposer.model import FastComposerModel
from diffusers import StableDiffusionPipeline
from transformers import CLIPTokenizer
from accelerate.utils import set_seed
from fastcomposer.utils import parse_args
from accelerate import Accelerator
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import os
from tqdm.auto import tqdm
from fastcomposer.pipeline import (
    stable_diffusion_call_with_references_delayed_conditioning,
)
import types
import itertools
import os

@torch.no_grad()
def main():
    args = parse_args()
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path, torch_dtype=weight_dtype
    )

    model = FastComposerModel.from_pretrained(args)

    ckpt_name = "pytorch_model.bin"

    model.load_state_dict(
        torch.load(Path(args.finetuned_model_path) / ckpt_name, map_location="cpu")
    )

    model = model.to(device=accelerator.device, dtype=weight_dtype)

    pipe.unet = model.unet

    if args.enable_xformers_memory_efficient_attention:
        pipe.unet.enable_xformers_memory_efficient_attention()

    pipe.text_encoder = model.text_encoder
    pipe.image_encoder = model.image_encoder

    pipe.postfuse_module = model.postfuse_module

    pipe.inference = types.MethodType(
        stable_diffusion_call_with_references_delayed_conditioning, pipe
    )

    del model

    pipe = pipe.to(accelerator.device)

    # Set up the dataset
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )

    object_transforms = get_object_transforms(args)

    demo_dataset = DemoDataset(
        test_caption=args.test_caption,
        test_reference_folder=args.test_reference_folder,
        tokenizer=tokenizer,
        object_transforms=object_transforms,
        device=accelerator.device,
        max_num_objects=args.max_num_objects,
    )

    image_ids = os.listdir(args.test_reference_folder)
    print(f"Image IDs: {image_ids}")
    demo_dataset.set_image_ids(image_ids)

    unique_token = "<|image|>"

    prompt = args.test_caption
    prompt_text_only = prompt.replace(unique_token, "")

    os.makedirs(args.output_dir, exist_ok=True)

    batch = demo_dataset.get_data()

    input_ids = batch["input_ids"].to(accelerator.device)
    text = tokenizer.batch_decode(input_ids)[0]
    print(prompt)
    # print(input_ids)
    image_token_mask = batch["image_token_mask"].to(accelerator.device)

    # print(image_token_mask)
    all_object_pixel_values = (
        batch["object_pixel_values"].unsqueeze(0).to(accelerator.device)
    )
    num_objects = batch["num_objects"].unsqueeze(0).to(accelerator.device)

    all_object_pixel_values = all_object_pixel_values.to(
        dtype=weight_dtype, device=accelerator.device
    )

    object_pixel_values = all_object_pixel_values  # [:, 0, :, :, :]
    if pipe.image_encoder is not None:
        object_embeds = pipe.image_encoder(object_pixel_values)
    else:
        object_embeds = None

    encoder_hidden_states = pipe.text_encoder(
        input_ids, image_token_mask, object_embeds, num_objects
    )[0]

    encoder_hidden_states_text_only = pipe._encode_prompt(
        prompt_text_only,
        accelerator.device,
        args.num_images_per_prompt,
        do_classifier_free_guidance=False,
    )

    encoder_hidden_states = pipe.postfuse_module(
        encoder_hidden_states,
        object_embeds,
        image_token_mask,
        num_objects,
    )

    cross_attention_kwargs = {}

    images = pipe.inference(
        prompt_embeds=encoder_hidden_states,
        num_inference_steps=args.inference_steps,
        height=args.generate_height,
        width=args.generate_width,
        guidance_scale=args.guidance_scale,
        num_images_per_prompt=args.num_images_per_prompt,
        cross_attention_kwargs=cross_attention_kwargs,
        prompt_embeds_text_only=encoder_hidden_states_text_only,
        start_merge_step=args.start_merge_step,
    ).images

    for instance_id in range(args.num_images_per_prompt):
        images[instance_id].save(
            os.path.join(
                args.output_dir,
                f"output_{instance_id}.png",
            )
        )


if __name__ == "__main__":
    main()
