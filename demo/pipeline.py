from fastcomposer.pipeline import StableDiffusionFastCompposerPipeline
from fastcomposer.model import FastComposerModel
from diffusers import StableDiffusionPipeline
from fastcomposer.transforms import PadToSquare
from torchvision import transforms as T
from transformers import CLIPTokenizer
from collections import OrderedDict
import torch


def convert_model_to_pipeline(args, device):
    model = FastComposerModel.from_pretrained(args)

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer.add_tokens(["img"], special_tokens=True)
    image_token_id = tokenizer.convert_tokens_to_ids("img")

    pipe = StableDiffusionFastCompposerPipeline.from_pretrained(
        args.pretrained_model_name_or_path, torch_dtype=weight_dtype
    ).to(device)

    model.load_state_dict(torch.load(args.finetuned_model_path, map_location="cpu"))
    model = model.to(dtype=weight_dtype, device=device)

    pipe.object_transforms = torch.nn.Sequential(
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
    pipe.unet = model.unet
    pipe.text_encoder = model.text_encoder
    pipe.postfuse_module = model.postfuse_module
    pipe.image_encoder = model.image_encoder
    pipe.image_token_id = image_token_id
    pipe.special_tokenizer = tokenizer

    del model
    return pipe


if __name__ == "__main__":
    convert_model_to_pipeline()
