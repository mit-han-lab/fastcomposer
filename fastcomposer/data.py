import os
import torch
from torchvision.io import read_image, ImageReadMode
import glob


def prepare_image_token_idx(image_token_mask, max_num_objects):
    image_token_idx = torch.nonzero(image_token_mask, as_tuple=True)[1]
    image_token_idx_mask = torch.ones_like(image_token_idx, dtype=torch.bool)
    if len(image_token_idx) < max_num_objects:
        image_token_idx = torch.cat(
            [
                image_token_idx,
                torch.zeros(max_num_objects - len(image_token_idx), dtype=torch.long),
            ]
        )
        image_token_idx_mask = torch.cat(
            [
                image_token_idx_mask,
                torch.zeros(
                    max_num_objects - len(image_token_idx_mask),
                    dtype=torch.bool,
                ),
            ]
        )

    image_token_idx = image_token_idx.unsqueeze(0)
    image_token_idx_mask = image_token_idx_mask.unsqueeze(0)
    return image_token_idx, image_token_idx_mask


class DemoDataset(object):
    def __init__(
        self,
        test_caption,
        test_reference_folder,
        tokenizer,
        object_transforms,
        image_token="<|image|>",
        max_num_objects=4,
        device=None,
    ) -> None:
        self.test_caption = test_caption
        self.test_reference_folder = test_reference_folder
        self.tokenizer = tokenizer
        self.image_token = image_token
        self.object_transforms = object_transforms

        tokenizer.add_tokens([image_token], special_tokens=True)
        self.image_token_id = tokenizer.convert_tokens_to_ids(image_token)
        self.max_num_objects = max_num_objects
        self.device = device
        self.image_ids = None

    def set_caption(self, caption):
        self.test_caption = caption

    def set_reference_folder(self, reference_folder):
        self.test_reference_folder = reference_folder

    def set_image_ids(self, image_ids=None):
        self.image_ids = image_ids

    def get_data(self):
        return self.prepare_data()

    def _tokenize_and_mask_noun_phrases_ends(self, caption):
        input_ids = self.tokenizer.encode(caption)

        noun_phrase_end_mask = [False for _ in input_ids]
        clean_input_ids = []
        clean_index = 0

        for i, id in enumerate(input_ids):
            if id == self.image_token_id:
                noun_phrase_end_mask[clean_index - 1] = True
            else:
                clean_input_ids.append(id)
                clean_index += 1

        max_len = self.tokenizer.model_max_length

        if len(clean_input_ids) > max_len:
            clean_input_ids = clean_input_ids[:max_len]
        else:
            clean_input_ids = clean_input_ids + [self.tokenizer.pad_token_id] * (
                max_len - len(clean_input_ids)
            )

        if len(noun_phrase_end_mask) > max_len:
            noun_phrase_end_mask = noun_phrase_end_mask[:max_len]
        else:
            noun_phrase_end_mask = noun_phrase_end_mask + [False] * (
                max_len - len(noun_phrase_end_mask)
            )

        clean_input_ids = torch.tensor(clean_input_ids, dtype=torch.long)
        noun_phrase_end_mask = torch.tensor(noun_phrase_end_mask, dtype=torch.bool)
        return clean_input_ids.unsqueeze(0), noun_phrase_end_mask.unsqueeze(0)

    def prepare_data(self):
        object_pixel_values = []
        image_ids = []

        for image_id in self.image_ids:
            reference_image_path = sorted(
                glob.glob(os.path.join(self.test_reference_folder, image_id, "*.jpg"))
                + glob.glob(os.path.join(self.test_reference_folder, image_id, "*.png"))
                + glob.glob(
                    os.path.join(self.test_reference_folder, image_id, "*.jpeg")
                )
            )[0]

            reference_image = self.object_transforms(
                read_image(reference_image_path, mode=ImageReadMode.RGB)
            ).to(self.device)
            object_pixel_values.append(reference_image)
            image_ids.append(image_id)

        input_ids, image_token_mask = self._tokenize_and_mask_noun_phrases_ends(
            self.test_caption
        )

        image_token_idx, image_token_idx_mask = prepare_image_token_idx(
            image_token_mask, self.max_num_objects
        )

        num_objects = image_token_idx_mask.sum().item()

        object_pixel_values = torch.stack(
            object_pixel_values
        )  # [max_num_objects, 3, 256, 256]
        object_pixel_values = object_pixel_values.to(
            memory_format=torch.contiguous_format
        ).float()

        return {
            "input_ids": input_ids,
            "image_token_mask": image_token_mask,
            "image_token_idx": image_token_idx,
            "image_token_idx_mask": image_token_idx_mask,
            "object_pixel_values": object_pixel_values,
            "num_objects": torch.tensor(num_objects),
            "filenames": image_ids,
        }


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    input_ids = torch.cat([example["input_ids"] for example in examples])
    image_ids = torch.stack([example["image_ids"] for example in examples])

    image_token_mask = torch.cat([example["image_token_mask"] for example in examples])
    image_token_idx = torch.cat([example["image_token_idx"] for example in examples])
    image_token_idx_mask = torch.cat(
        [example["image_token_idx_mask"] for example in examples]
    )

    object_pixel_values = torch.stack(
        [example["object_pixel_values"] for example in examples]
    )
    object_segmaps = torch.stack([example["object_segmaps"] for example in examples])

    num_objects = torch.stack([example["num_objects"] for example in examples])
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "image_token_mask": image_token_mask,
        "image_token_idx": image_token_idx,
        "image_token_idx_mask": image_token_idx_mask,
        "object_pixel_values": object_pixel_values,
        "object_segmaps": object_segmaps,
        "num_objects": num_objects,
        "image_ids": image_ids,
    }


def get_data_loader(dataset, batch_size, shuffle=True):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=0,
    )

    return dataloader
