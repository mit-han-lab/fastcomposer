import os
import torch
from torchvision.io import read_image, ImageReadMode
import glob
import json
import numpy as np
import random
from copy import deepcopy



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


class FastComposerDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        tokenizer,
        train_transforms,
        object_transforms,
        object_processor,
        device=None,
        max_num_objects=4,
        num_image_tokens=1,
        image_token="<|image|>",
        object_appear_prob=1,
        uncondition_prob=0,
        text_only_prob=0,
        object_types=None,
        split="all",
        min_num_objects=None,
        balance_num_objects=False,
    ):
        self.root = root
        self.tokenizer = tokenizer
        self.train_transforms = train_transforms
        self.object_transforms = object_transforms
        self.object_processor = object_processor
        self.max_num_objects = max_num_objects
        self.image_token = image_token
        self.num_image_tokens = num_image_tokens
        self.object_appear_prob = object_appear_prob
        self.device = device
        self.uncondition_prob = uncondition_prob
        self.text_only_prob = text_only_prob
        self.object_types = object_types

        if split == "all":
            image_ids_path = os.path.join(root, "image_ids.txt")
        elif split == "train":
            image_ids_path = os.path.join(root, "image_ids_train.txt")
        elif split == "test":
            image_ids_path = os.path.join(root, "image_ids_test.txt")
        else:
            raise ValueError(f"Unknown split {split}")

        with open(image_ids_path, "r") as f:
            self.image_ids = f.read().splitlines()

        tokenizer.add_tokens([image_token], special_tokens=True)
        self.image_token_id = tokenizer.convert_tokens_to_ids(image_token)

        if min_num_objects is not None:
            print(f"Filtering images with less than {min_num_objects} objects")
            filtered_image_ids = []
            for image_id in tqdm(self.image_ids):
                chunk = image_id[:5]
                info_path = os.path.join(self.root, chunk, image_id + ".json")
                with open(info_path, "r") as f:
                    info_dict = json.load(f)
                segments = info_dict["segments"]

                if self.object_types is not None:
                    segments = [
                        segment
                        for segment in segments
                        if segment["coco_label"] in self.object_types
                    ]

                if len(segments) >= min_num_objects:
                    filtered_image_ids.append(image_id)
            self.image_ids = filtered_image_ids

        if balance_num_objects:
            _balance_num_objects(self)

    def __len__(self):
        return len(self.image_ids)

    def _tokenize_and_mask_noun_phrases_ends(self, caption, segments):
        for segment in reversed(segments):
            end = segment["end"]
            caption = caption[:end] + self.image_token + caption[end:]

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

    @torch.no_grad()
    def preprocess(self, image, info_dict, segmap, image_id):
        caption = info_dict["caption"]
        segments = info_dict["segments"]

        if self.object_types is not None:
            segments = [
                segment
                for segment in segments
                if segment["coco_label"] in self.object_types
            ]

        pixel_values, transformed_segmap = self.train_transforms(image, segmap)

        object_pixel_values = []
        object_segmaps = []

        prob = random.random()
        if prob < self.uncondition_prob:
            caption = ""
            segments = []
        elif prob < self.uncondition_prob + self.text_only_prob:
            segments = []
        else:
            segments = [
                segment
                for segment in segments
                if random.random() < self.object_appear_prob
            ]

        if len(segments) > self.max_num_objects:
            # random sample objects
            segments = random.sample(segments, self.max_num_objects)

        segments = sorted(segments, key=lambda x: x["end"])

        background = self.object_processor.get_background(image)

        for segment in segments:
            id = segment["id"]
            bbox = segment["bbox"]  # [h1, w1, h2, w2]
            object_image = self.object_processor(
                deepcopy(image), background, segmap, id, bbox
            )
            object_pixel_values.append(self.object_transforms(object_image))
            object_segmaps.append(transformed_segmap == id)

        input_ids, image_token_mask = self._tokenize_and_mask_noun_phrases_ends(
            caption, segments
        )

        image_token_idx, image_token_idx_mask = prepare_image_token_idx(
            image_token_mask, self.max_num_objects
        )

        num_objects = image_token_idx_mask.sum().item()
        object_pixel_values = object_pixel_values[:num_objects]
        object_segmaps = object_segmaps[:num_objects]

        if num_objects > 0:
            padding_object_pixel_values = torch.zeros_like(object_pixel_values[0])
        else:
            padding_object_pixel_values = self.object_transforms(background)
            padding_object_pixel_values[:] = 0

        if num_objects < self.max_num_objects:
            object_pixel_values += [
                torch.zeros_like(padding_object_pixel_values)
                for _ in range(self.max_num_objects - num_objects)
            ]
            object_segmaps += [
                torch.zeros_like(transformed_segmap)
                for _ in range(self.max_num_objects - num_objects)
            ]

        object_pixel_values = torch.stack(
            object_pixel_values
        )  # [max_num_objects, 3, 256, 256]
        object_pixel_values = object_pixel_values.to(
            memory_format=torch.contiguous_format
        ).float()

        object_segmaps = torch.stack(
            object_segmaps
        ).float()  # [max_num_objects, 256, 256]

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "image_token_mask": image_token_mask,
            "image_token_idx": image_token_idx,
            "image_token_idx_mask": image_token_idx_mask,
            "object_pixel_values": object_pixel_values,
            "object_segmaps": object_segmaps,
            "num_objects": torch.tensor(num_objects),
            "image_ids": torch.tensor(image_id),
        }

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        chunk = image_id[:5]
        image_path = os.path.join(self.root, chunk, image_id + ".jpg")
        info_path = os.path.join(self.root, chunk, image_id + ".json")
        segmap_path = os.path.join(self.root, chunk, image_id + ".npy")

        image = read_image(image_path, mode=ImageReadMode.RGB)

        with open(info_path, "r") as f:
            info_dict = json.load(f)
        segmap = torch.from_numpy(np.load(segmap_path))

        if self.device is not None:
            image = image.to(self.device)
            segmap = segmap.to(self.device)

        return self.preprocess(image, info_dict, segmap, int(image_id))


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
