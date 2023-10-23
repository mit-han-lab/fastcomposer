IMAGE_TO_GENDER_EVAL = {
    "000001": "man",
    "000181": "man",
    "000268": "man",
    "000612": "woman",
    "000667": "woman",
    "001603": "woman",
    "001854": "man",
    "002088": "woman",
    "002464": "woman",
    "002782": "man",
    "002880": "woman",
    "002929": "woman",
    "002937": "woman",
    "003785": "man",
    "004153": "woman",
}

IMAGE_TO_GENDER_DEMO = {
    "bengio": "man",
    "einstein": "man",
    "feifei": "woman",
    "hinton": "man",
    "lecun": "man",
    "barbara": "woman",
    "newton": "man",
    "johnson": "woman"
}


def get_accessory_prompts(class_token, unique_token):
    prompts = [
        "a {0} {1} wearing a red hat".format(unique_token, class_token),
        "a {0} {1} wearing a santa hat".format(unique_token, class_token),
        "a {0} {1} wearing a rainbow scarf".format(unique_token, class_token),
        "a {0} {1} wearing a black top hat and a monocle".format(
            unique_token, class_token
        ),
        "a {0} {1} in a chef outfit".format(unique_token, class_token),
        "a {0} {1} in a firefighter outfit".format(unique_token, class_token),
        "a {0} {1} in a police outfit".format(unique_token, class_token),
        "a {0} {1} wearing pink glasses".format(unique_token, class_token),
        "a {0} {1} wearing a yellow shirt".format(unique_token, class_token),
        "a {0} {1} in a purple wizard outfit".format(unique_token, class_token),
    ]
    return prompts


def get_style_prompts(class_token, unique_token):
    prompts = [
        f"a painting of a {unique_token} {class_token} in the style of Banksy",
        f"a painting of a {unique_token} {class_token} in the style of Vincent Van Gogh",
        f"a colorful graffiti painting of a {unique_token} {class_token}",
        f"a watercolor painting of a {unique_token} {class_token}",
        f"a Greek marble sculpture of a {unique_token} {class_token}",
        f"a street art mural of a {unique_token} {class_token}",
        f"a black and white photograph of a {unique_token} {class_token}",
        f"a pointillism painting of a {unique_token} {class_token}",
        f"a Japanese woodblock print of a {unique_token} {class_token}",
        f"a street art stencil of a {unique_token} {class_token}",
    ]
    return prompts


def get_context_prompts(class_token, unique_token):
    prompts = [
        "a {0} {1} in the jungle".format(unique_token, class_token),
        "a {0} {1} in the snow".format(unique_token, class_token),
        "a {0} {1} on the beach".format(unique_token, class_token),
        "a {0} {1} on a cobblestone street".format(unique_token, class_token),
        "a {0} {1} on top of pink fabric".format(unique_token, class_token),
        "a {0} {1} on top of a wooden floor".format(unique_token, class_token),
        "a {0} {1} with a city in the background".format(unique_token, class_token),
        "a {0} {1} with a mountain in the background".format(unique_token, class_token),
        "a {0} {1} with a blue house in the background".format(
            unique_token, class_token
        ),
        "a {0} {1} on top of a purple rug in a forest".format(
            unique_token, class_token
        ),
    ]
    return prompts


def get_action_prompts(class_token, unique_token):
    prompts = [
        f"a {unique_token} {class_token} riding a horse",
        f"a {unique_token} {class_token} holding a glass of wine",
        f"a {unique_token} {class_token} holding a piece of cake",
        f"a {unique_token} {class_token} giving a lecture",
        f"a {unique_token} {class_token} reading a book",
        f"a {unique_token} {class_token} gardening in the backyard",
        f"a {unique_token} {class_token} cooking a meal",
        f"a {unique_token} {class_token} working out at the gym",
        f"a {unique_token} {class_token} walking the dog",
        f"a {unique_token} {class_token} baking cookies",
    ]
    return prompts


def get_combinations(unique_token, is_dreamer=False, split="eval"):
    if split == "eval":
        image_to_gender_dict = IMAGE_TO_GENDER_EVAL
    elif split == "demo":
        image_to_gender_dict = IMAGE_TO_GENDER_DEMO
    else:
        raise ValueError(f"split {split} not supported")

    images = list(image_to_gender_dict.keys())

    prompt_pairs = []

    for subject_name in sorted(images):
        class_token = image_to_gender_dict[subject_name]
        if is_dreamer:
            # dreamer swap the order of class_token and unique_token
            all_prompts = (
                get_accessory_prompts(unique_token, class_token)
                + get_style_prompts(unique_token, class_token)
                + get_action_prompts(unique_token, class_token)
                + get_context_prompts(unique_token, class_token)
            )
        else:
            all_prompts = (
                get_accessory_prompts(class_token, unique_token)
                + get_style_prompts(class_token, unique_token)
                + get_action_prompts(class_token, unique_token)
                + get_context_prompts(class_token, unique_token)
            )

        prompt_pairs.append((all_prompts, subject_name))

    return prompt_pairs
