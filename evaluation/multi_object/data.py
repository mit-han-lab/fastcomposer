import itertools

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
    "hopper": "woman",
    "einstein": "man",
    "feifei": "woman",
    "hinton": "man",
    "johnson": "woman",
    "lecun": "man",
    "newton": "man",
}

IMAGE_TO_GENDER_DEMO2 = {
    "bengio": "man",
    "einstein": "man",
    "feifei": "woman",
    "hinton": "man",
    "lecun": "man",
    "barbara": "woman",
    "newton": "man",
    "johnson": "woman",
}


def get_dreamer_demo_prompt_subject_pairs(unique_token="<|image|>"):
    if unique_token != "":
        unique_token = f" {unique_token}"
    subject_pairs = [
        (
            [f"a man{unique_token} and a man{unique_token} sitting in a park together"],
            ["einstein", "newton"],
        ),
        (
            [f"a man{unique_token} and a man{unique_token} on the beach"],
            ["hinton", "bengio"],
        ),
        (
            [
                f"a Japanese Woodblock Prints of a woman{unique_token} and a man{unique_token}"
            ],
            ["barbara", "lecun"],
        ),
        (
            [f"a woman{unique_token} and a woman{unique_token} cooking together"],
            ["johnson", "feifei"],
        ),
        (
            [f"a man{unique_token} holding a glass of wine"],
            ["einstein"],
        ),
        (
            [f"a woman{unique_token} riding a horse"],
            ["feifei"],
        ),
        (
            [f"a man{unique_token} in the snow"],
            ["hinton"],
        ),
        (
            [f"a man{unique_token} on the beach"],
            ["lecun"],
        ),
        (
            [f"a painting of a man{unique_token} in the style of Van Gogh"],
            ["bengio"],
        ),
        (
            [f"a Pointillism painting of a woman{unique_token}"],
            ["barbara"],
        ),
        (
            [f"a woman{unique_token} wearing a Santa hat"],
            ["johnson"],
        ),
        (
            [f"a man{unique_token} in a purple wizard outfit"],
            ["newton"],
        ),
    ]

    return subject_pairs


def get_dreamer_pets_prompt_subject_pairs(unique_token="<|image|>"):
    if unique_token != "":
        unique_token = f" {unique_token}"
    subject_pairs = [
        (
            [f"a dog{unique_token} and a cat{unique_token} sitting together"],
            ["corgie", "blue"],
        ),
        (
            [f"a cat{unique_token} wearing sunglasses"],
            ["orange"],
        ),
    ]

    return subject_pairs


def get_subject(class_tokens, unique_tokens):
    num_subjects = len(class_tokens)
    assert num_subjects == len(unique_tokens)
    subject = "a {0} {1}".format(unique_tokens[0], class_tokens[0])
    for i in range(1, num_subjects):
        subject += " and a {0} {1}".format(unique_tokens[i], class_tokens[i])
    return subject


def get_style_prompts(class_tokens, unique_tokens, split="eval"):
    subject = get_subject(class_tokens, unique_tokens)
    if split == "demo2":
        prompts = [
            f"a Japanese woodblock print of {subject}",
        ]
    else:
        prompts = [
            f"a painting of {subject} together in the style of Banksy",
            f"a painting of {subject} together in the style of Vincent Van Gogh",
            f"a watercolor painting of {subject} together",
            f"a street art mural of {subject} together",
            f"a black and white photograph of {subject} together",
            f"a pointillism painting of {subject} together",
            f"a Japanese woodblock print of {subject} together",
        ]
    return prompts


def get_context_prompts(class_tokens, unique_tokens, split="eval"):
    subject = get_subject(class_tokens, unique_tokens)
    if split == "demo2":
        prompts = [
            f"a photo of {subject} together on the beach",
        ]
    else:
        prompts = [
            f"a photo of {subject} together in the jungle",
            f"a photo of {subject} together in the snow",
            f"a photo of {subject} together on the beach",
            f"a photo of {subject} together with a city in the background",
            f"a photo of {subject} together with a mountain in the background",
            f"a photo of {subject} together with a blue house in the background",
        ]

    return prompts


def get_action_prompts(class_tokens, unique_tokens, split="eval"):
    subject = get_subject(class_tokens, unique_tokens)
    if split == "demo2":
        prompts = [
            f"a photo of {subject} cooking together",
            f"a photo of {subject} sitting in a park together",
        ]
    else:
        prompts = [
            f"a photo of {subject} gardening in the backyard together",
            f"a photo of {subject} cooking a meal together",
            f"a photo of {subject} sitting in a park together",
            f"a photo of {subject} working out at the gym together",
            f"a photo of {subject} baking cookies together",
            f"a photo of {subject} posing for a selfie together",
            f"a photo of {subject} making funny faces for a photo booth together",
            f"a photo of {subject} playing a musical duet together",
        ]
    return prompts


def get_combinations(unique_token, num_subjects, is_dreamer=False, split="eval"):
    if split == "dreamer_demo":
        return get_dreamer_demo_prompt_subject_pairs(unique_token)
    if split == "dreamer_pets":
        return get_dreamer_pets_prompt_subject_pairs(unique_token)
    if split == "eval":
        image_to_gender = IMAGE_TO_GENDER_EVAL
    elif split == "demo":
        image_to_gender = IMAGE_TO_GENDER_DEMO
    elif split == "demo2":
        image_to_gender = IMAGE_TO_GENDER_DEMO2
    else:
        raise ValueError("split must be either eval or demo")

    images = list(image_to_gender.keys())
    num_images = len(images)
    prompt_pairs = []

    combinations = list(itertools.combinations(range(num_images), num_subjects))
    combinations = sorted(combinations)

    if isinstance(unique_token, list):
        assert len(unique_token) == num_subjects
        unique_tokens = unique_token
    else:
        assert isinstance(unique_token, str)
        unique_tokens = [unique_token] * num_subjects

    for comb in sorted(combinations):
        image_ids = [images[i] for i in comb]
        class_tokens = [image_to_gender[i] for i in image_ids]
        if is_dreamer:
            # dreamer swap the order of class_token and unique_token.
            all_prompts = (
                get_style_prompts(unique_tokens, class_tokens, split)
                + get_action_prompts(unique_tokens, class_tokens, split)
                + get_context_prompts(unique_tokens, class_tokens, split)
            )
        else:
            all_prompts = (
                get_style_prompts(class_tokens, unique_tokens, split)
                + get_action_prompts(class_tokens, unique_tokens, split)
                + get_context_prompts(class_tokens, unique_tokens, split)
            )

        prompt_pairs.append((all_prompts, image_ids))

    return prompt_pairs
