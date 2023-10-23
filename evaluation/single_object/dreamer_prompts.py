
def get_stylization_prompt():
    stylization = [
        'a painting <|image|> in the style of Banksy' ,
        'a painting <|image|> in the style of Vincent Van Gogh' ,
        'a colorful graffiti painting <|image|>' ,
        'a watercolor painting <|image|>' ,
        'a Greek marble sculpture <|image|>' ,
        'a Greek bronze sculpture <|image|>' ,
        'a street art mural <|image|>' ,
        'a black and white photograph <|image|>' ,
        'a pointillism painting <|image|>' ,
        'a Japanese woodblock print <|image|>' ,
        'a street art stencil <|image|>' 
    ]
    return stylization

def get_non_style_prompts(class_token, unique_token):
    prompts = [
        'a {0} {1} in an enchanted forest.'.format(class_token, unique_token),
        'a {0} {1} in the desert.'.format(class_token, unique_token),
        'a {0} {1} on a mountaintop.'.format(class_token, unique_token),
        'a {0} {1} in a garden.'.format(class_token, unique_token),
        'a {0} {1} in a park.'.format(class_token, unique_token),
        'a {0} {1} in a meadow.'.format(class_token, unique_token),
        'a {0} {1} on a boat in the ocean.'.format(class_token, unique_token),
        'a {0} {1} in a cable car over the mountains.'.format(class_token, unique_token),
        'a {0} {1} in a city skyline.'.format(class_token, unique_token),
        'a {0} {1} in a historical castle.'.format(class_token, unique_token),
        'a {0} {1} at a bustling train station.'.format(class_token, unique_token),
        'a {0} {1} holding a glass of wine'.format(class_token, unique_token),
        'a {0} {1} holding a slice of pizza'.format(class_token, unique_token),
        'a {0} {1} holding a plate of cake'.format(class_token, unique_token)
    ]
    return prompts 


def get_prompts(unique_token, class_token):
    return get_non_style_prompts(class_token, unique_token) + get_stylization_prompt()

def get_test_subjects():
    face_images = [
        '000000020', 
        '000000257', 
        '000001486', 
        '000002312', 
        '000004319', 
        '000005366', 
        '000006505', 
        '000007619', 
        '000016948',
        '000018561'
    ]

    half_body_images = [
        '000000109', 
        '000000298', 
        '000000575', 
        '000000745', 
        '000002245', 
        '000002345', 
        '000006678', 
        '000006908', 
        '000007459', 
        '000016325'
    ]
    return face_images, half_body_images

IMAGE_TO_GENDER = {
    '000000020': 'woman', 
    '000000257': 'woman', 
    '000001486': 'man', 
    '000002312': 'man', 
    '000004319': 'man', 
    '000005366': 'woman', 
    '000006505': 'boy', 
    '000007619': 'woman', 
    '000016948': 'girl',
    '000018561': 'woman',
    '000000109': 'woman', 
    '000000298': 'woman', 
    '000000575': 'man', 
    '000000745': 'man', 
    '000002245': 'woman', 
    '000002345': 'woman', 
    '000006678': 'woman', 
    '000006908': 'man', 
    '000007459': 'girl', 
    '000016325': 'woman'
}

def get_subject_prompt_combinations(unique_token):
    face_images, half_body_images = get_test_subjects()

    style_pairs  = [] 
    non_style_pairs = [] 

    for face_image in sorted(face_images):
        style_pairs.append((sorted(get_stylization_prompt()), face_image))

    for half_body_image in sorted(half_body_images):
        non_style_pairs.append(
            (
                sorted(get_non_style_prompts(class_token=IMAGE_TO_GENDER[half_body_image], 
                    unique_token=unique_token)
                ), 
                half_body_image
            )
        )
    return style_pairs + non_style_pairs 

if __name__ == '__main__':
    actions = get_prompts('sks', 'man')

    for action in actions:
        print(action)
