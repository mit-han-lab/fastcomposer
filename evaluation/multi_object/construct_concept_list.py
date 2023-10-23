import itertools
import json 

IMAGE_TO_GENDER = {
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
    "004153": "woman"
}


def get_test_subjects():
    images = [
        '000001', 
        '000181', 
        '000268', 
        '000612', 
        '000667', 
        '001603', 
        '001854', 
        '002088', 
        '002464', 
        '002782', 
        '002880', 
        '002929', 
        '002937', 
        '003785', 
        '004153'
    ]
    return images

images = get_test_subjects()
num_images = len(images)

combinations = list(itertools.combinations(range(num_images), 2))
combinations = sorted(combinations)

for i, (idx1, idx2) in enumerate(combinations):
    idx1 = images[idx1]
    idx2 = images[idx2]
    
    gender1 = IMAGE_TO_GENDER[idx1]
    gender2 = IMAGE_TO_GENDER[idx2]
    
    output = [
        {
            "instance_prompt":     f"photo of a <new1> {gender1}",
            "class_prompt":         gender1,
            "instance_data_dir":    f"./data/celeba_test/{idx1}",
            "class_data_dir":       f"/dataset/dreamer/custom_diffusion/multiple_subject/{idx1}_classes"
        },
        {
            "instance_prompt":     f"photo of a <new2> {gender2}",
            "class_prompt":         gender2,
            "instance_data_dir":    f"./data/celeba_test/{idx2}",
            "class_data_dir":       f"/dataset/dreamer/custom_diffusion/multiple_subject/{idx2}_classes"
        }
    ]

    with open(f"concept_lists/celeba_subject_comb_{i}.json", "w") as f:
        json.dump(output, f, indent=4)
