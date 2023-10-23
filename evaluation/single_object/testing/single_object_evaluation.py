from transformers import AutoImageProcessor, DetaForObjectDetection
from dreamer.evaluation.single_object.testing.data import get_combinations
from dreamer.clip_eval import DINOEvaluator, CLIPEvaluator
from scipy.optimize import linear_sum_assignment
from torchvision.transforms import ToTensor
from accelerate import Accelerator
from typing import List, Tuple
from PIL import Image
from tqdm import tqdm
import numpy as np
import argparse
import torch
import glob
import os


def read_reference_images(folder_path: str) -> List[np.ndarray]:
    images = []
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        image = Image.open(image_path).convert("RGB")
        images.append(image)
    return images


def compute_similarity_matrix(
    evaluator, cropped_images: List[np.ndarray], reference_images: List[np.ndarray]
) -> np.ndarray:
    similarity_matrix = np.zeros((len(cropped_images), len(reference_images)))
    for i, cropped_image in enumerate(cropped_images):
        for j, reference_image in enumerate(reference_images):
            embed1 = evaluator(cropped_image)
            embed2 = evaluator(reference_image)
            similarity_matrix[i, j] = embed1 @ embed2.T

    print(similarity_matrix)
    return similarity_matrix


def greedy_matching(scores):
    n, m = scores.shape
    assert n == m
    res = []
    for _ in range(m):
        pos = np.argmax(scores)
        i, j = pos // m, pos % m

        res.append(scores[i, j])
        scores[i, :] = -1
        scores[:, j] = -1

    return min(res)


def save_image(tensor, path):
    tensor = (tensor[0] * 0.5 + 0.5).clamp(min=0, max=1).permute(1, 2, 0) * 255.0
    tensor = tensor.cpu().numpy().astype(np.uint8)

    Image.fromarray(tensor).save(path)


def compute_average_similarity(
    idx, face_detector, face_similarity, generated_image, reference_image
) -> float:
    generated_face = face_detector(generated_image)

    if generated_face == None:
        return 0.0
    generated_face = generated_face[:1]

    reference_face = face_detector(reference_image)[:1]
    assert len(reference_face) == 1, "no reference face detected in reference image"

    generated_face = generated_face.to(face_detector.device).reshape(1, 3, 160, 160)
    reference_face = reference_face.to(face_detector.device).reshape(1, 3, 160, 160)

    similarity = face_similarity(generated_face) @ face_similarity(reference_face).T
    return max(similarity.item(), 0.0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_images_per_prompt", type=int, default=4)
    parser.add_argument("--prediction_folder", type=str)
    parser.add_argument("--reference_folder", type=str)

    args = parser.parse_args()
    return args


def load_reference_image(reference_folder, image_id):
    path = os.path.join(reference_folder, image_id)
    image_path = sorted(glob.glob(os.path.join(path, "*.jpg")))[0]
    image = Image.open(image_path).convert("RGB")
    return image


@torch.no_grad()
def main():
    args = parse_args()

    accelerator = Accelerator()

    from facenet_pytorch import MTCNN, InceptionResnetV1

    face_detector = MTCNN(
        image_size=160,
        margin=0,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=True,
        device=accelerator.device,
        keep_all=True,
    )
    face_similarity = (
        InceptionResnetV1(pretrained="vggface2").eval().to(accelerator.device)
    )

    text_evaluator = CLIPEvaluator(device=accelerator.device, clip_model="ViT-L/14")

    # get subject
    prompt_subject_pairs = get_combinations("", is_dreamer=True)
    image_alignments, text_alignments = [], []

    for case_id, (prompt_list, subject) in enumerate(tqdm(prompt_subject_pairs)):
        # TODO: Load reference images using image_ids from subjects
        ref_image = load_reference_image(args.reference_folder, subject)

        for prompt_id, prompt in enumerate(prompt_list):
            for instance_id in range(args.num_images_per_prompt):
                generated_image_path = os.path.join(
                    args.prediction_folder,
                    f"subject_{case_id:04d}_prompt_{prompt_id:04d}_instance_{instance_id:04d}.jpg",
                )
                generated_image = Image.open(generated_image_path).convert("RGB")

                identity_similarity = compute_average_similarity(
                    case_id, face_detector, face_similarity, generated_image, ref_image
                )

                generated_image_tensor = (
                    ToTensor()(generated_image).unsqueeze(0) * 2.0 - 1.0
                )
                prompt_similarity = text_evaluator.txt_to_img_similarity(
                    prompt, generated_image_tensor
                )

                image_alignments.append(float(identity_similarity))
                text_alignments.append(float(prompt_similarity))

    image_alignment = sum(image_alignments) / len(image_alignments)
    text_alignment = sum(text_alignments) / len(text_alignments)
    image_std = np.std(image_alignments)
    text_std = np.std(text_alignments)

    print(f"Image Alignment: {image_alignment} +- {image_std}")
    print(f"Text Alignment: {text_alignment} +- {text_std}")
    with open(os.path.join(args.prediction_folder, "score.txt"), "w") as f:
        f.write(f"Image Alignment: {image_alignment} Text Alignment: {text_alignment}")
        f.write(f"Image Alignment Std: {image_std} Text Alignment Std: {text_std}")


if __name__ == "__main__":
    main()
