# FastComposer: Tuning-Free Multi-Subject Image Generation with Localized Attention [[website](https://fastcomposer.mit.edu/)] [[demo](https://2acfe10ec96df6f2b0.gradio.live)]

![multi-subject](figures/multi-subject.png)

## Abstract

Diffusion models excel at text-to-image generation, especially in subject-driven generation for personalized images. However, existing methods are inefficient due to the subject-specific fine-tuning, which is computationally intensive and hampers efficient deployment. Moreover, existing methods struggle with multi-subject generation as they often blend features among subjects. We present FastComposer which enables efficient, personalized, multi-subject text-to-image generation without fine-tuning. FastComposer uses subject embeddings extracted by an image encoder to augment the generic text conditioning in diffusion models, enabling personalized image generation based on subject images and textual instructions with only forward passes. To address the identity blending problem in the multi-subject generation, FastComposer proposes cross-attention localization supervision during training, enforcing the attention of reference subjects localized to the correct regions in the target images. Naively conditioning on subject embeddings results in subject overfitting. FastComposer proposes delayed subject conditioning in the denoising step to maintain both identity and editability in subject-driven image generation. FastComposer generates images of multiple unseen individuals with different styles, actions, and contexts. It achieves 300x-2500x speedup compared to fine-tuning-based methods and requires zero extra storage for new subjects. FastComposer paves the way for efficient, personalized, and high-quality multi-subject image creation.


## Usage

### Environment Setup

```bash
conda create -n fastcomposer python
conda activate fastcomposer
pip install torch torchvision torchaudio
pip install transformers accelerate datasets evaluate diffusers==0.16.1 xformers triton scipy clip gradio

python setup.py install
```

### Download the Pre-trained Models

```bash
mkdir -p model/fastcomposer ; cd model/fastcomposer
wget https://huggingface.co/mit-han-lab/fastcomposer/resolve/main/pytorch_model.bin
```

### Gradio Demo

We host a demo [here](https://17283ded5673112d93.gradio.live). You can also run the demo locally by 

```bash   
python demo/run_gradio.py --finetuned_model_path model/fastcomposer/pytorch_model.bin  --mixed_precision "fp16"
```

### Inference

```bash
bash scripts/run_inference.sh
```
## TODOs

- [x] Release inference code
- [x] Release pre-trained models
- [x] Release demo
- [ ] Release training code and data
- [ ] Release evaluation code and data

## Citation

If you find FastComposer useful or relevant to your research, please kindly cite our paper:

```bibtex
@article{xiao2023fastcomposer,
            title={FastComposer: Tuning-Free Multi-Subject Image Generation with Localized Attention},
            author={Xiao, Guangxuan and Yin, Tianwei and Freeman, William T. and Durand, Frédo and Han, Song},
            journal={arXiv},
            year={2023}
          }
```
