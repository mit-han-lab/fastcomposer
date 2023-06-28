export WANDB_NAME=postfuse-localize-ffhq-1_5-1e-5
export WANDB_DISABLE_SERVICE=true
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

DATASET_PATH="data/ffhq_wild_files"

DATASET_NAME="ffhq"
FAMILY=runwayml
MODEL=stable-diffusion-v1-5
IMAGE_ENCODER=openai/clip-vit-large-patch14

accelerate launch \
    --mixed_precision=bf16 \
    --machine_rank 0 \
    --num_machines 1 \
    --main_process_port 11135 \
    --num_processes 8 \
    --multi_gpu \
    fastcomposer/train.py \
    --pretrained_model_name_or_path ${FAMILY}/${MODEL} \
    --dataset_name ${DATASET_PATH} \
    --logging_dir logs/${MODEL}/${DATASET_NAME}/${WANDB_NAME} \
    --output_dir models/${MODEL}/${DATASET_NAME}/${WANDB_NAME} \
    --max_train_steps 150000 \
    --num_train_epochs 150000 \
    --train_batch_size 16 \
    --learning_rate 1e-5 \
    --unet_lr_scale 1.0 \
    --checkpointing_steps 200 \
    --mixed_precision bf16 \
    --allow_tf32 \
    --keep_only_last_checkpoint \
    --keep_interval 10000 \
    --seed 42 \
    --image_encoder_type clip \
    --image_encoder_name_or_path ${IMAGE_ENCODER} \
    --num_image_tokens 1 \
    --max_num_objects 4 \
    --train_resolution 512 \
    --object_resolution 224 \
    --text_image_linking postfuse \
    --object_appear_prob 0.9 \
    --uncondition_prob 0.1 \
    --object_background_processor random \
    --disable_flashattention \
    --train_image_encoder \
    --image_encoder_trainable_layers 2 \
    --object_types person \
    --mask_loss \
    --mask_loss_prob 0.5 \
    --object_localization \
    --object_localization_weight 1e-3 \
    --object_localization_loss balanced_l1 \
    --resume_from_checkpoint latest \
    --report_to wandb
