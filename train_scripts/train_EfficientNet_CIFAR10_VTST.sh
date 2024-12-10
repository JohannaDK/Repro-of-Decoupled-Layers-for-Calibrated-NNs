#!/bin/bash	

cd /home/johannakhodaverdian/DD2412-Final-Project/
export PYTHONPATH=$PWD
echo "!!Training model!!"
python3 src/experiments/00_train_models.py \
    --freeze_qyx \
    --model_name VTST_CIFAR10_Z128 \
    --model VTST_EfficientNet \
    --epochs 40 \
    --accelerator gpu \
    --latent_dim 128 \
    --seed 0 \
    --pretrained_qyx experiment_results/CIFAR10_EfficientNet/checkpoints/seed=1-epoch=28-valid_loss=0.775-model_name=EfficientNet_CIFAR10_Base_new_seed1.ckpt \
    --dataset CIFAR10 \
    --seeds_per_job 10
echo "!!Training done!!"