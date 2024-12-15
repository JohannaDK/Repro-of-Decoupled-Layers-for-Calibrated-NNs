#!/bin/bash	

cd /home/johannakhodaverdian/DD2412-Final-Project/
export PYTHONPATH=$PWD
echo "!!Training model!!"
python3 src/experiments/00_train_models.py \
    --freeze_qyx \
    --model_name VTST_CIFAR10_Z32 \
    --model VTST_ResNet50 \
    --epochs 40 \
    --accelerator gpu \
    --latent_dim 32 \
    --seed 0 \
    --pretrained_qyx experiment_results/CIFAR10_ResNet50/checkpoints/seed=1-epoch=194-valid_loss=0.566-model_name=ResNet50_CIFAR10_Base_seed1.ckpt \
    --dataset CIFAR10 \
    --seeds_per_job 10
echo "!!Training done!!"