#!/bin/bash	

cd /home/johannakhodaverdian/DD2412-Final-Project/
export PYTHONPATH=$PWD
echo "!!Training model!!"
python3 src/experiments/00_train_models.py \
    --freeze_qyx \
    --model_name VTST_CIFAR10_FLA_CE \
    --model VTST \
    --epochs 40 \
    --accelerator gpu \
    --latent_dim 32 \
    --seed 7 \
    --pretrained_qyx experiment_results/CIFAR10_WRN/checkpoints/seed=1-epoch=487-valid_loss=0.133-model_name=WRN_CIFAR10_28_10_FLA_seed1.ckpt \
    --dataset CIFAR10 \
    --seeds_per_job 3\
    --loss ce 
echo "!!Training done!!"
