#!/bin/bash	

cd /home/johannakhodaverdian/DD2412-Final-Project/
export PYTHONPATH=$PWD
echo "!!Training model!!"
python3 src/experiments/00_train_models.py \
    --freeze_qyx \
    --model_name TST_CIFAR10_Z128_FL \
    --model TST \
    --epochs 40 \
    --accelerator gpu \
    --latent_dim 128 \
    --seed 0 \
    --pretrained_qyx experiment_results/CIFAR10_WRN/checkpoints/seed=1-epoch=586-valid_loss=0.450-model_name=WRN_CIFAR10_28_10_Base_seed1.ckpt \
    --dataset CIFAR10 \
    --seeds_per_job 10 \
    --loss fl \
    --gammas 3 \
    --probs 1
echo "!!Training done!!"
