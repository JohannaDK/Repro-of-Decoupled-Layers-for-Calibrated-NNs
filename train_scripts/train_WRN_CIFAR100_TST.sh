#!/bin/bash	

cd /home/johannakhodaverdian/DD2412-Final-Project/
export PYTHONPATH=$PWD
echo "!!Training model!!"
python3 src/experiments/00_train_models.py \
    --freeze_qyx \
    --model_name TST_CIFAR100_Z512 \
    --model TST \
    --epochs 40 \
    --accelerator gpu \
    --latent_dim 512 \
    --seed 0 \
    --pretrained_qyx experiment_results/table_metrics/CIFAR100_WRN/checkpoints/seed=1-epoch=18-valid_loss=1.939-model_name=WRN_CIFAR100_28_10_Base_seed1.ckpt \
    --dataset CIFAR100 \
    --seeds_per_job 10
echo "!!Training done!!"
