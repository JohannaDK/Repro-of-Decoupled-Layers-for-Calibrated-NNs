#!/bin/bash	

cd /home/ericbanzuzi/DD2412-Final-Project/
export PYTHONPATH=$PWD
echo "!!Training model!!"
python3 src/experiments/00_train_models.py \
    --freeze_qyx \
    --model_name VTST_CIFAR10_Z32_FL \
    --model VTST \
    --epochs 40 \
    --accelerator gpu \
    --latent_dim 32 \
    --seed 0 \
    --pretrained_qyx experiment_results/CIFAR10_WRN/checkpoints/seed=1-epoch=479-valid_loss=0.147-model_name=WRN_CIFAR10_28_10_FL_seed1.ckpt \
    --dataset CIFAR10 \
    --seeds_per_job 10
echo "!!Training done!!"