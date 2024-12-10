#!/bin/bash	

cd /home/johannakhodaverdian/DD2412-Final-Project/
export PYTHONPATH=$PWD
echo "!!Training model!!"
python3 src/experiments/00_train_models.py \
    --model EfficientNet \
    --epochs 600 \
    --accelerator gpu \
    --seed 1 \
    --dataset CIFAR10 \
    --model_name EfficientNet_CIFAR10_Base_new \
    --batch_size 256
echo "!!Training done!!"