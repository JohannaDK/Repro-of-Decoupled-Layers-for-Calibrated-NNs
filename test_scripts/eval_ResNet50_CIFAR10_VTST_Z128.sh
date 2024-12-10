#!/bin/bash	

cd /home/johannakhodaverdian/DD2412-Final-Project/
export PYTHONPATH=$PWD
echo "!!Evaluating model!!"
python3 src/experiments/01_eval_models.py \
    --save_file_name ResNet50_CIFAR10_VTST_M=10.txt \
    --model_name_file evaluate_ResNet50_cifar10_vtst_m=10.txt \
    --num_samples 10
echo "!!Evaluation done!!"