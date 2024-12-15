#!/bin/bash	

cd /home/johannakhodaverdian/DD2412-Final-Project/
export PYTHONPATH=$PWD
echo "!!Evaluating model!!"
python3 src/experiments/01_eval_models.py \
    --save_file_name EfficientNet_CIFAR10_new.txt \
    --model_name_file evaluate_EfficientNet_cifar10.txt
echo "!!Evaluation done!!"
