#!/bin/bash	

cd /home/ericbanzuzi/DD2412-Final-Project/
export PYTHONPATH=$PWD
echo "!!Evaluating model!!"
python3 src/experiments/01_eval_models.py \
    --save_file_name VTSTEXP_CIFAR10_MLP4_Z128.txt \
    --model_name_file evaluate_cifar10_vtst_m=10_mlp4.txt \
    --num_samples 10
echo "!!Evaluation done!!"