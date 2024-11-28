#!/bin/bash	

cd /home/ericbanzuzi/DD2412-Final-Project/
export PYTHONPATH=$PWD
echo "!!Evaluating model!!"
python3 src/experiments/01_eval_models.py \
    --save_file_name VTSTEXP_CIFAR10_N5_Z32_M=1.txt \
    --model_name_file evaluate_cifar10_vtst_train_n=5_test_m=1.txt \
    --num_samples 1
echo "!!Evaluation done!!"