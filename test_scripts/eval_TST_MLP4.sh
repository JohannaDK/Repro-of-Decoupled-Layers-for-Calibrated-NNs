#!/bin/bash	

cd /home/ericbanzuzi/DD2412-Final-Project/
export PYTHONPATH=$PWD
echo "!!Evaluating model!!"
python3 src/experiments/01_eval_models.py \
    --save_file_name TSTEXP_CIFAR10_MLP4.txt \
    --model_name_file evaluate_cifar10_tst_mlp4.txt
echo "!!Evaluation done!!"