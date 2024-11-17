#!/bin/bash	

cd /home/johannakhodaverdian/DD2412-Final-Project/
export PYTHONPATH=$PWD
echo "!!Evaluating model!!"
python3 src/experiments/01_eval_models.py \
    --save_file_name WRN_CIFAR100_TST.txt \
    --model_name_file evaluate_wrn_cifar100_tst.txt
echo "!!Evaluation done!!"