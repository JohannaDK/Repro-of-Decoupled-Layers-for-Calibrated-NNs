#!/bin/bash	

cd /home/ericbanzuzi/DD2412-Final-Project/
export PYTHONPATH=$PWD
echo "!!Evaluating model!!"
python3 src/experiments/01_eval_models.py \
    --save_file_name REINIT_CIFAR10_REINIT.txt \
    --model_name_file evaluate_cifar10_reinit.txt
echo "!!Evaluation done!!"