#!/bin/bash	

cd /home/johannakhodaverdian/DD2412-Final-Project/
export PYTHONPATH=$PWD
echo "!!Evaluating model!!"
python3 src/experiments/01_eval_models.py \
    --save_file_name ResNet50_CIAFR10_TS_final.txt \
    --model_name_file evaluate_ResNet50_cifar10.txt \
    --temperature_scale
echo "!!Evaluation done!!"
