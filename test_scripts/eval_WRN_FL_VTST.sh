#!/bin/bash	

cd /home/ericbanzuzi/DD2412-Final-Project/
export PYTHONPATH=$PWD
echo "!!Evaluating model!!"
python3 src/experiments/01_eval_models.py \
    --save_file_name WRN_FL_VTST_M=10.txt \
    --model_name_file evaluate_wrn_fl_vtst_z128.txt \
    --num_samples 10
echo "!!Evaluation done!!"
