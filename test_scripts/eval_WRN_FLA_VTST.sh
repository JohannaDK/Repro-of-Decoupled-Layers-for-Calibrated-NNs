#!/bin/bash	

cd /home/ericbanzuzi/DD2412-Final-Project/
export PYTHONPATH=$PWD
echo "!!Evaluating model!!"
python3 src/experiments/01_eval_models.py \
    --save_file_name WRN_FLA_VTST_M=10_FLA_CE.txt \
    --model_name_file evaluate_wrn_fla_vtst_z128.txt \
    --num_samples 10
echo "!!Evaluation done!!"
