cd /home/kathideckenbach/DD2412-Final-Project/
export PYTHONPATH=$PWD
echo "!!Evaluating model!!"
python3 src/experiments/01_eval_models.py \
    --save_file_name VIT_VTST_M1.txt \
    --model_name_file evaluate_vit_vtst.txt \
    --num_samples 1
echo "!!Evaluation done!!"