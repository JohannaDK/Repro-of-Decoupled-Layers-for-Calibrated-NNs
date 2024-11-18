cd /home/kathideckenbach/DD2412-Final-Project/
export PYTHONPATH=$PWD
echo "!!Evaluating model!!"
python3 src/experiments/01_eval_models.py \
    --save_file_name simple_CNN_VTST_M10.txt \
    --model_name_file evaluate_cnn_vtst.txt \
    --num_samples 10
echo "!!Evaluation done!!"