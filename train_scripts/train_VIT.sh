cd /home/kathideckenbach/DD2412-Final-Project/
export PYTHONPATH=$PWD
echo "!!Training model!!"
python3 src/experiments/00_train_models.py \
    --model VIT \
    --epochs 2 \
    --accelerator gpu \
    --seed 1 \
    --dataset TINYIMAGENET \
    --model_name TINYIMAGENET_VIT_Base \
    --batch_size 256 
echo "!!Training done!!"