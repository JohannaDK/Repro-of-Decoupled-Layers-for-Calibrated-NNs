cd /home/johannakhodaverdian/DD2412-Final-Project/
export PYTHONPATH=$PWD
echo "!!Training model!!"
python3 src/experiments/00_train_models.py \
    --model WRN \
    --epochs 600 \
    --accelerator gpu \
    --seed 1 \
    --dataset CIFAR100 \
    --model_name WRN_CIFAR100_28_10_Base \
    --batch_size 256
echo "!!Training done!!"