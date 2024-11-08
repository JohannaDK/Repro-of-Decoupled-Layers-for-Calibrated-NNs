cd /home/kathideckenbach/DD2412-Final-Project/
export PYTHONPATH=$PWD
echo "!!Training model!!"
python3 src/experiments/00_train_models.py \
    --model CNN \
    --epochs 600 \
    --accelerator gpu \
    --seed 1 \
    --dataset CIFAR10 \
    --model_name CIFAR10_CNN_Base \
    --batch_size 256
echo "!!Training done!!"