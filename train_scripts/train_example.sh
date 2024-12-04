cd /home/ericbanzuzi/DD2412-Final-Project/
export PYTHONPATH=$PWD
echo "!!Training model!!"
python3 src/experiments/00_train_models.py \
    --model WRN \
    --epochs 1 \
    --accelerator gpu \
    --seed 1 \
    --dataset CIFAR10 \
    --model_name WRN_CIFAR10_28_10_FLA \
    --batch_size 256 \
    --loss fla \
    --gammas 5 3 \
    --probs 0.2 1
echo "!!Training done!!"

# CE
# wandb:               epoch 0
# wandb:     train_acc_epoch 0.44715
# wandb:      train_accuracy 0.57812
# wandb:          train_loss 1.12714
# wandb: trainer/global_step 156
# wandb:     valid_acc_epoch 0.20691
# wandb:      valid_accuracy 0.375
# wandb:          valid_loss 2.91303
# wandb:    valid_loss_epoch 3.54736

# FL 0
# wandb: Run summary:
# wandb:               epoch 0
# wandb:     train_acc_epoch 0.44653
# wandb:      train_accuracy 0.57031
# wandb:          train_loss 1.15175
# wandb: trainer/global_step 156
# wandb:     valid_acc_epoch 0.21271
# wandb:      valid_accuracy 0.3125
# wandb:          valid_loss 2.80762
# wandb:    valid_loss_epoch 3.36428

# FL const
# wandb: Run summary:
# wandb:               epoch 0
# wandb:     train_acc_epoch 0.43907
# wandb:      train_accuracy 0.53125
# wandb:          train_loss 0.60789
# wandb: trainer/global_step 156
# wandb:     valid_acc_epoch 0.18056
# wandb:      valid_accuracy 0.1875
# wandb:          valid_loss 2.38837
# wandb:    valid_loss_epoch 3.11902

# FLA
# wandb:               epoch 0
# wandb:     train_acc_epoch 0.43845
# wandb:      train_accuracy 0.53906
# wandb:          train_loss 0.53475
# wandb: trainer/global_step 156
# wandb:     valid_acc_epoch 0.17998
# wandb:      valid_accuracy 0.375
# wandb:          valid_loss 2.20299
# wandb:    valid_loss_epoch 2.89624