cd /home/kathideckenbach/DD2412-Final-Project/
export PYTHONPATH=$PWD
echo "!!Training model!!"
python3 src/experiments/00_train_models.py \
    --model VTST_VIT \
    --epochs 5 \
    --accelerator gpu \
    --seed 0 \
    --dataset TINYIMAGENET \
    --model_name VTST_TINYIMAGENET_VIT_Base \
    --batch_size 64 \
    --pretrained_qyx ./experiment_results/TINYIMAGENET_VIT/checkpoints/seed=1-epoch=01-valid_loss=0.703-model_name=VIT_TINYIMAGENET_Base_seed1.ckpt \
    --seeds_per_job 10 \
    --latent_dim 128 \
    --freeze_qyx
echo "!!Training done!!"