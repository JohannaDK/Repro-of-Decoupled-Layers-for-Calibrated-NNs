# A Reproducibility Study of Decoupling Feature Extraction and Classification Layers for Calibrated Neural Networks

Link to our paper: [https://openreview.net/forum?id=5Hwzd48ILf](https://openreview.net/forum?id=5Hwzd48ILf)

## Overview
This project replicates and extends the paper [Decoupling Feature Extraction and Classification Layers for Calibrated Neural Networks](https://github.com/MJordahn/Decoupled-Layers-for-Calibrated-NNs). We to reproduced the main experiments of the paper "Decoupling Feature Extraction and Classification Layers for Calibrated Neural Networks" by Jordahn and Olmos, and extended the code base in the following way:

- Added a simple CNN architecture
- Added a ViT architecture and integrated Tiny ImageNet dataset
- Added different TST and V-TST MLP heads
- Added training with multiple samples
- Added a ResNet-50 and EfficientNet-B5 architectures
- Added [Focal Loss](https://arxiv.org/abs/1708.02002) and [Adaptive Focal Loss](https://github.com/torrvision/focal_calibration/tree/main)
- Added L2 regularization and Label Smoothing

## Datasets
We used CIFAR-10 and CIFAR-100 as well as CIFAR-10-C and CIFAR-100-C to reproduce the main results. For finetuning the ViT, we used Tiny ImageNet. For *out of distribution detection*, we included SVHN for evaluation.

## Experiments
We did the following experiments:

- Reproduce first two rows of Table 1 and 2 from the original paper.
- Reproduce Table 3 and 6 from the original paper.
- Six additional ablation studies with CIFAR10:    
    - **Dependence on Network Architecture:** Run the experiments of Table 1 and 2 with ResNet-50 to see if they extend to a new architecture.
    - **Second Stage Network Architecture:** Run the experiments of Table 1 and 2 with different sized MLP for the second stage to see their effect.
    - **VTST Dependence on Number of Training Samples:** Run the experiments of Table 1 and 2 for V-TST with different number of samples used for training to see their effect.
    - **Focal Loss:**  Run the experiments of Table 1 and 2 with Focal Loss and Adaptive Focal Loss to see how the TST and V-TST techniques perform in combination with another implicit regularization method for calibration.
    - **L2-regularization:**  Run the experiments of Table 1 and 2 with L2-regularization to see how the TST and V-TST techniques perform in combination with another explicit regularization method for calibration.
    - **Label Smoothing:**  Run the experiments of Table 1 and 2 with Label Smoothing to see how the TST and V-TST techniques perform in combination with another similar implicit regularization method for calibration.

## Conclusions
We showed that the methods suggested by Jordahn and Olmos improves calibration independently of the architecture of the model. However, we were unable to replicate the results fully for WRN trained on CIFAR100. The choice of latent variable for the MLP head in the second stage can affect the results greatly though and should be considered carefully along with the number of layers used. However, increasing the number of MC samples during training was not found to have a positive impact on calibration. Combining their method with focal loss did not improve calibration in all cases, however, calibration can be improved further by using focal loss to train the base model which is used in combination with a CE loss in the second stage of training. 

Overall, we found that TST and V-TST can further improve calibration when combined with another regularization method. However, both the choice of regularization method and the way it is combined with TST or V-TST significantly influence the method’s effectiveness on calibration.

## Getting started
To get started follow the instructions of [the original authors](https://github.com/MJordahn/Decoupled-Layers-for-Calibrated-NNs), and if you want to run our additional experiments use the following arguments as shown below.

For the model you can now use the following arguments: `WRN`, `VIT`, `CNN` and `ResNet50`. The `VIT` is supposed to be used in combination with `--dataset TINYIMAGENET`, and through `--vitbase` you can specify which weights you want to load for finetuning. 

If you want to experiment with TST or V-TST MLP sizes or number of training samples, you can use the argument `--model TSTEXP` or `--model VTSTEXP` together with arguments `--mlp_size` and `--train_samples`. 

If you want to use different loss functions, we provide three examples of how to specify it into your `experiment.sh` bash script.

To firstly train a base WRN 28-10 for a dataset, run the following command but replacing \<DATASET\> with CIFAR10, SVHN or CIFAR100:

```
python3 src/experiments/00_train_models.py \
    --model <MODEL> \
    --epochs 600 \
    --accelerator gpu \
    --seed <SEED> \
    --dataset <DATASET> \
    --model_name <MODEL>_<DATASET>_Base \
    --batch_size 256 \
    --loss <loss>
```
where `loss` is one of the following: `ce`, `fl` or `fla`. If the loss is `fl` or `fla`, the arguments `--gammas` and `--probs` have to be included. The default loss is `ce`.


The trained model found with best validation should be saved in `./experiment_results/\<DATASET\>_<MODEL>/checkpoints`. Now to run TST using Adaptive Focal Loss (`fla`), run the following command-line command:

```
python3 src/experiments/00_train_models.py \
    --freeze_qyx \
    --model_name TST_<DATASET>_Z<Z> \
    --model TST \
    --epochs 40 \
    --accelerator gpu \
    --latent_dim <Z> \
    --seed <SEED> \
    --pretrained_qyx <PATH_TO_TRAINED_MODEL> \
    --dataset <DATASET> \
    --loss fla \
    --gammas 5 3 \
    --probs 0.2 1
```

To similarly train V-TST using Focal Loss (`fl`) with constant gamma, run:

```
python3 src/experiments/00_train_models.py \
    --freeze_qyx \
    --model_name VTST_<DATASET>_Z<Z> \
    --model VTST \
    --epochs 40 \
    --accelerator gpu \
    --latent_dim <Z> \
    --seed <SEED> \
    --pretrained_qyx <PATH_TO_TRAINED_MODEL> \
    --dataset <DATASET> \
    --loss fl \
    --gammas 3 \
    --probs 1
```

> [!TIP]
> If you need help setting up a deep learning VM instance on Google Cloud Platform and connecting it to your local VS Code, check out [this repository](https://github.com/rosameliacarioni/finetune-llm) for detailed guidance.
> 
> To run experiments in a background process use `nohup`, a small example is provided in `train_scripts/example_train_run.ipynb`.
