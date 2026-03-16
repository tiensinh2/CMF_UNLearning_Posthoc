#!/usr/bin/env bash
set -euo pipefail

#########################
# Basic configuration
#########################

# Data root directory (default ~/data, or pass as first argument at runtime)
DATA_ROOT=${1:-$HOME/data}

# Available GPU list
GPUS=(0 1 2 3 4 5)

#  dataset;arch
EXPS=(
  cifar10;vit_s_16
  cifar100;vit_s_16
  tinyimagenet;vit_s_16
)

# ViT  CIFAR TinyImageNet 
declare -A LR
LR[cifar10]=3e-4
LR[cifar100]=3e-4
LR[tinyimagenet]=1e-4

#########################
# GPU  GPU
#########################

FIFO=/tmp/$$.fifo
mkfifo $FIFO
exec 3<>$FIFO
rm $FIFO

# Initialize token pool
for gpu in ${GPUS[@]}; do
  echo $gpu >&3
done

#########################
#  fine-tune 
#########################

for cfg in ${EXPS[@]}; do
  IFS=; read -r dataset arch <<< $cfg

  echo dataset=${dataset} arch=${arch} → GPU=${gpu} lr=${LR[$dataset]}

  #  GPU 
  read -u 3 gpu

  {
    run_name=${dataset}_${arch}
    outdir=checkpoints/pre_train_finetune/${run_name}
    mkdir -p $outdir
    logfile=${outdir}.log

    

    CUDA_VISIBLE_DEVICES=$gpu nohup python -u main.py \
      --dataset            $dataset \
      --data-path          $DATA_ROOT \
      --arch               $arch \
      --unlearn-method     pre_train \
      --pretrained \
      --epochs-or-steps    10 \
      --batch-size         128 \
      --lr                 ${LR[$dataset]} \
      --val-ratio          0.1 \
      --seed               1234 \
      --gpu-id             0 \
      --save-model \
      > $logfile 2>&1

    #  GPU 
    echo $gpu >&3
  } &
done

# 
wait

# 
exec 3>&-
exec 3<&-

echo All ViT fine-tuning experiments finished

