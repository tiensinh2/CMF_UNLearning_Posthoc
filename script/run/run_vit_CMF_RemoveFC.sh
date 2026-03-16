#!/usr/bin/env bash
set -euo pipefail

GPUS=(0 1 4 7)
#DATASETS=(cifar10 cifar100 tinyimagenet)
#ARCH=(vit_s_16)

DATASETS=(tinyimagenet)
ARCH=resnet50

declare -A TOTAL
TOTAL[cifar10]=50000
TOTAL[cifar100]=50000
TOTAL[tinyimagenet]=100000

declare -A LR
LR[cifar10]=3e-4
LR[cifar100]=3e-4
LR[tinyimagenet]=1e-4

declare -A EPOCHS
EPOCHS[cifar10]=1
EPOCHS[cifar100]=1
EPOCHS[tinyimagenet]=1

# token queue
FIFO=/tmp/$$.fifo
mkfifo $FIFO
exec 3<>$FIFO
rm $FIFO

for gpu in ${GPUS[@]}; do
  echo $gpu >&3
done

for dataset in ${DATASETS[@]}; do
  ckpt=./checkpoints/CMF_FT_RemoveFC/${dataset}_${ARCH}.pt
  log=./checkpoints/CMF_FT_RemoveFC/${dataset}_${ARCH}.log
  mkdir -p ./checkpoints/CMF_FT_RemoveFC

  read -u 3 gpu

  {
    echo Launching CMF_FT_RemoveFC on ${dataset} ${ARCH} (GPU ${gpu})
    echo Will save to: ${ckpt}

    CUDA_VISIBLE_DEVICES=$gpu nohup python main.py \
      --dataset $dataset \
      --arch $ARCH \
      --unlearn-method CMF_FT_RemoveFC \
      --epochs-or-steps ${EPOCHS[$dataset]} \
      --batch-size 128 \
      --lr ${LR[$dataset]} \
      --weight-decay 1e-4 \
      --save-model \
      --remove_FC \
      --CMFClassifier \
      --pretrained \
      --num-retain-samples ${TOTAL[$dataset]} \
      --num-forget-samples 0 \
      --unlearn-class 0 \
      > $log 2>&1

    echo $gpu >&3
  } &

done

wait
exec 3>&-
exec 3<&-

echo All CMF ViT fine-tuning experiments finished.
