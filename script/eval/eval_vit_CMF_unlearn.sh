#!/usr/bin/env bash
set -euo pipefail

# GPUs
GPUS=(0 1 
)

# Unified ViT-Small
ARCH=vit_s_16

# dataset;classes
EXPS=(
  # -------------------------
  # CIFAR10 - 3-class forget
  # -------------------------
  #cifar10;0,1,2
  #cifar10;3,4,5
  #cifar10;6,7,8
  #cifar10;0,5,9
  #cifar10;2,4,8

  # -------------------------
  # CIFAR10 - 1-class forget (0-9 sweep)
  # -------------------------
  #cifar10;0
  #cifar10;1
  #cifar10;2
  #cifar10;3
  #cifar10;4
  #cifar10;5
  #cifar10;6
  #cifar10;7
  #cifar10;8
  #cifar10;9

  # -------------------------
  # CIFAR100 - 10-class forget (your 5 groups)
  # -------------------------
  cifar100;3,15,19,21,31,38,42,43,88,97
  cifar100;47,52,54,56,59,62,70,82,92,96
  cifar100;5,20,22,25,39,40,84,86,87,94
  cifar100;8,13,41,48,59,69,81,85,89,90
  cifar100;1,4,30,32,55,67,72,73,91,95

  # -------------------------
  # CIFAR100 - 1-class forget (0,1,2,3,5)
  # -------------------------
  cifar100;0
  cifar100;1
  cifar100;2
  cifar100;3
  cifar100;5

  # -------------------------
  # tinyimagenet 
  # -------------------------
  #tinyimagenet;2
  #tinyimagenet;3
  #tinyimagenet;5
  #tinyimagenet;7
  #tinyimagenet;9
  #tinyimagenet;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19
  #tinyimagenet;20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39
  #tinyimagenet;40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59
  #tinyimagenet;60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79
  #tinyimagenet;80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99
)

# totals
declare -A TOTAL PER_CLASS
TOTAL[cifar10]=50000         PER_CLASS[cifar10]=5000
TOTAL[cifar100]=50000        PER_CLASS[cifar100]=500
TOTAL[tinyimagenet]=100000   PER_CLASS[tinyimagenet]=500

# eval methods (normal)
METHODS=(
  random_label_CMF_RemoveFC
  salun_CMF_RemoveFC
  grad_ascent_descent_CMF_RemoveFC
)

# token queue
FIFO=/tmp/$$.fifo
mkfifo $FIFO
exec 3<>$FIFO
rm $FIFO
for gpu in ${GPUS[@]}; do echo $gpu >&3; done

for method in ${METHODS[@]}; do
  for cfg in ${EXPS[@]}; do
    IFS=; read -r dataset classes <<< $cfg

    # compute retain/forget
    IFS=',' read -ra cls_arr <<< $classes
    n_forget=${#cls_arr[@]}
    per=${PER_CLASS[$dataset]}
    total=${TOTAL[$dataset]}
    forget=$(( n_forget * per ))
    retain=$(( total - forget ))

    cls_name=${classes//,/_}
    outdir=evaluations/${method}/${dataset}_${ARCH}
    mkdir -p $outdir
    logfile=${outdir}/${cls_name}.log

    read -u 3 gpu

    {
      echo Launching eval: method=$method dataset=$dataset arch=$ARCH classes=$classes GPU=$gpu
      CUDA_VISIBLE_DEVICES=$gpu nohup python -u evaluation.py \
        --dataset $dataset \
        --arch $ARCH \
        --unlearn-method $method \
        --epochs-or-steps 10 \
        --batch-size 128 \
        --lr 3e-4 \
        --num-retain-samples $retain \
        --num-forget-samples $forget \
        --unlearn-class $classes \
        --save-model \
        --do-linear-probe \
        --prob-batch-size 128 \
        --save-model \
        --remove_FC \
        --use-last-only \
        --do-ncc-mismatch \
        > $logfile 2>&1

      echo $gpu >&3
    } &
  done
done

wait
exec 3>&-
exec 3<&-

echo All CMF ViT evaluations finished.
