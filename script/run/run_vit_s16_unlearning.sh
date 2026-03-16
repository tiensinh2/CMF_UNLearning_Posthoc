#!/usr/bin/env bash
set -euo pipefail

########################################
# Basic configuration
########################################

# Data root directory (default ~/data, can be passed at runtime)
DATA_ROOT=${1:-$HOME/data}

# Available GPU listOne experiment uses one GPU
GPUS=( 6 7)

# dataset;arch;classes
#  vit_s_16
EXPS=(
  # -------------------------
  # CIFAR10 - 3-class forget
  # -------------------------
  cifar10;vit_s_16;0,1,2
  cifar10;vit_s_16;3,4,5
  cifar10;vit_s_16;6,7,8
  cifar10;vit_s_16;0,5,9
  cifar10;vit_s_16;2,4,8

  # -------------------------
  # CIFAR10 - 1-class forget (0-9 sweep)
  # -------------------------
  cifar10;vit_s_16;0
  cifar10;vit_s_16;1
  cifar10;vit_s_16;2
  cifar10;vit_s_16;3
  cifar10;vit_s_16;4
  cifar10;vit_s_16;5
  cifar10;vit_s_16;6
  cifar10;vit_s_16;7
  cifar10;vit_s_16;8
  cifar10;vit_s_16;9


  # -------------------------
  # CIFAR100 - 10-class forget (your 5 groups)
  # -------------------------
  cifar100;vit_s_16;3,15,19,21,31,38,42,43,88,97
  cifar100;vit_s_16;47,52,54,56,59,62,70,82,92,96
  cifar100;vit_s_16;5,20,22,25,39,40,84,86,87,94
  cifar100;vit_s_16;8,13,41,48,59,69,81,85,89,90
  cifar100;vit_s_16;1,4,30,32,55,67,72,73,91,95

  # -------------------------
  # CIFAR100 - 1-class forget (0,1,2,3,5)
  # -------------------------
  cifar100;vit_s_16;0
  cifar100;vit_s_16;1
  cifar100;vit_s_16;2
  cifar100;vit_s_16;3
  cifar100;vit_s_16;5

  # -------------------------
  # tinyimagenet 
  # -------------------------
  tinyimagenet;vit_s_16;2
  tinyimagenet;vit_s_16;3
  tinyimagenet;vit_s_16;5
  tinyimagenet;vit_s_16;7
  tinyimagenet;vit_s_16;9
  tinyimagenet;vit_s_16;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19
  tinyimagenet;vit_s_16;20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39
  tinyimagenet;vit_s_16;40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59
  tinyimagenet;vit_s_16;60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79
  tinyimagenet;vit_s_16;80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99
)
#  & 
declare -A TOTAL PER_CLASS
TOTAL[cifar10]=50000      PER_CLASS[cifar10]=5000
TOTAL[cifar100]=50000     PER_CLASS[cifar100]=500
TOTAL[tinyimagenet]=100000 PER_CLASS[tinyimagenet]=500   # 200  * 500 

#  dataset
declare -A LR
LR[cifar10]=3e-4
LR[cifar100]=3e-4
LR[tinyimagenet]=1e-4

#  unlearning 
METHODS=(
  retrain
  grad_ascent_descent
  #salun
  #random_label
)

########################################
#  GPU 
########################################

FIFO=/tmp/$$.fifo
mkfifo $FIFO
exec 3<>$FIFO
rm $FIFO

# Initialize token pool
for gpu in ${GPUS[@]}; do
  echo $gpu >&3
done

########################################
# 
########################################

for method in ${METHODS[@]}; do
  for cfg in ${EXPS[@]}; do
    IFS=; read -r dataset arch classes <<< $cfg

    #  forget / retain 
    IFS=',' read -ra cls_arr <<< $classes
    n_forget=${#cls_arr[@]}
    per=${PER_CLASS[$dataset]}
    total=${TOTAL[$dataset]}
    forget=$(( n_forget * per ))
    retain=$(( total - forget ))

    # 
    case $method in
      retrain)
        epochs=10
        lr=${LR[$dataset]}
        extra_flags=
        out_root=checkpoints/retrain
        ;;
      grad_ascent_descent)
        epochs=3
        lr=${LR[$dataset]}
        extra_flags=--grad-norm-clip 1.0
        out_root=checkpoints/grad_ascent_descent
        ;;

      salun)
        epochs=5
        lr=${LR[$dataset]}
        extra_flags=--salun-threshold 0.5 --lp_every 100 --grad-norm-clip 1.0
        out_root=checkpoints/salun
        ;;

      random_label)
        epochs=5
        lr=${LR[$dataset]}
        extra_flags=--grad-norm-clip 1.0
        out_root=checkpoints/random_label
        ;;

      *)
        echo Unknown method: $method
        exit 1
        ;;
    esac

    # 
    cls_name=${classes//,/_}
    run_name=${dataset}_${arch}
    outdir=${out_root}/${run_name}
    mkdir -p $outdir
    logfile=${outdir}/${cls_name}.log

    #  GPU 
    read -u 3 gpu

    {
      echo Launching: method=$method dataset=$dataset arch=$arch classes=$classes on GPU=$gpu
      echo   retain=$retain forget=$forget epochs=$epochs lr=$lr

      CUDA_VISIBLE_DEVICES=$gpu nohup python -u main.py \
        --dataset            $dataset \
        --data-path          $DATA_ROOT \
        --arch               $arch \
        --unlearn-method     $method \
        --epochs-or-steps    $epochs \
        --batch-size         128 \
        --lr                 $lr \
        --num-retain-samples $retain \
        --num-forget-samples $forget \
        --unlearn-class      $classes \
        --gpu-id 0 \
        $extra_flags \
        --save-model \
        --lp_every 0 \
        > $logfile 2>&1

      #  GPU 
      echo $gpu >&3
    } &
  done
done

# 
wait

# 
exec 3>&-
exec 3<&-

echo All ViT-Small unlearning experiments finished.
