#!/usr/bin/env bash
set -euo pipefail

########################################
# Basic configuration
########################################

DATA_ROOT=${1:-$HOME/data}

GPUS=(0 1 2 3)

# Unified archViT-S/16
ARCH=vit_s_16

EXPS=(
  # -------------------------
  # CIFAR10 - 3-class forget
  # -------------------------
  cifar10;${ARCH};0,1,2
  cifar10;${ARCH};3,4,5
  #cifar10;${ARCH};6,7,8
  #cifar10;${ARCH};0,5,9
  #cifar10;${ARCH};2,4,8

  # -------------------------
  # CIFAR10 - 1-class forget (0-9 sweep)
  # -------------------------
  cifar10;${ARCH};0
  cifar10;${ARCH};1
  #cifar10;${ARCH};2
  #cifar10;${ARCH};3
  #cifar10;${ARCH};4
  #cifar10;${ARCH};5
  #cifar10;${ARCH};6
  #cifar10;${ARCH};7
  #cifar10;${ARCH};8
  #cifar10;${ARCH};9

  # -------------------------
  # CIFAR100 - 10-class forget
  # -------------------------
  cifar100;${ARCH};3,15,19,21,31,38,42,43,88,97
  cifar100;${ARCH};47,52,54,56,59,62,70,82,92,96
  #cifar100;${ARCH};5,20,22,25,39,40,84,86,87,94
  #cifar100;${ARCH};8,13,41,48,59,69,81,85,89,90
  #cifar100;${ARCH};1,4,30,32,55,67,72,73,91,95

  # -------------------------
  # CIFAR100 - 1-class forget
  # -------------------------
  cifar100;${ARCH};0
  cifar100;${ARCH};1
  #cifar100;${ARCH};2
  #cifar100;${ARCH};3
  #cifar100;${ARCH};5

  # -------------------------
  # tinyimagenet
  # -------------------------
  tinyimagenet;${ARCH};0
  tinyimagenet;${ARCH};1
  #tinyimagenet;${ARCH};2
  #tinyimagenet;${ARCH};3
  #tinyimagenet;${ARCH};5
  #tinyimagenet;${ARCH};7
  #tinyimagenet;${ARCH};9
  tinyimagenet;${ARCH};0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19
  tinyimagenet;${ARCH};20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39
  #tinyimagenet;${ARCH};40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59
  #tinyimagenet;${ARCH};60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79
  #tinyimagenet;${ARCH};80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99
)

declare -A TOTAL PER_CLASS
TOTAL[cifar10]=50000        PER_CLASS[cifar10]=5000
TOTAL[cifar100]=50000       PER_CLASS[cifar100]=500
TOTAL[tinyimagenet]=100000  PER_CLASS[tinyimagenet]=500

# ======== Method list ========
METHODS=(
  random_label_CMF_RemoveFC
  salun_CMF_RemoveFC
  grad_ascent_descent_CMF_RemoveFC
)

########################################
# LR & EPOCHS method + dataset + (single/multi)
# key: method|dataset|single  or  method|dataset|multi
########################################
declare -A LR
declare -A EPOCHS

# ---- random_label_CMF_RemoveFC ----
LR[random_label_CMF_RemoveFC|cifar10|single]=1e-3
LR[random_label_CMF_RemoveFC|cifar10|multi]=1e-3
LR[random_label_CMF_RemoveFC|cifar100|single]=2e-2
LR[random_label_CMF_RemoveFC|cifar100|multi]=1e-2
LR[random_label_CMF_RemoveFC|tinyimagenet|single]=3e-3
LR[random_label_CMF_RemoveFC|tinyimagenet|multi]=3e-3

EPOCHS[random_label_CMF_RemoveFC|cifar10|single]=3
EPOCHS[random_label_CMF_RemoveFC|cifar10|multi]=3
EPOCHS[random_label_CMF_RemoveFC|cifar100|single]=3
EPOCHS[random_label_CMF_RemoveFC|cifar100|multi]=3
EPOCHS[random_label_CMF_RemoveFC|tinyimagenet|single]=3
EPOCHS[random_label_CMF_RemoveFC|tinyimagenet|multi]=3

# ---- salun_CMF_RemoveFC ----
LR[salun_CMF_RemoveFC|cifar10|single]=8e-4
LR[salun_CMF_RemoveFC|cifar10|multi]=1e-3
LR[salun_CMF_RemoveFC|cifar100|single]=2e-2
LR[salun_CMF_RemoveFC|cifar100|multi]=1e-2
LR[salun_CMF_RemoveFC|tinyimagenet|single]=2e-3
LR[salun_CMF_RemoveFC|tinyimagenet|multi]=2e-3

EPOCHS[salun_CMF_RemoveFC|cifar10|single]=3
EPOCHS[salun_CMF_RemoveFC|cifar10|multi]=3
EPOCHS[salun_CMF_RemoveFC|cifar100|single]=3
EPOCHS[salun_CMF_RemoveFC|cifar100|multi]=3
EPOCHS[salun_CMF_RemoveFC|tinyimagenet|single]=3
EPOCHS[salun_CMF_RemoveFC|tinyimagenet|multi]=3

# ---- grad_ascent_descent_CMF_RemoveFC ( METHODS ) ----
LR[grad_ascent_descent_CMF_RemoveFC|cifar10|single]=5e-5
LR[grad_ascent_descent_CMF_RemoveFC|cifar10|multi]=5e-4
LR[grad_ascent_descent_CMF_RemoveFC|cifar100|single]=5e-5
LR[grad_ascent_descent_CMF_RemoveFC|cifar100|multi]=5e-4
LR[grad_ascent_descent_CMF_RemoveFC|tinyimagenet|single]=1e-5
LR[grad_ascent_descent_CMF_RemoveFC|tinyimagenet|multi]=1e-2

EPOCHS[grad_ascent_descent_CMF_RemoveFC|cifar10|single]=5
EPOCHS[grad_ascent_descent_CMF_RemoveFC|cifar10|multi]=3
EPOCHS[grad_ascent_descent_CMF_RemoveFC|cifar100|single]=3
EPOCHS[grad_ascent_descent_CMF_RemoveFC|cifar100|multi]=3
EPOCHS[grad_ascent_descent_CMF_RemoveFC|tinyimagenet|single]=3
EPOCHS[grad_ascent_descent_CMF_RemoveFC|tinyimagenet|multi]=1


########################################
# GPU 
########################################
FIFO=/tmp/$$.fifo
mkfifo $FIFO
exec 3<>$FIFO
rm $FIFO

for gpu in ${GPUS[@]}; do
  echo $gpu >&3
done

########################################
# 
########################################
for method in ${METHODS[@]}; do
  for cfg in ${EXPS[@]}; do
    IFS=; read -r dataset arch classes <<< $cfg

    IFS=',' read -ra cls_arr <<< $classes
    n_forget=${#cls_arr[@]}

    per=${PER_CLASS[$dataset]}
    total=${TOTAL[$dataset]}
    forget=$(( n_forget * per ))
    retain=$(( total - forget ))

    # single / multi
    if [[ $n_forget -eq 1 ]]; then
      sm=single
    else
      sm=multi
    fi

    #  flags / out_root
    case $method in
      random_label_CMF_RemoveFC)
        # random label clip 
        extra_flags=--grad-norm-clip 0.5
        out_root=checkpoints/random_label_CMF_RemoveFC
        ;;
      salun_CMF_RemoveFC)
        extra_flags=--salun-threshold 0.5 --grad-norm-clip 1.0
        out_root=checkpoints/salun_CMF_RemoveFC
        ;;
      grad_ascent_descent_CMF_RemoveFC)
        extra_flags=--grad-norm-clip 1.0
        out_root=checkpoints/grad_ascent_descent_CMF_RemoveFC
        ;;
      *)
        echo Unknown method: $method
        exit 1
        ;;
    esac

    #  lr & epochsmethod + dataset + single/multi
    key=${method}|${dataset}|${sm}
    lr=${LR[$key]:-}
    epochs=${EPOCHS[$key]:-}

    if [[ -z $lr ]]; then
      echo [ERROR] LR not set for key: $key
      exit 1
    fi
    if [[ -z $epochs ]]; then
      echo [ERROR] EPOCHS not set for key: $key
      exit 1
    fi

    cls_name=${classes//,/_}
    run_name=${dataset}_${arch}
    outdir=${out_root}/${run_name}
    mkdir -p $outdir
    logfile=${outdir}/${cls_name}.log

    read -u 3 gpu

    {
      echo Launching: method=$method dataset=$dataset arch=$arch classes=$classes on GPU=$gpu
      echo mode=$sm retain=$retain forget=$forget epochs=$epochs lr=$lr
      echo logfile=$logfile

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
        --lp_every 0 \
        --remove_FC \
        --CMFClassifier \
        --pretrained \
        > $logfile 2>&1

      echo $gpu >&3
    } &
  done
done

wait
exec 3>&-
exec 3<&-

echo All CMF ViT-S/16 unlearning experiments finished.
