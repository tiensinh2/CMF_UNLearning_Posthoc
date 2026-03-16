#!/usr/bin/env bash
set -euo pipefail

########################################
# Basic configuration
########################################
DATA_ROOT=${1:-$HOME/data}
PY=${PYTHON:-python}

# You can only run 4-6 at a timeHere is 6 GPUs
GPUS=(6 7)

# Whether to initialize retrain from pretrainedKeep 0 if you don't need it
USE_PRETRAINED=0

########################################
#  tinyimagenet + resnet50
########################################
METHODS=(retrain)

# dataset;arch;classes    3 
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
  tinyimagenet;resnet50;2
  tinyimagenet;resnet50;3
  tinyimagenet;resnet50;5
  tinyimagenet;resnet50;7
  tinyimagenet;resnet50;9

  tinyimagenet;resnet50;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19
  tinyimagenet;resnet50;20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39
  tinyimagenet;resnet50;40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59
  tinyimagenet;resnet50;60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79
  tinyimagenet;resnet50;80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99
)

########################################
# / sanity check
# Train len 90000, Val 10000, Test 10000
# total  retain/forget “” 100000trainval
########################################
declare -A TOTAL PER_CLASS
TOTAL[tinyimagenet]=100000
PER_CLASS[tinyimagenet]=500

########################################
# Learning rate method|dataset|single/multi
########################################
declare -A LR
LR[retrain|tinyimagenet|single]=0.05
LR[retrain|tinyimagenet|multi]=0.05

########################################
# GPU  = GPU 
########################################
FIFO=/tmp/$$.fifo
mkfifo $FIFO
exec 3<>$FIFO
rm $FIFO
for gpu in ${GPUS[@]}; do echo $gpu >&3; done

########################################
# 
########################################
for method in ${METHODS[@]}; do
  for cfg in ${EXPS[@]}; do
    IFS=; read -r dataset arch classes <<< $cfg

    IFS=',' read -ra cls_arr <<< $classes
    n_forget=${#cls_arr[@]}
    if [[ $n_forget -eq 1 ]]; then
      forget_mode=single
    else
      forget_mode=multi
    fi

    per=${PER_CLASS[$dataset]}
    total=${TOTAL[$dataset]}
    forget=$(( n_forget * per ))
    retain=$(( total - forget ))

    # ========== retrain  hyper parameter  setting ==========
    case $method in
      retrain)
        epochs=300
        extra_flags=\
          --seed 1234 \
          --momentum 0.9 \
          --weight-decay 1e-4 \
          --patience 75 \
          --val-ratio 0.1 \
          --lr-scheduler cosine \
          --warmup-epochs 5 \
          --min-lr 1e-5 \
          --group-name Retrain \
          --project-name Retrain \
        
        out_root=checkpoints/retrain
        ;;
      *)
        echo Unknown method: $method
        exit 1
        ;;
    esac

    #  lr
    lr_key=${method}|${dataset}|${forget_mode}
    lr=${LR[$lr_key]:-}
    if [[ -z $lr ]]; then
      echo [ERROR] LR not set for key: $lr_key
      exit 1
    fi

    cls_name=${classes//,/_}
    run_name=${dataset}_${arch}
    outdir=${out_root}/${run_name}
    mkdir -p $outdir

    # log dir  checkpoint dir  outdir 
    logfile=${outdir}/${cls_name}.log

    read -u 3 gpu

    {
      echo Launching: method=$method dataset=$dataset arch=$arch classes=$classes on GPU=$gpu
      echo retain=$retain forget=$forget n_forget=$n_forget mode=$forget_mode epochs=$epochs lr=$lr
      echo logfile=$logfile outdir=$outdir

      pretrained_flag=
      if [[ $USE_PRETRAINED -eq 1 ]]; then
        pretrained_flag=--pretrained
      fi

      CUDA_VISIBLE_DEVICES=$gpu nohup $PY -u main.py \
        --dataset            $dataset \
        --data-path          $DATA_ROOT \
        --arch               $arch \
        --unlearn-method     $method \
        --epochs-or-steps    $epochs \
        --batch-size         128 \
        --test-batch-size    256 \
        --lr                 $lr \
        --num-retain-samples $retain \
        --num-forget-samples $forget \
        --unlearn-class      $classes \
        --gpu-id             0 \
        $extra_flags \
        $pretrained_flag \
        --save-model \
        --lp_every 0 \
        > $logfile 2>&1

      echo Done: $cfg on GPU=$gpu
      echo $gpu >&3
    } &

  done
done

wait
exec 3>&-
exec 3<&-

echo All retrain tinyimagenet resnet50 jobs finished.
