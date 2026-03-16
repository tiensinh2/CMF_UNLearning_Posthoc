#!/usr/bin/env bash
set -euo pipefail

########################################
# Basic configuration
########################################

DATA_ROOT=${1:-$HOME/data}

# Available GPU list
GPUS=(4 5 6 7)

# Whether to initialize retrain from pretrainedold scripts don't have 1
USE_PRETRAINED=0

# Experiment configuration: dataset;arch;classes
EXPS=(
  # CIFAR10 - 3-class forget
  #cifar10;resnet18;0,1,2
  #cifar10;resnet18;3,4,5
  #cifar10;resnet18;6,7,8
  #cifar10;resnet18;0,5,9
  #cifar10;resnet18;2,4,8

  # CIFAR10 - 1-class forget (0-9 sweep)
  cifar10;resnet18;0
  cifar10;resnet18;1
  cifar10;resnet18;2
  cifar10;resnet18;3
  cifar10;resnet18;4
  cifar10;resnet18;5
  cifar10;resnet18;6
  cifar10;resnet18;7
  cifar10;resnet18;8
  cifar10;resnet18;9

  # CIFAR100 - 10-class forget
  #cifar100;resnet18;3,15,19,21,31,38,42,43,88,97
  #cifar100;resnet18;47,52,54,56,59,62,70,82,92,96
  #cifar100;resnet18;5,20,22,25,39,40,84,86,87,94
  #cifar100;resnet18;8,13,41,48,59,69,81,85,89,90
  #cifar100;resnet18;1,4,30,32,55,67,72,73,91,95

  # CIFAR100 - 1-class forget
  #cifar100;resnet18;0
  #cifar100;resnet18;1
  #cifar100;resnet18;2
  #cifar100;resnet18;3
  #cifar100;resnet18;5

  # -------------------------
  # tinyimagenet 
  # -------------------------
  # tinyimagenet + resnet50
  #tinyimagenet;resnet50;2
  #tinyimagenet;resnet50;3
  #tinyimagenet;resnet50;5
  #tinyimagenet;resnet50;7
  #tinyimagenet;resnet50;9
  #tinyimagenet;resnet50;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19
  #tinyimagenet;resnet50;20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39
  #tinyimagenet;resnet50;40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59
  #tinyimagenet;resnet50;60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79
  #tinyimagenet;resnet50;80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99


)

# Total dataset samples & samples per class
declare -A TOTAL PER_CLASS
TOTAL[cifar10]=50000   PER_CLASS[cifar10]=5000
TOTAL[cifar100]=50000  PER_CLASS[cifar100]=500
TOTAL[tinyimagenet]=100000  PER_CLASS[tinyimagenet]=500

# ========  retrain========
METHODS=(
  #retrain
  #pre-train
  #random_label
  #salun
  #grad_ascent_descent
  #grad_descent
  #SVD
  tarun
  #scrub


)

########################################
# Learning rate method|dataset|single/multi 
########################################
declare -A LR

# ---- retrain ----
LR[retrain|cifar10|single]=1e-2
LR[retrain|cifar10|multi]=1e-2
LR[retrain|cifar100|single]=1e-2
LR[retrain|cifar100|multi]=1e-2
LR[retrain|tinyimagenet|single]=0.05
LR[retrain|tinyimagenet|multi]=0.05


# ---- random_label ----
LR[random_label|cifar10|single]=1e-2
LR[random_label|cifar10|multi]=1e-2
LR[random_label|cifar100|single]=3e-3
LR[random_label|cifar100|multi]=3e-3
LR[random_label|tinyimagenet|single]=5e-4
LR[random_label|tinyimagenet|multi]=5e-4

# ---- salun ----
LR[salun|cifar10|single]=1e-2
LR[salun|cifar10|multi]=1e-2
LR[salun|cifar100|single]=3e-3
LR[salun|cifar100|multi]=3e-3   
LR[salun|tinyimagenet|single]=5e-4
LR[salun|tinyimagenet|multi]=5e-4

# ---- grad_ascent_descent ----
LR[grad_ascent_descent|cifar10|single]=1e-3
LR[grad_ascent_descent|cifar10|multi]=1e-3
LR[grad_ascent_descent|cifar100|single]=5e-5
LR[grad_ascent_descent|cifar100|multi]=1e-4   
LR[grad_ascent_descent|tinyimagenet|single]=5e-4
LR[grad_ascent_descent|tinyimagenet|multi]=5e-4

# ---- grad_descent ----
LR[grad_descent|cifar10|single]=1e-3
LR[grad_descent|cifar10|multi]=1e-3
LR[grad_descent|cifar100|single]=1e-3
LR[grad_descent|cifar100|multi]=1e-3   
LR[grad_descent|tinyimagenet|single]=1e-3
LR[grad_descent|tinyimagenet|multi]=1e-3

# ---- SVD ----
LR[SVD|cifar10|single]=1e-2
LR[SVD|cifar10|multi]=1e-2
LR[SVD|cifar100|single]=1e-2
LR[SVD|cifar100|multi]=1e-2
LR[SVD|tinyimagenet|single]=1e-3
LR[SVD|tinyimagenet|multi]=1e-3


# ---- Tarun ----
#for normal
#LR[tarun|cifar10|single]=5e-5
#for classifier only
LR[tarun|cifar10|single]=2e-3
LR[tarun|cifar10|multi]=5e-5
LR[tarun|cifar100|single]=3e-5
LR[tarun|cifar100|multi]=3e-5
LR[tarun|tinyimagenet|single]=2e-5
LR[tarun|tinyimagenet|multi]=2e-5


LR[scrub|cifar10|single]=1e-4
LR[scrub|cifar10|multi]=1e-4
LR[scrub|cifar100|single]=1e-3
LR[scrub|cifar100|multi]=3e-4
LR[scrub|tinyimagenet|single]=5e-3
LR[scrub|tinyimagenet|multi]=1e-3

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

    #  forget / retain 
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

    # 
    case $method in
      grad_ascent_descent)
        epochs=3
        batch_size=128
        extra_flags=--grad-norm-clip 1.0
        out_root=checkpoints/grad_ascent_descent
        ;;

      grad_descent)
        epochs=3
        batch_size=128
        extra_flags=
        out_root=checkpoints/grad_descent
        ;;
      salun)
        epochs=3
        batch_size=128
        extra_flags=--salun-threshold 0.5 
        out_root=checkpoints/salun
        ;;

      random_label)
        epochs=3
        batch_size=128
        extra_flags=
        out_root=checkpoints/random_label
        ;;

      SVD)
        epochs=200
        batch_size=64
        if [[ $dataset == cifar10 ]]; then
          svd_alpha_r=100
          svd_alpha_f=3
          svd_samples=900
          svd_max_patches=10000
        elif [[ $dataset == cifar100 ]]; then
          svd_alpha_r=1000
          svd_alpha_f=30
          svd_samples=990
          svd_max_patches=10000
        elif [[ $dataset == tinyimagenet ]]; then
          svd_alpha_r=30
          svd_alpha_f=10
          svd_samples=999
          svd_max_patches=10000
        else
          echo Unsupported dataset for SVD hyperparameters: $dataset
          exit 1
        fi
        extra_flags=--SVD-alpha-r $svd_alpha_r --SVD-alpha-f $svd_alpha_f --SVD-samples $svd_samples --SVD-max-patches $svd_max_patches
        out_root=checkpoints/SVD
        ;;
      tarun)
        # ---- Tarun / UNSIR style ----
        epochs=3
        batch_size=128
        # dataset-specific hyperparams tinyimagenet 
        if [[ $dataset == tinyimagenet ]]; then
          tarun_impair_lr=2e-4
          tarun_samples_per_class=1000
        elif [[ $dataset == cifar10 ]]; then
          tarun_impair_lr=1e-4
          tarun_samples_per_class=1000
        elif [[ $dataset == cifar100 ]]; then
          tarun_impair_lr=2e-4
          tarun_samples_per_class=1000
        else
          echo Unsupported dataset for tarun hyperparameters: $dataset
          exit 1
        fi

        # Tarun  flags argparse 
        extra_flags=--tarun-impair-lr $tarun_impair_lr --tarun-samples-per-class $tarun_samples_per_class
        out_root=checkpoints/tarun
        ;;
      scrub)
        # ========== SCRUB ==========
        epochs=3            #  scrub-epochs loop
        batch_size=64       #  SGDA  scrub-sgda-bsz 

        extra_flags=
          --scrub-del-bsz 64
          --scrub-sgda-bsz 64
          --scrub-msteps 2
          --scrub-epochs 3
        

        out_root=checkpoints/scrub
        ;;
      *)
        echo Unknown method: $method
        exit 1
        ;;
    esac

    key=${method}|${dataset}|${sm}
    lr=${LR[$key]:-}
    

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
      if [[ $method == SVD ]]; then
        echo   SVD-alpha-r=$svd_alpha_r SVD-alpha-f=$svd_alpha_f SVD-samples=$svd_samples SVD-max-patches=$svd_max_patches
      fi

      CUDA_VISIBLE_DEVICES=$gpu nohup python -u main.py \
        --dataset            $dataset \
        --data-path          $DATA_ROOT \
        --arch               $arch \
        --unlearn-method     $method \
        --epochs-or-steps    $epochs \
        --batch-size         $batch_size \
        --lr                 $lr \
        --num-retain-samples $retain \
        --num-forget-samples $forget \
        --unlearn-class      $classes \
        --gpu-id 0 \
        $extra_flags \
        --save-model \
        --lp_every 0 \
        --zero_last_layer \
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

echo All Resnet unlearning experiments finished.
