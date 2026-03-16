import argparse
import torch
import torch.optim as optim
import random
from utils import get_retain_forget_partition, get_dataset, get_model, test, SVC_MIA
import evaluation
from evaluation.tsne import tsne_visualize
from evaluation.mia import evaluate_mia, evaluate_feature_mia
from evaluation.linear_prob_whole import run_linear_probe_both, plot_LP
import os
import numpy as np
from unlearn import unlear_func
from train import train
import json
import copy
import pandas as pd

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def main():
    parser = argparse.ArgumentParser(description='PyTorch cifar10 Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--dataset', type=str, default="cifar10",
                        help='')
    parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs-or-steps', type=int, default=350, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='LR',
                        help='')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='LR',
                        help='')
    parser.add_argument('--gamma', type=float, default=0.5, metavar='M',
                        help='Learning rate step gamma (default: 0.7) after 50 epochs')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--no-train-transform', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    
    parser.add_argument('--data-path', type=str, default='../data/',
                        help='')
    parser.add_argument('--sub-set-mode', action='store_true', default=False,
                        help='use validation set')
    parser.add_argument('--sub-set-samples', type=int, default=10000, metavar='EPS',
                        help='')
    parser.add_argument('--val-ratio', type=float, default=0.1, metavar='R',
                        help='proportion of train set used for validation (default: 0.1)')
    parser.add_argument('--patience', type=int, default=None, metavar='P',
                        help='early stopping patience (number of epochs without improvement)')
    parser.add_argument('--save-path', type=str, default=None,
                        help='Custom checkpoint path; if omitted, use the built-in baseline path')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='Index of GPU to use when CUDA is available (0-7)')
    ### Unlearn parameters

    parser.add_argument('--num-retain-samples', type=int, default=45000,
                        help='') 
    parser.add_argument('--num-forget-samples', type=int, default=5000,
                        help='') 
    parser.add_argument('--grad-norm-clip', type=float, default=None,
                        help='')                        
    parser.add_argument('--unlearn-class', type=str, default="0, 1",
                        help='0')
    
    parser.add_argument('--arch', type=str, default='resnet18',
                        help='')
    parser.add_argument('--unlearn-method', type=str, default='pre_train',
                    choices=['pre_train', 'grad_ascent_descent', "grad_descent",
                             'salun', 'goel', 'ssd', 'scrub', 'scrub_CMF_RemoveFC', 'SVD', 'tarun', 'tarun_CMF_RemoveFC',
                             'random_label','random_label_once', 'retrain', 'prune', 
                             'grad_ascent_descent_CMF', 'grad_ascent_descent_CMF_RemoveFC',
                             'random_label_CMF_RemoveFC',  "CMF_FT_RemoveFC", "salun_CMF_RemoveFC"],              # <- add this
                    help='pick unlearning algorithm')

    parser.add_argument('--salun-threshold', type=float, default=0.1,
                        help='')

    parser.add_argument('--goel-exact', action="store_true", default=False,
                        help='')

    parser.add_argument('--ssd-lambda', type=float, default=1,
                        help='')
    parser.add_argument('--ssd-alpha', type=float, default=10,
                        help='')

    parser.add_argument('--scrub-del-bsz', type=int, default=512,
                        help='')
    parser.add_argument('--scrub-sgda-bsz', type=int, default=64,
                        help='')
    parser.add_argument('--scrub-msteps', type=int, default=2,
                        help='')
    parser.add_argument('--scrub-epochs', type=int, default=3,
                        help='')

    parser.add_argument('--SVD-alpha-r', type=int, default=100,
                        help='')
    parser.add_argument('--SVD-alpha-f', type=int, default=3,
                        help='')  
    parser.add_argument('--SVD-samples', type=int, default=900,
                        help='')  
    parser.add_argument('--SVD-max-patches', type=int, default=10000,
                        help='')  


    parser.add_argument('--tarun-impair-lr', type=float, default=2e-4,
                        help='')
    parser.add_argument('--tarun-samples-per-class', type=int, default=1000,
                        help='') 
    ### wandb parameters
    parser.add_argument('--project-name', type=str, default='baseline',
                        help='')
    parser.add_argument('--group-name', type=str, default='final',
                        help='')
    parser.add_argument('--multiclass', action='store_true', default=False,
                        help='For Saving the current Model')  
    parser.add_argument('--class-names', type=str, default=None,
                        help='')   
    
    
    # —— Unlearn parameters —— 
    parser.add_argument('--freeze_except_last', action='store_true', default=False,
                        help='if set, freeze all layers except the last one before unlearning')
    
    parser.add_argument('--zero_last_layer', action='store_true', default=False,
                        help='if set, zero out the last top‐level layer weights before training')
    # ——CMF parameters —— 
    parser.add_argument('--remove_FC', action='store_true', default=True,
                        help='if set, remove the fully connected layer from the model')

    # ----- SVC MIA parameters -----    
    parser.add_argument('--do-mia',action='store_true', default=False,
                        help='')   
    parser.add_argument('--do-mia-ulira',action='store_true', default=False,
                        help='')   
    parser.add_argument('--plot-mia-roc',action='store_true', default=False,
                        help='') 
    
    # ----- Linear probing parameters -----
    parser.add_argument('--do-linear-probe', action='store_true', default=False,
                        help='if set, perform linear probing on the penultimate layer')
    parser.add_argument('--use-last-only', action='store_true', default=False,
                        help='If set, run linear probe only on the last layer (skip full-layer probe)')
    parser.add_argument('--prob-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    

    # ----- t-SNE parameters -----
    parser.add_argument('--do-tsne', action='store_true', default=False,
                        help='run t-SNE visualization (features, optionally logits)')
    parser.add_argument('--tsne-perplexity', type=float, default=30.0)
    parser.add_argument('--tsne-max-points', type=int, default=10000)
    parser.add_argument('--tsne-pca-dim', type=int, default=50)
    parser.add_argument('--tsne-n-iter', type=int, default=1000)
    parser.add_argument('--tsne-seed', type=int, default=0)
    parser.add_argument('--tsne-no-logits', action='store_true', default=False,
                        help='if set, skip logits t-SNE plot')
    
    parser.add_argument('--do-nc', action='store_true',
                        help='measure NC-3 (feature-classifier alignment) and NCC (NC-4)')
    # ----- NCC mismatch (NC-4) -----
    parser.add_argument('--do-ncc-mismatch',
                        action='store_true', default=False,
                        help='if set, compute NCC accuracy/mismatch (NC-4) using CMF-aware features')
    
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use ImageNet-1K pretrained weights for ViT backbones."
    )

    args = parser.parse_args()
    args.train_transform = not args.no_train_transform
    if args.unlearn_class:
        args.unlearn_class=[int(val) for val in args.unlearn_class.split(",")]
    else:
        args.unlearn_class = []

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    #add new random seed setting with random label solution
    
    if use_cuda:
        torch.cuda.set_device(args.gpu_id)
        device = torch.device(f"cuda:{args.gpu_id}")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")


    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        train_kwargs.update({'num_workers': 16, 'pin_memory': True, 'shuffle': True})
        test_kwargs.update({'num_workers': 16, 'pin_memory': True})

    # Load full dataset.
    # build cifar
    if args.dataset in ["cifar10", "cifar100"]:
        dataset1, dataset2 = get_dataset(args)
        if args.unlearn_method == "pre_train":
            total_len = len(dataset1)
            val_len = int(total_len * args.val_ratio)
            train_len = total_len - val_len
            subset_train, subset_val = torch.utils.data.random_split(
                dataset1, [train_len, val_len],
                generator=torch.Generator().manual_seed(args.seed)
            )
            train_indices = subset_train.indices  # list or Tensor
            val_indices   = subset_val.indices

            train_dataset = copy.deepcopy(dataset1)
            val_dataset   = copy.deepcopy(dataset1)

            train_idx_arr = list(train_indices)
            val_idx_arr   = list(val_indices)

            train_dataset.data = dataset1.data[train_idx_arr]
            val_dataset.data   = dataset1.data[val_idx_arr]

            train_dataset.targets = [dataset1.targets[i] for i in train_idx_arr]
            val_dataset.targets   = [dataset1.targets[i] for i in val_idx_arr]

            test_dataset = dataset2
            
        else:
            train_dataset = dataset1
            val_dataset = None
            test_dataset = dataset2
    #build more need to complete
    else:
        dataset1, dataset2 = get_dataset(args)
        train_dataset = dataset1
        val_dataset = None
        test_dataset = dataset2

    
    retain_dataset, forget_dataset = get_retain_forget_partition(args, train_dataset, args.unlearn_class)
    
    
    if args.class_names is not None and args.multiclass: 
        args.unlearn_class = [args.class_label_names.index(val) for val in args.class_names.split(",")] 
        print(args.unlearn_class)
    # Partition into retain and forget dataset.
    
    sub_index= np.arange(len(train_dataset))
    if args.sub_set_mode:
        np.random.shuffle(sub_index)
        sub_index = sub_index[:args.sub_set_samples]
    sub_dataset = torch.utils.data.Subset(train_dataset, sub_index) 
    sub_loader = torch.utils.data.DataLoader(sub_dataset, **train_kwargs)


    
    model = get_model(args, device)
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(val_dataset, **train_kwargs)
    else:
        val_loader = None
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    retain_loader = torch.utils.data.DataLoader(retain_dataset, **train_kwargs)
    forget_loader = None if len(forget_dataset) == 0 else torch.utils.data.DataLoader(forget_dataset, **train_kwargs)
    
    
    
    suffix = ""
    if args.freeze_except_last:
        suffix += "_freeze"
    if getattr(args, "zero_last_layer", False):
        suffix += "_zero"
    method = args.unlearn_method + suffix
    
    if args.unlearn_method == "pre_train" or "CMF_FT" in args.unlearn_method:
        ckpt_path = f"./checkpoints/{args.unlearn_method}/{args.dataset}_{args.arch}.pt"
    else:
        ckpt_path = f"./checkpoints/{method}/{args.dataset}_{args.arch}/{','.join([str(v) for v in args.unlearn_class])}.pt"

    print(f"Checkpoint path: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Loaded model from {ckpt_path}")


    
    model.eval()
    #train_retain_acc, train_forget_acc, train_metric = test(
    #    model, device, train_loader, args.unlearn_class, args.class_label_names, args.num_classes,
    ##    job_name=args.unlearn_method, set_name="Final Train Set"
    #)
    test_retain_acc, test_forget_acc, test_metric = test(
        model, device, test_loader, args.unlearn_class, args.class_label_names, args.num_classes,
        job_name=args.unlearn_method, set_name="Final Test Set"
    )

    evaluation_results = {}    
    #evaluation_results["train_retain_acc"] = train_retain_acc
    #evaluation_results["train_forget_acc"] = train_forget_acc
    #evaluation_results["train_metric"] = train_metric
    evaluation_results["test_retain_acc"] = test_retain_acc
    evaluation_results["test_forget_acc"] = test_forget_acc
    evaluation_results["test_metric"] = test_metric

    if args.do_tsne:
        tsne_visualize(
            args, model, test_loader, device,
            out_root="result/plot_T_SNE",
            perplexity=args.tsne_perplexity,
            max_points=args.tsne_max_points,
            pca_dim=args.tsne_pca_dim,
            n_iter=args.tsne_n_iter,
            seed=args.tsne_seed,
            also_logits=not args.tsne_no_logits
        )

    if args.do_mia:
        classes_for_mia = list(set(args.unlearn_class))         
        '''''
        train_retain_dataset, train_forget_dataset, train_retain_index, train_forget_index = get_retain_forget_partition(args, dataset1, classes_for_mia, return_ind = True)
        train_retain_loader = torch.utils.data.DataLoader(train_retain_dataset,**test_kwargs)
        train_forget_loader = torch.utils.data.DataLoader(train_forget_dataset,**test_kwargs)
        test_retain_dataset, test_forget_dataset = get_retain_forget_partition(args, dataset2, classes_for_mia)
        test_forget_loader = torch.utils.data.DataLoader(test_forget_dataset,**test_kwargs)
        test_retain_loader = torch.utils.data.DataLoader(test_retain_dataset,**test_kwargs)
        evaluation_results["SVC_MIA_forget_efficacy"] = SVC_MIA(
                            shadow_train=train_retain_loader,
                            shadow_test=test_retain_loader,
                            target_train=None,
                            target_test=test_forget_loader,
                            model=model,
                        )
        '''''''''
        _, train_forget_dataset, _, _ = get_retain_forget_partition(
        args, dataset1, classes_for_mia, return_ind=True
        )
        _, test_forget_dataset = get_retain_forget_partition(args, dataset2, classes_for_mia)

        train_forget_loader = torch.utils.data.DataLoader(train_forget_dataset, **test_kwargs)
        test_forget_loader  = torch.utils.data.DataLoader(test_forget_dataset,  **test_kwargs)

        mia_stats = evaluate_mia(
            model, train_forget_loader, test_forget_loader,
            device, args=args, return_both=True
        )

        evaluation_results["MIA_forget_acc_percent"] = mia_stats["acc_percent"]
        evaluation_results["MIA_forget_norm0_100"]   = mia_stats["mia_0to100"]
        print(f"MIA forget-class acc: {mia_stats['acc_percent']:.2f}%, score: {mia_stats['mia_0to100']:.2f}/100")
        
        feat_mia_lr = evaluate_feature_mia(
        model, train_forget_loader, test_forget_loader,
        device, args=args, return_both=True, clf="lr"
        )
        evaluation_results["FeatureMIA_LR_acc_percent"] = feat_mia_lr["acc_percent"]
        evaluation_results["FeatureMIA_LR_norm0_100"]   = feat_mia_lr["mia_0to100"]
        print(f"Feature-MIA (LR) acc: {feat_mia_lr['acc_percent']:.2f}%, score: {feat_mia_lr['mia_0to100']:.2f}/100")

        # 2) SVC（RBF）
        feat_mia_svc = evaluate_feature_mia(
            model, train_forget_loader, test_forget_loader,
            device, args=args, return_both=True, clf="svc"
        )
        evaluation_results["FeatureMIA_SVC_acc_percent"] = feat_mia_svc["acc_percent"]
        evaluation_results["FeatureMIA_SVC_norm0_100"]   = feat_mia_svc["mia_0to100"]
        print(f"Feature-MIA (SVC) acc: {feat_mia_svc['acc_percent']:.2f}%, score: {feat_mia_svc['mia_0to100']:.2f}/100")
            


    if args.do_linear_probe:
        if args.use_last_only:
            print("[Linear Probe] last layer only")
            outs_LP = evaluation.linear_probe_last_layer(
                args, model, train_loader, test_loader,
                num_classes=args.num_classes,
                bs_probe=args.prob_batch_size,
                device=device
            )
            outs_LP = [outs_LP]   # ensure is list
            lp_df = pd.DataFrame(outs_LP)

            # ---- add lp_* keys ----
            if 'acc_test_retain' in lp_df.columns:
                evaluation_results['lp_acc_test_retain'] = float(lp_df['acc_test_retain'].iloc[0])
                
            else:
                evaluation_results['lp_acc_test_retain'] = None
            if 'acc_test_forget' in lp_df.columns:
                evaluation_results['lp_acc_test_forget'] = float(lp_df['acc_test_forget'].iloc[0])
            else:
                evaluation_results['lp_acc_test_forget'] = None


        else:
            print("[Linear Probe] full pipeline (all layers + penultimate)")
            outs_LP, lp_df = run_linear_probe_both(
                args, model, train_loader, test_loader,
                device=device,
                num_classes=args.num_classes,
                bs_probe=args.prob_batch_size
            )
            plot_LP(args, lp_df)

        # Save results
        evaluation_results["Linear_probe"] = outs_LP
        print(lp_df)


        '''''
        use_last_only = args.freeze_except_last or args.zero_last_layer or args.unlearn_method == "prune"
        use_last_only = True
        print(model)
        if "CMF" in args.unlearn_method and "RemoveFC" not in args.unlearn_method:
            outs_LP = evaluation.linear_probe_CMF(
                args, model, train_loader, test_loader,
                args.num_classes, args.prob_batch_size, device
            )
            outs_LP = [outs_LP]
        elif "CMF" in args.unlearn_method and "RemoveFC" in args.unlearn_method:
            outs_LP = evaluation.linear_probe_CMF_RemoveFC(
                args, model, train_loader, test_loader,
                args.num_classes, args.prob_batch_size, device
            )
            outs_LP = [outs_LP]
        elif use_last_only:
            outs_LP = evaluation.linear_probe_last_layer(
                args, model, train_loader, test_loader,
                args.num_classes, args.prob_batch_size, device
            )
            outs_LP = [outs_LP]
        else:
            outs_LP = evaluation.linear_probe_old(
                args, model, train_loader, test_loader,
                args.num_classes, args.prob_batch_size, device
            )

        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        

        outs_LP_df = pd.DataFrame(outs_LP)
        print(outs_LP_df)

        if not use_last_only:
            evaluation.plot_LP(args, outs_LP_df)

        evaluation_results["Linear_prob"] = outs_LP
        forget_classes = args.unlearn_class
        retain_classes=[c for c in range(args.num_classes) if c not in forget_classes]
        '''''


    # ================== NCC mismatch (NC-4 on TEST retain/forget) ==================
    if getattr(args, "do_ncc_mismatch", False):
        from evaluation.nc_cmf import ncc_mismatch

        print("[NCC] computing NCC accuracy/mismatch on test retain/forget ...")

        # 1) re-split retain/forget by unlearn_class from test_dataset (similar logic to MIA):contentReference[oaicite:8]{index=8}
        test_retain_dataset, test_forget_dataset = get_retain_forget_partition(
            args, test_dataset, args.unlearn_class
        )
        test_retain_loader = torch.utils.data.DataLoader(test_retain_dataset, **test_kwargs)
        test_forget_loader = torch.utils.data.DataLoader(test_forget_dataset, **test_kwargs)

        # 2) train_loader estimate class center, respectively on test_retain / test_forget for NCC calculation
        ncc_ret = ncc_mismatch(
            args=args,
            model=model,
            train_loader=train_loader,
            eval_loader=test_retain_loader,
            device=device,
            pool_mode="avg",
        )
        ncc_forget = ncc_mismatch(
            args=args,
            model=model,
            train_loader=train_loader,
            eval_loader=test_forget_loader,
            device=device,
            pool_mode="avg",
        )

        # 3) writeinto evaluation_results，convenientunified/statisticonekeepmaintainto JSON
        evaluation_results["NCC_mismatch_retain_test"] = ncc_ret
        evaluation_results["NCC_mismatch_forget_test"] = ncc_forget
        # ---- add scalar fields (the two printed numbers) ----
        evaluation_results["ncc_acc_test_retain"] = float(ncc_ret["ncc_acc"])
        evaluation_results["ncc_mismatch_test_retain"] = float(ncc_ret["ncc_mismatch"])
        evaluation_results["ncc_acc_test_forget"] = float(ncc_forget["ncc_acc"])
        evaluation_results["ncc_mismatch_test_forget"] = float(ncc_forget["ncc_mismatch"])


        print(f"[NCC] test_retain  acc = {ncc_ret['ncc_acc']:.4f}, "
              f"mismatch = {ncc_ret['ncc_mismatch']:.4f}")
        print(f"[NCC] test_forget  acc = {ncc_forget['ncc_acc']:.4f}, "
              f"mismatch = {ncc_forget['ncc_mismatch']:.4f}")



    # ================== NC (NC-3 + NC-4) evaluate ==================
    if getattr(args, "do_nc", False):
        from evaluation.nc_cmf import compute_nc_all, compute_nc_on_loader

        print("[NC] computing NC-3 & NCC metrics ...")
        nc_stats = compute_nc_all(
            args=args,
            model=model,
            train_loader=train_loader,
            retain_loader=retain_loader,
            forget_loader=forget_loader,
            device=device,
        )

        # save JSON results to convenient unified file for drawing figures / rebuttal
        evaluation_results["NC_metrics"] = nc_stats

        # print a few key numbers to console (convenient sanity check)
        print(f"[NC] NCC train_all    = {nc_stats['ncc_train_all']:.4f}")
        if "ncc_train_retain" in nc_stats:
            print(f"[NC] NCC train_retain = {nc_stats['ncc_train_retain']:.4f}")
        if "ncc_train_forget" in nc_stats:
            print(f"[NC] NCC train_forget = {nc_stats['ncc_train_forget']:.4f}")
        print(f"[NC] NC3 train_all    = {nc_stats['nc3_train_all']:.4f}")
        if "nc3_train_retain" in nc_stats:
            print(f"[NC] NC3 train_retain = {nc_stats['nc3_train_retain']:.4f}")
        if "nc3_train_forget" in nc_stats:
            print(f"[NC] NC3 train_forget = {nc_stats['nc3_train_forget']:.4f}")


        # ====== Now is test set of NC ======

        # Use test_dataset by unlearn_class again to split retain/forget
        test_retain_dataset, test_forget_dataset = get_retain_forget_partition(
            args, test_dataset, args.unlearn_class
        )
        test_retain_loader = torch.utils.data.DataLoader(
            test_retain_dataset, **test_kwargs
        )
        test_forget_loader = torch.utils.data.DataLoader(
            test_forget_dataset, **test_kwargs
        )

        # 1) test_all
        nc_test_all = compute_nc_on_loader(
            args=args,
            model=model,
            loader=test_loader,
            device=device,
            pool_mode="avg",
        )

        # 2) test_retain only
        nc_test_retain = compute_nc_on_loader(
            args=args,
            model=model,
            loader=test_retain_loader,
            device=device,
            pool_mode="avg",
        )

        # 3) test_forget only
        nc_test_forget = compute_nc_on_loader(
            args=args,
            model=model,
            loader=test_forget_loader,
            device=device,
            pool_mode="avg",
        )

        # writeinto evaluation_results in
        evaluation_results["NC_metrics_test_all"]     = nc_test_all
        evaluation_results["NC_metrics_test_retain"]  = nc_test_retain
        evaluation_results["NC_metrics_test_forget"]  = nc_test_forget

        print(f"[NC][test] NCC test_all    = {nc_test_all['ncc']:.4f}, "
              f"NC3 test_all = {nc_test_all['nc3']:.4f}")
        print(f"[NC][test] NCC test_retain = {nc_test_retain['ncc']:.4f}, "
              f"NC3 test_retain = {nc_test_retain['nc3']:.4f}")
        print(f"[NC][test] NCC test_forget = {nc_test_forget['ncc']:.4f}, "
              f"NC3 test_forget = {nc_test_forget['nc3']:.4f}")
        

        
    #metrics_penultimate = evaluation.nc_metrics(model,test_loader,device,
    #                    retain_classes, forget_classes)
        
    #evaluation_results["nc_metrics_penultimate"] = metrics_penultimate
    #evaluation_results["nc_metrics_final"] = metrics_final

    #print("nc_metrics_penultimate")
    #print(metrics_penultimate)

    #print("nc_metrics_final")
    #print(metrics_final)

    if args.save_model:
        suffix = ""
        if args.freeze_except_last:
            suffix += "_freeze"
        if getattr(args, "zero_last_layer", False):
            suffix += "_zero"
        method = args.unlearn_method + suffix
        result_path  = f"./evaluations/{method}/{args.dataset}_{args.arch}/{','.join([str(v) for v in args.unlearn_class])}.json"
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(evaluation_results, f)
    
if __name__=="__main__":
    main()

    