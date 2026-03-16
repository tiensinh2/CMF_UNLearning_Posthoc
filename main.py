import argparse
from xml.parsers.expat import model
import torch
import torch.optim as optim
import random
from utils import get_retain_forget_partition, get_dataset, get_model, test, SVC_MIA, to_jsonable, load_encoder_ckpt_safely
import evaluation
import os
import numpy as np
from unlearn import unlear_func
from train import train
import json
import copy
import pandas as pd
from unlearn.cmf_weights import CMFWeights, ModelModule

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def main():
    parser = argparse.ArgumentParser(description='PyTorch cifar10 Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
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
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--no-train-transform', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--arch', type=str, default='vgg11_bn',
                        help='')
    parser.add_argument('--data-path', type=str, default='../data/',
                        help='')
    parser.add_argument('--sub-set-mode', action='store_true', default=False,
                        help='use validation set')
    parser.add_argument('--sub-set-samples', type=int, default=10000, metavar='EPS',
                        help='')
    parser.add_argument('--val-ratio', type=float, default=0.1, metavar='R',
                        help='proportion of train set used for validation (default: 0.1)')
    parser.add_argument('--patience', type=int, default=25, metavar='P',
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
    parser.add_argument('--unlearn-class', type=str, default="",
                        help='')
    parser.add_argument('--unlearn-method', type=str, default='retrain',
                    choices=['pre_train', 'grad_ascent_descent', "grad_descent",
                             'salun', 'goel', 'ssd', 'scrub', 'scrub_CMF_RemoveFC', 'SVD', 'tarun', 'tarun_CMF_RemoveFC',
                             'random_label','random_label_once', 'retrain', 'prune', 
                             'random_label_layer3', "salun_CMF_RemoveFC","CMF_FT", "grad_ascent_descent_CMF", 
                             "CMF_FT_RemoveFC", "grad_ascent_descent_CMF_RemoveFC", "random_label_CMF_RemoveFC",  "random_label_once_CMF_RemoveFC", "random_label_iter_eval"],              # <- add this
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
    parser.add_argument('--do-mia',action='store_true', default=False,
                        help='')   
    parser.add_argument('--do-mia-ulira',action='store_true', default=False,
                        help='')   
    parser.add_argument('--plot-mia-roc',action='store_true', default=False,
                        help='') 
    parser.add_argument('--prob-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    
    # —— Unlearn parameters —— 
    parser.add_argument('--freeze_except_last', action='store_true', default=False,
                        help='if set, freeze all layers except the last one before unlearning')
    
    parser.add_argument('--zero_last_layer', action='store_true', default=False,
                        help='if set, zero out the last top‐level layer weights before training')
    
    # ——CMF parameters —— 
    parser.add_argument('--remove_FC', action='store_true', default=False,
                        help='if set, remove the fully connected layer from the model')
    
    parser.add_argument('--CMF_momentum', type=float, default=0.9,
                        help='')
    parser.add_argument('--CMFClassifier',  action='store_true', default=True,
                        help='')    
    
    parser.add_argument('--do_lp', action='store_true', default=False,
                        help='if set, do linear probe evaluation during unlearning')
    parser.add_argument('--lp_every', type=int, default=1,
                        help='how often (in epochs) to run linear probe evaluation; set to 0 to disable LP evaluation')
    parser.add_argument("--ncc_every", type=int, default=0,
                    help="run NCC evaluation every n epochs (0 to disable)")
    parser.add_argument(
        "--eval_every_iter",
        type=int,
        default=0,
        help="Run forward evaluation every N iterations (0 to disable)."
    )

    parser.add_argument(
        "--lp_every_iter",
        type=int,
        default=0,
        help="Run linear probe every N iterations (0 to disable)."
    )

    parser.add_argument(
        "--ncc_every_iter",
        type=int,
        default=0,
        help="Run NCC evaluation every N iterations (0 to disable)."
    )

    
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use ImageNet-1K pretrained weights for ViT backbones."
    )


    parser.add_argument('--lr-scheduler', type=str, default='cosine',
                    choices=['cosine', 'step', 'plateau', 'none'])
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--min-lr', type=float, default=1e-5)


    # args  = add_additional_args(parser)
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
        cuda_kwargs = {'num_workers': 16,
                       'pin_memory': True,
                       'shuffle': True,
                       'prefetch_factor': 2}
        #train_kwargs.update(cuda_kwargs)
        #test_kwargs.update(cuda_kwargs)
        train_kwargs.update({'num_workers': 2, 'pin_memory': True,  'shuffle': True})
        test_kwargs.update({'num_workers': 2, 'pin_memory': True, 'shuffle': False})


    # Load full dataset.
    # build cifar
    # build datasets (cifar + tinyimagenet + imagenette support pre_train 's/of train/val separate)
    dataset1, dataset2 = get_dataset(args)  # dataset1=train, dataset2=test/val

    if args.unlearn_method == "pre_train" or "FT" in args.unlearn_method:
        total_len = len(dataset1)
        val_len   = int(total_len * args.val_ratio)
        train_len = total_len - val_len

        g = torch.Generator().manual_seed(args.seed)
        all_indices = torch.randperm(total_len, generator=g).tolist()
        train_indices = all_indices[:train_len]
        val_indices   = all_indices[train_len:]

        # ---------- case 1: CIFAR（have .data / .targets） ----------
        if hasattr(dataset1, "data"):
            train_dataset = copy.deepcopy(dataset1)
            val_dataset   = copy.deepcopy(dataset1)

            train_dataset.data = dataset1.data[train_indices]
            val_dataset.data   = dataset1.data[val_indices]

            train_dataset.targets = [dataset1.targets[i] for i in train_indices]
            val_dataset.targets   = [dataset1.targets[i] for i in val_indices]

        # ---------- case 2: ImageFolder（tinyimagenet / imagenette / imagenet） ----------
        else:
            # ImageFolder inhave .samples（(path, label) columnexpress）and .targets
            train_dataset = copy.deepcopy(dataset1)
            val_dataset   = copy.deepcopy(dataset1)

            # subsetify samples
            train_dataset.samples = [dataset1.samples[i] for i in train_indices]
            val_dataset.samples   = [dataset1.samples[i] for i in val_indices]

            # havesomethisuse imgs thisname
            if hasattr(dataset1, "imgs"):
                train_dataset.imgs = train_dataset.samples
                val_dataset.imgs   = val_dataset.samples

            # surekeephave .targets，after get_retain_forget_partition use
            if hasattr(dataset1, "targets"):
                train_dataset.targets = [dataset1.targets[i] for i in train_indices]
                val_dataset.targets   = [dataset1.targets[i] for i in val_indices]
            else:
                train_dataset.targets = [s[1] for s in train_dataset.samples]
                val_dataset.targets   = [s[1] for s in val_dataset.samples]

        test_dataset = dataset2

    else:
        train_dataset = dataset1
        val_dataset   = None
        test_dataset  = dataset2

    print("== Dataset sanity check ==")
    print("Dataset:", args.dataset)
    print("Train len:", len(train_dataset))
    print("Val len:",   len(val_dataset) if val_dataset is not None else 0)
    print("Test len:",  len(test_dataset))

    print("learning rate:", args.lr)

    if args.dataset == "tinyimagenet":
        from torch.utils.data import DataLoader
        dl = DataLoader(train_dataset, batch_size=4, shuffle=True)
        imgs, labels = next(iter(dl))
        print("Batch shape:", imgs.shape)
        print("Train batch labels:", labels[:10])
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
    
    if args.save_model and args.save_path is not None and os.path.exists(args.save_path):
        raise FileExistsError(f"{args.save_path} already exists")
    
    if args.unlearn_method == "pre_train" :
        best_val_acc = 0.
        best_epoch = 0

        import math
        from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR

        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True
        )
        # ---- warmup + cosine ----
        if args.lr_scheduler == "cosine":
            warmup_epochs = max(0, int(args.warmup_epochs))
            total_epochs  = int(args.epochs_or_steps)

            def warmup_lambda(cur_epoch: int):
                # linearly increase lr from 0 -> 1 in warmup_epochs
                if warmup_epochs <= 0:
                    return 1.0
                return min(1.0, float(cur_epoch + 1) / float(warmup_epochs))

            warmup_sched = LambdaLR(optimizer, lr_lambda=warmup_lambda)

            # CosineAnnealingLR 's/of T_max is cosine 's/ofnumber（notcontain warmup）
            cosine_epochs = max(1, total_epochs - warmup_epochs)
            cosine_sched = CosineAnnealingLR(
                optimizer,
                T_max=cosine_epochs,
                eta_min=args.min_lr
            )

            # first warmup，again cosine
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_sched, cosine_sched],
                milestones=[warmup_epochs]
            )

        elif args.lr_scheduler == "step":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=args.gamma)
        elif args.lr_scheduler == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.gamma)
        else:
            scheduler = None

        history = {
            "epoch": [],
            "lr": [],

            
            "train_acc": [],
            "train_loss": [],

            "val_retain_acc": [],
            "val_forget_acc": [],
            "val_metric": [],

            "test_retain_acc": [],
            "test_forget_acc": [],
            "test_metric": [],
        }

        epochs_no_improve = 0

        ckpt_path = f"./checkpoints/{args.unlearn_method}/{args.dataset}_{args.arch}.pt"
        results_path = f"./checkpoints/{args.unlearn_method}/{args.dataset}_{args.arch}.json"
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        
        for epoch in range(1, args.epochs_or_steps + 1):
            train_loss, train_acc = train(args, model, device, train_loader, optimizer, epoch, "descent")
            
            print("Val set results:")
            val_retain_acc, val_forget_acc, val_metric = test(
                model, device, val_loader,
                args.unlearn_class,
                args.class_label_names,
                args.num_classes,
                plot_cm=False,
                job_name=args.unlearn_method,
                set_name="Val Set"
            )

            print("Test set results:")
            test_retain_acc, test_forget_acc, test_metric = test(
                model, device, test_loader,
                args.unlearn_class,
                args.class_label_names,
                args.num_classes,
                plot_cm=False,
                job_name=args.unlearn_method,
                set_name="Test Set"
            )


            cur_lr = float(optimizer.param_groups[0]["lr"])

            # ---- record curve ----
            history["epoch"].append(epoch)
            history["lr"].append(cur_lr)
            history["train_acc"].append(float(train_acc))
            history["train_loss"].append(float(train_loss))

            history["val_retain_acc"].append(float(val_retain_acc))
            history["val_forget_acc"].append(float(val_forget_acc))
            history["val_metric"].append(val_metric)   # metric cancanis dict/list，line/execute

            history["test_retain_acc"].append(float(test_retain_acc))
            history["test_forget_acc"].append(float(test_forget_acc))
            history["test_metric"].append(test_metric)


            if val_retain_acc > best_val_acc:
                best_val_acc = val_retain_acc
                epochs_no_improve = 0
                best_epoch = epoch
                print(f"Epoch {epoch}: New best validation accuracy = {val_retain_acc:.4f}.", 
                  "Saving model." if args.save_model else "Best model updated.")
                if args.save_model:
                    torch.save(model.state_dict(), ckpt_path)
                    print(f"[SAVE] epoch={epoch} best_val={best_val_acc:.4f} -> {ckpt_path}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    print(f"No improvement for {args.patience} epochs. Early stopping at epoch {epoch}.")
                    break

            if scheduler is not None:
                if args.lr_scheduler == "plateau":
                    scheduler.step(train_loss)
                else:
                    scheduler.step()

            print("lr now:", optimizer.param_groups[0]["lr"])

        print(f"Best epoch: {best_epoch}")

        if args.save_model and os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        
        unlearn_model = model
        retain_acc, forget_acc, metric  = test(model, device, test_loader, args.unlearn_class, args.class_label_names, args.num_classes,
                job_name=args.unlearn_method, set_name="Test Set")
        
        if args.save_model:
            os.makedirs(os.path.dirname(results_path), exist_ok=True)

            with open(results_path, "w") as f:
                json.dump(
                {"best_epoch": best_epoch, "best_val_acc": best_val_acc, "history": history},
                f,
                ensure_ascii=False,
                indent=2,
                default=to_jsonable
            )
                
        print(f"Best epoch: {best_epoch}, best val acc: {best_val_acc:.4f}")

    elif "CMF" in args.unlearn_method:
        print(args.unlearn_method)
        classes_for_mia = list(set(args.unlearn_class))
        _, test_forget_dataset = get_retain_forget_partition(args, test_dataset, classes_for_mia)
        test_forget_loader = torch.utils.data.DataLoader(test_forget_dataset)
        target_device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
        if args.unlearn_method == "CMF_FT" or args.unlearn_method == "CMF_FT_RemoveFC":
            ckpt_path = f"./checkpoints/pre_train/{args.dataset}_{args.arch}.pt"

            assert os.path.exists(ckpt_path), f"Checkpoint file not found: {ckpt_path}"

            #state = torch.load(ckpt_path, map_location=target_device)
            print(model)
            #print("Loading model state from checkpoint:"+ ckpt_path)
            #model.encoder.load_state_dict(state, strict=False)
            #ret = model.encoder.load_state_dict(state, strict=False)
            print("Loading model state from checkpoint:", ckpt_path)
            ret = load_encoder_ckpt_safely(model, ckpt_path, device=target_device)
            #print("[load_state_dict] missing:", len(ret.missing_keys))
            #print("[load_state_dict] unexpected:", len(ret.unexpected_keys))
            #print("unexpected head:", [k for k in ret.unexpected_keys[:20]])
            test(
            model, device, test_loader, args.unlearn_class, args.class_label_names, args.num_classes,
            job_name=args.unlearn_method, set_name="Final Test Set"
            )
        else:
            if "RemoveFC" in args.unlearn_method:
                ckpt_path = f"./checkpoints/CMF_FT_RemoveFC/{args.dataset}_{args.arch}.pt"
            else:
                ckpt_path = f"./checkpoints/CMF_FT/{args.dataset}_{args.arch}.pt"

            assert os.path.exists(ckpt_path), f"Checkpoint file not found: {ckpt_path}"
            print("Loading model state from checkpoint:"+ ckpt_path)
            state = torch.load(ckpt_path, map_location=target_device)
            model.load_state_dict(state)
            test(
            model, device, test_loader, args.unlearn_class, args.class_label_names, args.num_classes,
            job_name=args.unlearn_method, set_name="Final Test Set"
            )
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
        unlearn_model = unlear_func[args.unlearn_method](
            args=args,
            model=model,
            device=device,
            retain_loader=retain_loader,
            forget_loader=forget_loader,
            train_loader=sub_loader if args.sub_set_mode else train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            epochs=args.epochs_or_steps,
            train_dataset=dataset1,
            val_index=sub_index,
            test_forget_loader=test_forget_loader
        )
        torch.cuda.synchronize()
        if args.save_model:
            if args.unlearn_method == "CMF_FT" or args.unlearn_method == "CMF_FT_RemoveFC":
                ckpt_path = f"./checkpoints/{args.unlearn_method}/{args.dataset}_{args.arch}.pt"
            else:
                ckpt_path = f"./checkpoints/{args.unlearn_method}/{args.dataset}_{args.arch}/{','.join([str(v) for v in args.unlearn_class])}.pt"
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save(unlearn_model.state_dict(), ckpt_path)

            
            os.makedirs(f"./checkpoints/{args.unlearn_method}/{args.dataset}_{args.arch}", exist_ok=True)
            
            with open(f"./checkpoints/{args.unlearn_method}/{args.dataset}_{args.arch}/{','.join([str(v) for v in args.unlearn_class])}.json", "w") as f:
                json.dump(unlearn_model.history_log, f, ensure_ascii=False, indent=2, default=to_jsonable)
            print("Log saved!")
        
            
    else:
        print(sub_index)
        classes_for_mia = list(set(args.unlearn_class))
        _, test_forget_dataset = get_retain_forget_partition(args, test_dataset, classes_for_mia)
        test_forget_loader = torch.utils.data.DataLoader(test_forget_dataset)
        target_device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
        if args.unlearn_method != "retrain":
            ckpt_path = f"./checkpoints/pre_train/{args.dataset}_{args.arch}.pt"

            assert os.path.exists(ckpt_path), f"Checkpoint file not found: {ckpt_path}"

            state = torch.load(ckpt_path, map_location=target_device)
            model.load_state_dict(state)
            test(
            model, device, test_loader, args.unlearn_class, args.class_label_names, args.num_classes,
            job_name=args.unlearn_method, set_name="Final Test Set"
            )
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
        unlearn_model = unlear_func[args.unlearn_method](
            args=args,
            model=model,
            device=device,
            retain_loader=retain_loader,
            forget_loader=forget_loader,
            train_loader=sub_loader if args.sub_set_mode else train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            epochs=args.epochs_or_steps,
            train_dataset=dataset1,
            val_index=sub_index,
            test_forget_loader=test_forget_loader
        )
        torch.cuda.synchronize()
        if args.save_model:
            suffix = ""
            if args.freeze_except_last:
                suffix += "_freeze"
            if getattr(args, "zero_last_layer", False):
                suffix += "_zero"
            #method = args.unlearn_method + suffix
            ckpt_path = f"./checkpoints/{args.unlearn_method}/{args.dataset}_{args.arch}/{','.join([str(v) for v in args.unlearn_class])}.pt"
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save(unlearn_model.state_dict(), ckpt_path)

            with open(f"./checkpoints/{args.unlearn_method}/{args.dataset}_{args.arch}/{','.join([str(v) for v in args.unlearn_class])}.json", "w") as f:
                json.dump(unlearn_model.history_log, f, ensure_ascii=False, indent=2, default=to_jsonable)
            print("Log saved!")

        
    
        print("-" * 40)
        print(f"Model Unlearnt with {args.unlearn_method}")
        print("-" * 40)

        unlearn_model.eval()
        train_retain_acc, train_forget_acc, train_metric = test(
            unlearn_model, device, train_loader, args.unlearn_class, args.class_label_names, args.num_classes,
            job_name=args.unlearn_method, set_name="Final Train Set"
        )
        test_retain_acc, test_forget_acc, test_metric = test(
            unlearn_model, device, test_loader, args.unlearn_class, args.class_label_names, args.num_classes,
            job_name=args.unlearn_method, set_name="Final Test Set"
        )
        evaluation_results = {}    
        evaluation_results["train_retain_acc"] = train_retain_acc
        evaluation_results["train_forget_acc"] = train_forget_acc
        evaluation_results["train_metric"] = train_metric
        evaluation_results["test_retain_acc"] = test_retain_acc
        evaluation_results["test_forget_acc"] = test_forget_acc
        evaluation_results["test_metric"] = train_metric

        
        
        

        '''''''''
        classes_for_mia = args.unlearn_class
        classes_for_mia = list(set(classes_for_mia))         
        model.do_log_softmax  = True
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
        use_last_only = args.freeze_except_last or args.zero_last_layer or args.unlearn_method == "prune"

        if use_last_only:
            outs_LP = evaluation.linear_probe_last_layer(
                args, model, train_loader, test_loader,
                args.num_classes, args.prob_batch_size, device
            )
            outs_LP = [outs_LP]
        else:
            outs_LP = evaluation.linear_probe(
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

        
        metrics_penultimate = evaluation.nc_metrics(model,test_loader,device,
                                            retain_classes, forget_classes)
        
        evaluation_results["nc_metrics_penultimate"] = metrics_penultimate
        #evaluation_results["nc_metrics_final"] = metrics_final

        print("nc_metrics_penultimate")
        print(metrics_penultimate)

        #print("nc_metrics_final")
        #print(metrics_final)

        if args.save_model:
            suffix = ""
            if args.freeze_except_last:
                suffix += "_freeze"
            if getattr(args, "zero_last_layer", False):
                suffix += "_zero"
            method = args.unlearn_method + suffix
            result_path  = f"./checkpoints/{method}/{args.dataset}_{args.arch}/{','.join([str(v) for v in args.unlearn_class])}.json"
        with open(result_path, "w") as f:
            json.dump(evaluation_results, f)
            '''''''''
    
if __name__=="__main__":
    main()

    