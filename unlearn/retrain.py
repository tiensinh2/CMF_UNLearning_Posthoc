import copy
import torch
import torch.nn.functional as F
import torch.optim as optim

def re_train(
    args,
    model,
    device,
    retain_loader,
    test_loader,
    epochs,
    test_forget_loader=None,   # keep，timenotuse
    **kwargs
):
    """
     retrain-on-retain baseline:
    - onlyin retain dataset training（notagainuseuse forget data）
    - useuse early stopping（in retain_acc - forget_acc）
    """

    from utils import test    # youhave's/of test number
    num_classes = args.num_classes

    # ========= 1. construct retrain use's/of retain subset =========
    full_retain = copy.deepcopy(retain_loader.dataset)
    assert args.num_retain_samples <= len(full_retain), \
        "num_retain_samples larger than retain_dataset size"

    gen_r = torch.Generator().manual_seed(args.seed + 1)
    retain_sub, _ = torch.utils.data.random_split(
        full_retain,
        [args.num_retain_samples, len(full_retain) - args.num_retain_samples],
        generator=gen_r
    )

    train_loader_retain = torch.utils.data.DataLoader(
        retain_sub,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=getattr(args, "num_workers", 4),
        pin_memory=True
    )

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

    # ========= 3. training + early stopping =========
    forget_classes = set(args.unlearn_class)

    # firstoneinitialstate
    test(
        model, device, test_loader,
        args.unlearn_class, args.class_label_names, num_classes,
        job_name=args.unlearn_method, set_name="Test Set (before retrain)"
    )

    best_score = -float("inf")
    best_model = copy.deepcopy(model.state_dict())
    best_epoch = 0
    patience = getattr(args, "patience", 10)
    patience_counter = 0

    retain_acc_list, forget_acc_list = [], []
    LP_retain_acc_list, LP_forget_acc_list = [], []
    LP_history_list = []
    epoch_list = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        loss_sum = 0.0
        correct = 0
        seen = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader_retain):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            bs = labels.size(0)

            running_loss += loss.detach().item()
            loss_sum += loss.detach().item() * bs
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            seen += bs

            if batch_idx % args.log_interval == 0:
                cur_loss = loss_sum / max(1, seen)
                cur_acc  = 100.0 * correct / max(1, seen)
                print(
                    f"Train Epoch: {epoch} [{seen}/{len(train_loader_retain.dataset)} "
                    f"({100.0 * seen / len(train_loader_retain.dataset):.0f}%)]\t"
                    f"Loss: {cur_loss:.6f}\tAcc: {cur_acc:.2f}%"
                )
                if args.dry_run:
                    break

            
        epoch_loss = loss_sum / max(1, seen)
        epoch_acc  = correct / max(1, seen)  # 0~1

        print(f"[Epoch {epoch}] train loss: {epoch_loss:.4f}")

        if getattr(args, "dry_run", False):
                break

        # ---- eval on test set: retain / forget accuracy ----
        model.eval()
        retain_acc, forget_acc, metric = test(
            model, device, test_loader,
            args.unlearn_class, args.class_label_names, num_classes,
            job_name=args.unlearn_method, set_name="Test Set (retrain)"
        )

        retain_acc_list.append(retain_acc)
        forget_acc_list.append(forget_acc)
        epoch_list.append(epoch)

        #  retain_acc / forget_acc isseparate
        score = epoch_acc   # retain good，forget good

        #if score > best_score:
        #    best_score = score
        #    best_model = copy.deepcopy(model.state_dict())
        #    best_epoch = epoch
        #    patience_counter = 0
        #    print(f"New best score: {score:.4f} at epoch {epoch}")
        #else:
        #    patience_counter += 1
        #    print(f"No improvement. Patience {patience_counter}/{patience}")
        #    if patience_counter >= patience:
        #        print(f"Early stopping at epoch {epoch}")
        #        break

        # scheduler usetraining loss
        if scheduler is not None:
            if args.lr_scheduler == "plateau":
                scheduler.step(epoch_loss)
            else:
                scheduler.step()

        print("lr now:", optimizer.param_groups[0]["lr"])

    #print(f"Best epoch: {best_epoch}, best score: {best_score:.4f}")
    #model.load_state_dict(best_model)


    model.history_log = {
        "retain_acc": retain_acc_list,
        "forget_acc": forget_acc_list,
        #"LP_history": LP_history_list,
        #"Wn": Wn_list,
        #"Hn": Hn_list,
        #"G_WW": G_WW_list,
        #"G_HH": G_HH_list,
        #"G_WH": G_WH_list,
        "epoch": epoch_list,
    }
    return model
