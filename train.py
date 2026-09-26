
import torch
import torch.nn.functional as F

def train(args, model, device, train_loader, optimizer, epoch, mode = "descent", clip=None):
    model.train()
    loss_sum = 0.0

    correct = 0
    seen = 0
    # CMF momentum for online W update (default 0.9 per paper §A.4)
    _cmf_mom = getattr(getattr(model, "args", args), "CMF_momentum", 0.9)
    _use_cmf = hasattr(model, "CMFweights")

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        if hasattr(model, "do_log_softmax") and getattr(model, "do_log_softmax"):
            loss = F.nll_loss(output, target)
        else:
            loss = F.cross_entropy(output, target)
        loss.backward()
        if mode == "ascent":
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data *= -1.0
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        # ── Online CMF weight update (EMA of per-class feature means) ──────
        # W is a register_buffer so SGD never touches it.  Update it here
        # using the current batch features so gradient in the NEXT forward
        # pass is computed against an up-to-date W.
        # Uses CMFWeights.update_class_means() which does:
        #   W_c ← momentum * W_c + (1-momentum) * batch_mean_c  (no normalize)
        # We then normalize W so each row is a unit vector.
        if _use_cmf:
            with torch.no_grad():
                feats = model.extract_features(data)
                feats_n = torch.nn.functional.normalize(feats, dim=1)
                model.CMFweights.update_class_means(feats_n, target, _cmf_mom)
                model.CMFweights.weight.copy_(
                    torch.nn.functional.normalize(model.CMFweights.weight, dim=1)
                )

        bs = target.size(0)
        loss_sum += loss.detach().item() * bs
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        seen += bs

        if batch_idx % args.log_interval == 0:
            cur_loss = loss_sum / max(1, seen)
            cur_acc  = 100.0 * correct / max(1, seen)
            print(
                f"Train Epoch: {epoch} [{seen}/{len(train_loader.dataset)} "
                f"({100.0 * seen / len(train_loader.dataset):.0f}%)]\t"
                f"Loss: {cur_loss:.6f}\tAcc: {cur_acc:.2f}%"
            )
            if args.dry_run:
                break

    epoch_loss = loss_sum / max(1, seen)
    epoch_acc  = correct / max(1, seen)  # 0~1
    return epoch_loss, epoch_acc