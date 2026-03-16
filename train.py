
import torch
import torch.nn.functional as F

def train(args, model, device, train_loader, optimizer, epoch, mode = "descent", clip=None):
    model.train()
    loss_sum = 0.0

    correct = 0
    seen = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        if mode == "ascent":
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data *= -1.0
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

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