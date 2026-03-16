import torch
import torch.nn.functional as F
from torch import nn
import copy

def calc_importance(model, device, optimizer, dataloader):
    importances = {k: torch.zeros_like(p, device=p.device) for k, p in model.named_parameters()}
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = F.nll_loss(out, y)
        loss.backward()
        for (k1, p), (k2, imp) in zip(model.named_parameters(), importances.items()):
            if p.grad is not None:
                imp.data += p.grad.data.clone().pow(2)
    for imp in importances.values():
        imp.data /= float(len(dataloader))
    return importances


def ssd_unlearn(args, model, device, retain_loader, forget_loader, train_loader, test_loader, optimizer, **kwargs):
    from unlearn.tools import maybe_eval_and_save
    forget_dataset = forget_loader.dataset
    forget_dataset, _ = torch.utils.data.random_split(
        forget_dataset, [args.num_forget_samples, len(forget_dataset) - args.num_forget_samples]
    )
    retain_dataset = retain_loader.dataset
    retain_dataset, _ = torch.utils.data.random_split(
        retain_dataset, [args.num_retain_samples, len(retain_dataset) - args.num_retain_samples]
    )
    ssd_retain_loader = torch.utils.data.DataLoader(retain_dataset, batch_size=args.batch_size, shuffle=True)
    train_dataset = torch.utils.data.ConcatDataset([forget_loader.dataset, retain_dataset])
    ssd_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    lower_bound = 1
    exponent = 1
    dampening_constant = args.ssd_lambda
    selection_weighting = args.ssd_alpha
    model.eval()
    forget_importance = calc_importance(model, device, optimizer, forget_loader)
    original_importance = calc_importance(model, device, optimizer, ssd_train_loader)
    with torch.no_grad():
        for (n, p), (__, oimp), (__, fimp) in zip(
            model.named_parameters(), original_importance.items(), forget_importance.items()
        ):
            oimp_norm = oimp.mul(selection_weighting)
            locations = torch.where(fimp > oimp_norm)
            weight = ((oimp.mul(dampening_constant)).div(fimp)).pow(exponent)
            update = weight[locations]
            min_locs = torch.where(update > lower_bound)
            update[min_locs] = lower_bound
            p[locations] = p[locations].mul(update)
    return model