import torch
import torch.nn.functional as F
from torch import nn

def _reinit(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def get_retrain_layers(module, name, collected):
    if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
        collected.append((module, name))
    for child_name, child in module.named_children():
        get_retrain_layers(child, f"{name}.{child_name}", collected)
    return collected


def reset_final_resnet(model, num_retrain, reinit=True):
    for param in model.parameters():
        param.requires_grad = False
    collected = get_retrain_layers(model, 'M', [])
    collected.reverse()
    done = 0
    for module, name in collected:
        if reinit and isinstance(module, (nn.Conv2d, nn.Linear)):
            _reinit(module)
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            done += 1
        for param in module.parameters():
            param.requires_grad = True
        if done >= num_retrain:
            break
    return model


def goel_last_unlearn(args, model, device, retain_loader, forget_loader, train_loader, test_loader, optimizer, epochs, **kwargs):
    from unlearn.tools import maybe_eval_and_save
    retain_dataset = retain_loader.dataset
    retain_dataset, _ = torch.utils.data.random_split(
        retain_dataset, [args.num_retain_samples, len(retain_dataset) - args.num_retain_samples]
    )
    goel_train_loader = torch.utils.data.DataLoader(retain_dataset, batch_size=args.batch_size, shuffle=True)
    model.train()
    model = reset_final_resnet(model, 1, reinit=args.goel_exact)
    for epoch in range(epochs):
        for data, target in goel_train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.dry_run:
                break
        if args.dry_run:
            break
    return model