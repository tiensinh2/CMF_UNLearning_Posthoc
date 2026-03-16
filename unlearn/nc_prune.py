import torch
import torch.nn as nn
from functools import wraps

def prune(args, model, device,
                         retain_loader, forget_loader,
                         train_loader, test_loader,
                         optimizer, epochs, test_forget_loader, **kwargs):
    """
    Neural-Collapse Pruning Unlearning Method

    This method implements a one-step “prune”:
      - It finds the model’s last nn.Linear module.
      - For every class in args.unlearn_class, it zeros out that row in weight
        and (if present) that element in bias.
    No further training is performed.
    """

    from utils import test

    test(model, device, test_loader,
         args.unlearn_class, args.class_label_names, args.num_classes,
         job_name=args.unlearn_method, set_name="Test Set")
    # 1) Gather all top-level children
    children = list(model.named_children())
    if not children:
        raise RuntimeError("Model has no submodules to prune.")

    # 2) Try the very last child first
    name, module = children[-1]

    # 3) If that's not a Linear, search backwards for the last nn.Linear
    if not isinstance(module, nn.Linear):
        for n, m in reversed(list(model.named_modules())):
            if isinstance(m, nn.Linear):
                name, module = n, m
                break
        else:
            raise RuntimeError("Could not find any nn.Linear layer to prune.")

    # 4) Zero out the rows corresponding to the forget classes
    forget_classes = args.unlearn_class
    with torch.no_grad():
        for cls in forget_classes:
            # weight: [num_classes, in_features]
            module.weight.data[cls].zero_()
            # bias:    [num_classes]
            if module.bias is not None:
                module.bias.data[cls].zero_()

    print(f"[nc_prune] Zeroed out final Linear '{name}' rows for classes {forget_classes}")
    test(model, device, test_loader,
         args.unlearn_class, args.class_label_names, args.num_classes,
         job_name=args.unlearn_method, set_name="Test Set")
    test(model, device, train_loader,
         args.unlearn_class, args.class_label_names, args.num_classes,
         job_name=args.unlearn_method, set_name="Test Set")
    return model
