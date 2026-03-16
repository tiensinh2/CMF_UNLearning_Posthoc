import os
import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict
from functools import wraps

def maybe_eval_and_save(step, model, args,
                        train_loader, test_loader,
                        retain_loader, forget_loader, test_forget_loader):
    """
    Every K steps:
      • run full evaluation via test() imports
      • compute SVC_MIA
      • save checkpoint
    """
    from utils import test, SVC_MIA  # import here to avoid circular
    print("\n" + "-" * 40)
    print(f"Epoch: {step}")
    print("-" * 40)
    device = torch.device(f"cuda:{args.gpu_id}") if torch.cuda.is_available() else torch.device("cpu")
    # TRAIN set
    test(model, device, train_loader,
         args.unlearn_class, args.class_label_names, args.num_classes,
         job_name=f"{args.unlearn_method}@{step}", set_name="Train Set")

    # TEST set
    test(model, device, test_loader,
         args.unlearn_class, args.class_label_names, args.num_classes,
         job_name=f"{args.unlearn_method}@{step}", set_name="Test Set")
    
    # MIA on forget efficacy
    classes_for_mia = list(set(args.unlearn_class))
    mia_forget = SVC_MIA(
        shadow_train = retain_loader,
        shadow_test  = forget_loader,
        target_train = None,
        target_test  = test_forget_loader,
        model        = model,
    )["confidence"]
    print(f"[MIA] forget efficacy : {mia_forget:.4f}")

    # Save checkpoint
    if args.save_model:
        ckpt_path = f"./checkpoints/{args.unlearn_method}/{args.unlearn_method}_{step}/{args.dataset}_{args.arch}_{args.unlearn_method}_{','.join([str(v) for v in args.unlearn_class])}.pt"
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)


def freeze_except_last_layer(model, freeze: bool):
    """
    If freeze=True:
      1. Freeze all parameters in the model (requires_grad=False).
      2. Identify the last top-level child module and unfreeze its parameters only.
    If freeze=False, do nothing.
    """
    if not freeze:
        return

    # 1. Freeze every parameter in the entire model
    for param in model.parameters():
        param.requires_grad = False

    # 2. Get the last top-level named child module
    children = list(model.named_children())
    last_name, last_module = children[-1]
    #    └─ children[-1] is a (name, module) tuple
    #             └─ [1] picks the module; here we unpack directly instead

    # 3. Unfreeze only that last module
    for param in last_module.parameters():
        param.requires_grad = True

    print(f"[Freeze] All layers except '{last_name}' are now frozen.")





import torch
import torch.nn as nn

def zero_except_last_layer(model, freeze: bool):
    """
    Freeze all parameters, then unfreeze the last top-level child module
    that actually has parameters. Also zero its weight/bias if they exist.
    """
    if not freeze:
        return

    # 1. Freeze all params
    for p in model.parameters():
        p.requires_grad = False

    # 2. Find the last top-level child that has parameters
    children = list(model.named_children())
    if not children:
        raise ValueError("Model has no child modules.")

    last_name, last_module = None, None
    for name, module in reversed(children):
        has_params = any(True for _ in module.parameters())
        if has_params:
            last_name, last_module = name, module
            break

    if last_module is None:
        raise ValueError("No child module with parameters found.")

    # 3. Zero weight / bias if present
    with torch.no_grad():
        if hasattr(last_module, "weight") and last_module.weight is not None:
            last_module.weight.zero_()
            print(f"[Zero] Layer '{last_name}' weights have been zeroed.")
        if hasattr(last_module, "bias") and last_module.bias is not None:
            last_module.bias.zero_()
            print(f"[Zero] Layer '{last_name}' bias has been zeroed.")

    # 4. Unfreeze only this module
    for p in last_module.parameters():
        p.requires_grad = True

    # 5. Freeze batchnorm behavior
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            module.track_running_stats = False
            if module.affine:
                module.weight.requires_grad = False
                module.bias.requires_grad = False

    # 6. Safety check
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("No trainable parameters left after freezing.")

    print(f"[ZeroFreeze] froze all except '{last_name}'")
    print("[Trainable params]")
    for n in trainable:
        print(" ", n)

def apply_prep(func):
    @wraps(func)
    def wrapper(args, model, *rest, **kw):
        # 1) ifyafteronelayer（ifopenthis flag）
        if getattr(args, 'zero_last_layer', False):
            zero_except_last_layer(model, True)
        # 2) afteronelayer（ifopenthis flag）
        if getattr(args, 'freeze_except_last', False):
            freeze_except_last_layer(model, True)



        #print("===== model =====")
        #print(model)

        #print("\n===== named_children =====")
        #for i, (name, module) in enumerate(model.named_children()):
        #    print(i, name, "->", module.__class__.__name__)

        #print("\n===== named_parameters =====")
        #for name, p in model.named_parameters():
        #    print(name, p.shape, "requires_grad=", p.requires_grad)


        return func(args, model, *rest, **kw)
    return wrapper