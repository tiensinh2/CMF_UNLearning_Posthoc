import copy, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# -----------------------------------------------------------
# new：number，to/for layer3 do & MSE to/foralign
# -----------------------------------------------------------
def distill_layer3(model, device, retain_loader, n_epoch=150, lr=3e-4):
    model.eval()

    # 1. Hook get layer3_in / out
    feat_in, feat_out = [], []

    def hfunc(m, inp, out):
        feat_in.append(inp[0].cpu())
        feat_out.append(out.cpu())

    h = model.layer3.register_forward_hook(hfunc)
    with torch.no_grad():
        for xb, _ in tqdm(retain_loader, desc="Hook layer3"):
            _ = model(xb.to(device))
    h.remove()

    x_in  = torch.cat(feat_in).half()   # [N,128,4,4] or [N,128,4,4]
    y_out = torch.cat(feat_out).half()  # [N,256,2,2]

    dl = DataLoader(TensorDataset(x_in, y_out),
                    batch_size=512, shuffle=True,
                    num_workers=4, pin_memory=True)

    # 2.  layer3 and
    new_l3 = copy.deepcopy(model.layer3).to(device)
    for p in new_l3.parameters():
        nn.init.zeros_(p)

    opt = torch.optim.Adam(new_l3.parameters(), lr=lr, weight_decay=1e-4)
    mse = nn.MSELoss()

    # 3. training
    for ep in range(1, n_epoch+1):
        if ep % 20 == 0:
            with torch.no_grad():
                w = new_l3[0].conv1.weight        #  1  BasicBlock 's/of conv1
                print("mean|w| =", w.abs().mean().item())
        loss_sum = 0.
        for xb, yb in dl:
            xb = xb.to(device).float()
            yb = yb.to(device).float()

            pred = new_l3(xb)
            loss = mse(pred, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * xb.size(0)
        print(f"[Distill E{ep:03d}]  MSE={loss_sum/len(x_in):.6f}")

    # 4. weightand BN
    model.layer3.load_state_dict(new_l3.state_dict())
    model.train(False)
    with torch.no_grad():
        for xb, _ in retain_loader:
            _ = model(xb.to(device))
    return model


# -----------------------------------------------------------
# after's/of random_label_unlearn (onlyadd 3 line/execute)
# -----------------------------------------------------------
def random_label_unlearn_layer3(args, model, device,
                         retain_loader, forget_loader,
                         train_loader, test_loader,
                         optimizer, epochs, test_forget_loader, **kwargs):
    from utils import test
    from unlearn.tools import freeze_except_last_layer, zero_except_last_layer
    import torch.optim as optim
    import copy, torch

    # ---------- original Stage A：randomlabel ----------
    if args.freeze_except_last:
        freeze_except_last_layer(model, args.freeze_except_last)
    elif args.zero_last_layer:
        zero_except_last_layer(model, args.zero_last_layer)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay, nesterov=True)

    ...  # thisinkeepsupportyouoriginalcome's/oftrainingnot

    # ============ new Stage B： layer3 =============
    if getattr(args, "distill_layer3", True):
        print("\n=== Stage‑B: Distill layer3 (MSE) ===")
        model = distill_layer3(model, device, retain_loader,
                               n_epoch=150, lr=3e-4)

    return model
