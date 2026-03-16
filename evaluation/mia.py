# evaluation/mia.py
import numpy as np
from sklearn import linear_model, model_selection
import torch
import torch.nn.functional as F
from sklearn.svm import SVC


import torch
from .collect_feature import collect_features_for_head  # use your existing function


@torch.no_grad()
def compute_losses(model, loader, device, args=None):
    """
    Compute per-sample cross-entropy loss.
    - For CMF models (args.unlearn_method contains "CMF"), the model forward returns logits -> use CrossEntropyLoss.
    - For other models (ResNet_* with do_log_softmax=True), the model forward returns log-prob -> use NLLLoss.
    """
    if args is not None and ("CMF" in getattr(args, "unlearn_method", "")):
        outputs = "logits"
    elif hasattr(model, "do_log_softmax") and bool(getattr(model, "do_log_softmax")):
        outputs = "logprob"
    else:
        outputs = "logits"

    model.eval()
    losses = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        if outputs == "logprob":
            loss = F.nll_loss(out, yb, reduction="none")
        else:
            loss = F.cross_entropy(out, yb, reduction="none")
        losses.append(loss.detach().cpu().numpy())
    return np.concatenate(losses, axis=0)


def simple_mia(sample_loss, members, n_splits=5, random_state=0):
    """
    ESC-style logistic regression attack on one-dimensional loss values.
    Returns cross-validation accuracy across n_splits.
    """
    members = np.asarray(members)
    if not np.array_equal(np.unique(members), np.array([0, 1])):
        raise ValueError("members should only contain 0 and 1")
    X = np.asarray(sample_loss)
    clf = linear_model.LogisticRegression()
    cv = model_selection.StratifiedShuffleSplit(
        n_splits=n_splits, random_state=random_state
    )
    return model_selection.cross_val_score(clf, X, members, cv=cv, scoring="accuracy")


def evaluate_mia(model, forget_train_loader, forget_test_loader, device, args=None,
                 return_both=False, rng_seed=0):
    """
    ESC-style MIA:
    - Members = forget-class training samples
    - Non-members = forget-class test samples
    Returns mean accuracy (%) or both accuracy and normalized [0,100] score.
    """
    print("*" * 100)
    print(" " * 20 + "Membership Inference Attack (ESC-style)")
    print("*" * 100)

    f_losses = compute_losses(model, forget_train_loader, device, args=args)
    t_losses = compute_losses(model, forget_test_loader,  device, args=args)

    # Align sample counts by random subsampling
    n = min(len(f_losses), len(t_losses))
    rng = np.random.default_rng(rng_seed)
    f_losses = rng.choice(f_losses, size=n, replace=False)
    t_losses = rng.choice(t_losses, size=n, replace=False)

    X = np.concatenate([t_losses, f_losses]).reshape(-1, 1)
    y = np.array([0] * n + [1] * n)

    scores = simple_mia(X, y) * 100.0
    acc = float(scores.mean())

    # ESC paper reports normalized values: random=0, perfect=100
    mia_norm = max(0.0, (acc - 50.0) * 2.0)

    print(f"[MIA] accuracy={acc:.2f}% | normalized={mia_norm:.2f}")
    return {'acc_percent': acc, 'mia_0to100': mia_norm} if return_both else acc








def evaluate_feature_mia(model, forget_train_loader, forget_test_loader, device,
                         args=None, return_both=False, rng_seed=0, pca_dim=None,
                         clf: str = "lr"):
    """
    Feature-level MIA:
    - Members   = forget-class training embeddings
    - NonMembers= forget-class test embeddings
    clf: "lr" for LogisticRegression, "svc" for Support Vector Classifier
    """
    print("*" * 100)
    print(" " * 20 + f"Membership Inference Attack (Feature-level, clf={clf})")
    print("*" * 100)

    # Collect embeddings
    F_mem, _, _, d_used = collect_features_for_head(model, forget_train_loader, device)
    F_non, _, _, _      = collect_features_for_head(model, forget_test_loader,  device)

    # Balance sample counts
    n = min(F_mem.size(0), F_non.size(0))
    rng = np.random.default_rng(rng_seed)
    idx_mem = rng.choice(F_mem.size(0), size=n, replace=False)
    idx_non = rng.choice(F_non.size(0), size=n, replace=False)

    X = torch.cat([F_non[idx_non], F_mem[idx_mem]], dim=0).numpy()
    y = np.array([0] * n + [1] * n)

    # Optional PCA
    if pca_dim is not None and pca_dim > 0 and pca_dim < X.shape[1]:
        from sklearn.decomposition import PCA
        X = PCA(n_components=pca_dim, random_state=rng_seed).fit_transform(X)

    # Choose classifier
    if clf == "lr":
        attacker = linear_model.LogisticRegression(max_iter=2000)
    elif clf == "svc":
        attacker = SVC(C=3.0, kernel="rbf", gamma="scale")
    else:
        raise ValueError(f"Unknown clf: {clf}, must be 'lr' or 'svc'.")

    # Cross-validation accuracy
    cv = model_selection.StratifiedShuffleSplit(
        n_splits=5, random_state=rng_seed
    )
    scores = model_selection.cross_val_score(attacker, X, y, cv=cv, scoring="accuracy")
    acc = float(scores.mean() * 100.0)

    # Normalized score: random=0, perfect=100
    mia_norm = max(0.0, (acc - 50.0) * 2.0)

    print(f"[Feature-MIA] accuracy={acc:.2f}% | normalized={mia_norm:.2f}")
    return {'acc_percent': acc, 'mia_0to100': mia_norm} if return_both else acc
