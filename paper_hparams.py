"""
paper_hparams.py
================
Single authoritative source of all hyperparameters used in the pipeline.
Values are sourced from Table 4 (Appendix D.1) of the paper; §A.4 used only
where Table 4 has no entry.

Cross-reference key (Table 4, page 18 of the PDF):
  §A.4  = Appendix A.4, Original Model Training
  T4    = Table 4, Hyperparameters for ResNet-based experiments
  T5    = Table 5, Hyperparameters for ViT-based experiments

Usage:
    from paper_hparams import (
        GLOBAL_CFG, PRETRAIN_CFG, METHOD_CFG, CMF_METHOD_CFG,
        RETRAIN_CFG, CMF_CFG, STAGE3_LR, POSTHOC_CFG
    )

Do NOT read hyperparameters from any shell script (script/run/*.sh).
"""

# ---------------------------------------------------------------------------
# Global protocol config
# ---------------------------------------------------------------------------
GLOBAL_CFG = dict(
    dataset         = "cifar10",
    arch            = "resnet18",
    data_path       = "./data",
    num_classes     = 10,
    batch_size      = 128,
    test_batch_size = 256,
    num_workers     = 4,
    # whole-class-single: all forget classes 0–9
    forget_classes    = list(range(10)),
    SEEDS             = [0],    # extend to [0, 1, 2] for multi-seed runs
    TEST_MODE         = False,  # True → shortened epochs, "_test" suffix in filenames
    TEST_EPOCHS_SCALE = 0.1,    # fraction of epochs when TEST_MODE=True
)

# ---------------------------------------------------------------------------
# Pre-training (NB1) — §A.4 + Table 4
#
# §A.4 states:
#   "For ResNet training, we use batch size 128 and train for up to 300 epochs
#    with early stopping (patience 50).  SGD, momentum 0.9, weight decay 5e-4.
#    Initial LR = 5×10^-2 [for TinyImageNet]; warmup 5 epochs; cosine decay;
#    min LR = 1×10^-5."
#
# Table 4 lists:
#   CIFAR-10 / CIFAR-100 Original: LR = 0.01, epochs = 300, batch = 128
#   Tiny-ImageNet Original:        LR = 0.05, epochs = 300, batch = 128
#
# We use the CIFAR-10/100 defaults here; override lr for TinyImageNet via
# hparam_overrides in the notebook config.
# ---------------------------------------------------------------------------
PRETRAIN_CFG = dict(
    epochs              = 300,      # T4 (all datasets)
    lr                  = 0.01,     # T4 — CIFAR-10 and CIFAR-100
                                    # Override to 0.05 for Tiny-ImageNet
    momentum            = 0.9,      # §A.4
    weight_decay        = 5e-4,     # T4 / §A.4
    nesterov            = True,
    scheduler           = "cosine", # §A.4 — cosine LR decay
    warmup_epochs       = 5,        # §A.4
    min_lr              = 1e-5,     # §A.4
    early_stop_patience = 50,       # §A.4
    CMF_momentum        = 0.9,      # §A.4
    temperature         = 1.0,
    CMFClassifier       = True,
    remove_FC           = True,
)

# ---------------------------------------------------------------------------
# Retain-only Retrain (oracle / gold-standard baseline)
# Table 4:
#   CIFAR-10/100 ResNet-18: epochs=200, batch=128, LR=0.01
#   Tiny-ImageNet ResNet-50: epochs=150, batch=256, LR=0.05
# Same optimiser / scheduler settings as pre-training.
# ---------------------------------------------------------------------------
RETRAIN_CFG = dict(
    epochs              = 200,      # T4 — CIFAR-10 / CIFAR-100
                                    # Override to 150, lr=0.05, batch=256 for Tiny-ImageNet
    lr                  = 0.01,     # T4 — CIFAR-10 / CIFAR-100
    momentum            = 0.9,
    weight_decay        = 5e-4,
    nesterov            = True,
    scheduler           = "cosine",
    warmup_epochs       = 5,
    min_lr              = 1e-5,
    early_stop_patience = 50,
    val_ratio           = 0.1,      # T4 note for CIFAR-10
)

# ---------------------------------------------------------------------------
# Per-method hyperparameters — Table 4 (ResNet rows)
#
# Notes on multi-dataset lr values (Table 4):
#   Each method has different LRs per dataset.  The dict holds the CIFAR-10
#   default (primary benchmark); use the per_dataset sub-dict to look up
#   dataset-specific overrides when running CIFAR-100 or Tiny-ImageNet.
# ---------------------------------------------------------------------------
METHOD_CFG = dict(

    # ── NegGrad+ (gradient ascent on forget + descent on retain) ────────────
    # T4: CIFAR-10 LR=1e-4, CIFAR-100 LR=5e-3, TinyIN LR=5e-4
    #     epochs=3, batch=128, grad-clip=1.0 (all datasets)
    grad_ascent_descent = dict(
        lr               = 1e-4,    # T4 — CIFAR-10 default
        epochs           = 3,       # T4
        batch_size       = 128,     # T4
        momentum         = 0.9,
        weight_decay     = 5e-4,
        nesterov         = True,
        grad_norm_clip   = 1.0,     # T4
        per_dataset_lr   = dict(    # T4 dataset-specific overrides
            cifar10      = 1e-4,
            cifar100     = 5e-3,
            tinyimagenet = 5e-4,
        ),
    ),

    # ── Random Label ────────────────────────────────────────────────────────
    # T4: CIFAR-10 LR=1e-4, CIFAR-100 LR=3e-3, TinyIN LR=5e-4
    #     epochs=3, batch=128, no momentum listed (dash in table → use 0.9)
    random_label = dict(
        lr               = 1e-4,    # T4 — CIFAR-10 default
        epochs           = 3,       # T4
        batch_size       = 128,     # T4
        momentum         = 0.9,
        weight_decay     = 5e-4,
        nesterov         = True,
        grad_norm_clip   = None,
        per_dataset_lr   = dict(
            cifar10      = 1e-4,
            cifar100     = 3e-3,
            tinyimagenet = 5e-4,
        ),
    ),

    # ── SalUn ────────────────────────────────────────────────────────────────
    # T4: CIFAR-10 LR=1e-4, CIFAR-100 LR=1e-3, TinyIN LR=5e-4
    #     epochs=3, batch=128, threshold=0.5 (all datasets)
    salun = dict(
        lr               = 1e-4,    # T4 — CIFAR-10 default
        epochs           = 3,       # T4
        batch_size       = 128,     # T4
        momentum         = 0.9,
        weight_decay     = 5e-4,
        nesterov         = True,
        grad_norm_clip   = None,
        salun_threshold  = 0.5,     # T4
        per_dataset_lr   = dict(
            cifar10      = 1e-4,
            cifar100     = 1e-3,
            tinyimagenet = 5e-4,
        ),
    ),

    # ── SCRUB ────────────────────────────────────────────────────────────────
    # T4: CIFAR-10 LR=1e-4, CIFAR-100 LR=1e-3, TinyIN LR=5e-3
    #     epochs=3, del-batch=64, sgda-bsz=64, msteps=2 (all datasets)
    scrub = dict(
        lr               = 1e-4,    # T4 — CIFAR-10 default
        epochs           = 3,       # T4
        batch_size       = 64,      # T4 (del-batch)
        momentum         = 0.9,
        weight_decay     = 5e-4,
        nesterov         = True,
        scrub_del_bsz    = 64,      # T4
        scrub_sgda_bsz   = 64,      # T4
        scrub_msteps     = 2,       # T4
        scrub_gamma      = 0.99,    # §A.4 / prior defaults
        scrub_alpha      = 0.001,   # §A.4 / prior defaults
        scrub_kd_T       = 4.0,     # §A.4
        per_dataset_lr   = dict(
            cifar10      = 1e-4,
            cifar100     = 1e-3,
            tinyimagenet = 5e-3,
        ),
    ),

    # ── UNSIR / Tarun impair-repair ──────────────────────────────────────────
    # T4: CIFAR-10 LR=5e-5, CIFAR-100 LR=3e-5, TinyIN LR=2e-5
    #     epochs=3 (3 epochs impair/repair), batch=128 (all datasets)
    tarun = dict(
        lr               = 5e-5,    # T4 — CIFAR-10 default
        epochs           = 3,       # T4 (impair + repair)
        batch_size       = 128,     # T4
        momentum         = 0.9,
        weight_decay     = 5e-4,
        nesterov         = True,
        tarun_impair_lr  = 2e-4,    # §A.4 (impair phase LR)
        tarun_samples_per_class = 100,  # §A.4
        per_dataset_lr   = dict(
            cifar10      = 5e-5,
            cifar100     = 3e-5,
            tinyimagenet = 2e-5,
        ),
    ),

    # ── SVD (training-free) ──────────────────────────────────────────────────
    # T4: no epoch / LR (training-free).  Batch = sample count drawn from
    #     retain/forget datasets separately.
    # T4 alpha values:
    #   CIFAR-10/100:   alpha_r=1000, alpha_f=30
    #   Tiny-ImageNet:  alpha_r=30,   alpha_f=10
    # T4 sample counts (batch column):
    #   CIFAR-10:       900
    #   CIFAR-100:      990
    #   Tiny-ImageNet:  999
    svd = dict(
        lr               = None,    # training-free
        epochs           = 0,
        SVD_alpha_r      = 1000,    # T4 — CIFAR-10/100 default
        SVD_alpha_f      = 30,      # T4 — CIFAR-10/100 default
        SVD_samples      = 900,     # T4 — CIFAR-10 default
        SVD_max_patches  = 10,      # §A.4
        per_dataset      = dict(
            cifar10      = dict(SVD_alpha_r=1000, SVD_alpha_f=30,  SVD_samples=900),
            cifar100     = dict(SVD_alpha_r=1000, SVD_alpha_f=30,  SVD_samples=990),
            tinyimagenet = dict(SVD_alpha_r=30,   SVD_alpha_f=10,  SVD_samples=999),
        ),
    ),
)

# ---------------------------------------------------------------------------
# CMF-variant per-method hyperparameters — Table 4 (ResNet rows, "+ CMF")
#
# All CMF variants: epochs=4 for RL/SalUn, epochs=3 for NegGrad+/SCRUB/UNSIR.
# T4: CIFAR-10 LR values listed below; dataset-specific overrides in per_dataset_lr.
# ---------------------------------------------------------------------------
CMF_METHOD_CFG = dict(

    # ── Random Label + CMF ──────────────────────────────────────────────────
    # T4: CIFAR-10 LR=2e-3, CIFAR-100 LR=2e-3, TinyIN LR=1e-2, epochs=4
    random_label = dict(
        lr               = 2e-3,    # T4
        epochs           = 4,       # T4
        batch_size       = 128,
        momentum         = 0.9,
        weight_decay     = 5e-4,
        nesterov         = True,
        grad_norm_clip   = None,
        per_dataset_lr   = dict(
            cifar10      = 2e-3,
            cifar100     = 2e-3,
            tinyimagenet = 1e-2,
        ),
    ),

    # ── SalUn + CMF ─────────────────────────────────────────────────────────
    # T4: CIFAR-10 LR=2e-3, CIFAR-100 LR=2e-3, TinyIN LR=1e-2, epochs=4
    salun = dict(
        lr               = 2e-3,    # T4
        epochs           = 4,       # T4
        batch_size       = 128,
        momentum         = 0.9,
        weight_decay     = 5e-4,
        nesterov         = True,
        grad_norm_clip   = None,
        salun_threshold  = 0.5,
        per_dataset_lr   = dict(
            cifar10      = 2e-3,
            cifar100     = 2e-3,
            tinyimagenet = 1e-2,
        ),
    ),

    # ── NegGrad+ + CMF ──────────────────────────────────────────────────────
    # T4: CIFAR-10 LR=1e-4, CIFAR-100 LR=1e-4, TinyIN LR=3e-5, epochs=3
    grad_ascent_descent = dict(
        lr               = 1e-4,    # T4
        epochs           = 3,       # T4
        batch_size       = 128,
        momentum         = 0.9,
        weight_decay     = 5e-4,
        nesterov         = True,
        grad_norm_clip   = 1.0,
        per_dataset_lr   = dict(
            cifar10      = 1e-4,
            cifar100     = 1e-4,
            tinyimagenet = 3e-5,
        ),
    ),

    # ── SCRUB + CMF ─────────────────────────────────────────────────────────
    # T4: CIFAR-10 LR=5e-3, CIFAR-100 LR=5e-3, TinyIN LR=1e-3, epochs=3
    scrub = dict(
        lr               = 5e-3,    # T4
        epochs           = 3,       # T4
        batch_size       = 64,
        momentum         = 0.9,
        weight_decay     = 5e-4,
        nesterov         = True,
        scrub_del_bsz    = 64,
        scrub_sgda_bsz   = 64,
        scrub_msteps     = 2,
        scrub_gamma      = 0.99,
        scrub_alpha      = 0.001,
        scrub_kd_T       = 4.0,
        per_dataset_lr   = dict(
            cifar10      = 5e-3,
            cifar100     = 5e-3,
            tinyimagenet = 1e-3,
        ),
    ),

    # ── UNSIR + CMF ─────────────────────────────────────────────────────────
    # T4: CIFAR-10 LR=5e-5, CIFAR-100 LR=5e-5, TinyIN LR=2e-5, epochs=3
    tarun = dict(
        lr               = 5e-5,    # T4
        epochs           = 3,       # T4
        batch_size       = 128,
        momentum         = 0.9,
        weight_decay     = 5e-4,
        nesterov         = True,
        tarun_impair_lr  = 2e-4,
        tarun_samples_per_class = 100,
        per_dataset_lr   = dict(
            cifar10      = 5e-5,
            cifar100     = 5e-5,
            tinyimagenet = 2e-5,
        ),
    ),
)

# ---------------------------------------------------------------------------
# CMF classifier construction config (shared across all CMF methods)
# ---------------------------------------------------------------------------
CMF_CFG = dict(
    mean_source   = "train",   # "train" = μ and m_c from retain+forget (paper default)
                               # "retain" = ablation only
    CMF_momentum  = 0.9,
    temperature   = 1.0,
    CMFClassifier = True,
    remove_FC     = True,
)

# ---------------------------------------------------------------------------
# Stage-3 reduced learning rates (per method, already validated)
# ---------------------------------------------------------------------------
STAGE3_LR = dict(
    grad_ascent_descent = 5e-5,
    random_label        = 5e-5,
    salun               = 5e-5,
    scrub               = 5e-5,
    tarun               = 5e-5,
)

# Max epochs for Stage-3 encoder update
STAGE3_EPOCHS = 5

# ---------------------------------------------------------------------------
# Post-hoc (NB4b) config
# ---------------------------------------------------------------------------
POSTHOC_CFG = dict(
    k_posthoc_values  = [3],
    phase2_data_opts  = ["retain_only", "retain_plus_forget"],
    lr_posthoc        = 1e-3,
    momentum          = 0.9,
    weight_decay      = 0.0,
)

# ---------------------------------------------------------------------------
# Checkpoint directory layout
# ---------------------------------------------------------------------------
CKPT_DIRS = dict(
    pretrain        = "./checkpoints/pretrain",
    splits          = "./checkpoints/splits",
    oracle          = "./checkpoints/oracle",
    unlearn_no_cmf  = "./checkpoints/unlearn_no_cmf",   # NB3 — no-CMF baselines
    no_cmf          = "./checkpoints/no_cmf",
    cmf_static      = "./checkpoints/cmf_static",
    posthoc         = "./checkpoints/posthoc",
    stage3          = "./checkpoints/stage3",
    results         = "./results",
)

# ---------------------------------------------------------------------------
# NB3 — per-method unlearn hyperparameters (no-CMF baselines)
#
# This dict is the single authoritative source consumed by NB3
# (03_unlearn_no_cmf.ipynb).  Values are the same as METHOD_CFG above;
# they are re-exported here under the name UNLEARN_CFG_BY_METHOD so that
# NB3 can import a single, self-describing symbol rather than aliasing
# METHOD_CFG internally.
#
# Keys deliberately mirror METHOD_CFG exactly so that
#   UNLEARN_CFG_BY_METHOD[method]  is always valid for all 6 base methods.
# ---------------------------------------------------------------------------
UNLEARN_CFG_BY_METHOD = METHOD_CFG

# ---------------------------------------------------------------------------
# ViT fine-tuning config — Table 5 (ViT-S/16, ImageNet-pretrained backbone)
# ---------------------------------------------------------------------------
VIT_PRETRAIN_CFG = dict(
    epochs       = 10,      # T5
    lr           = 3e-4,    # T5 — CIFAR-10/100 default
    batch_size   = 128,     # T5
    momentum     = 0.9,
    weight_decay = 5e-4,
    nesterov     = True,
    pretrained   = True,    # T5 — initialized from ImageNet weights
    per_dataset_lr = dict(
        cifar10      = 3e-4,
        cifar100     = 3e-4,
        tinyimagenet = 1e-4,
    ),
)
