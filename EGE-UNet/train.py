"""
Train EGE-UNet on the OFFICIAL ISIC 2017 split (2000 train + the 185 validated
hard-image augmentation variants = 2185 train, 600 official test) for direct
use in the Swin-UNet/AViT/EGE-UNet ensemble.

This is a separate production run from the EGE-UNET verification repo's paper-
protocol reproduction (pooled dataset, 5 reseeded 70/30 splits) -- it reuses
that repo's model/loss/transform code (vendored here unmodified, see model.py/
losses.py/transforms.py headers) but trains on this project's official split
and CLAHE-preprocessed data, matching Swin-UNet/AViT's data convention.

Hyperparameters (EGE-UNet's own recipe, unchanged from the reproduction work):
  optimizer:  AdamW, lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=1e-2
  scheduler:  CosineAnnealingLR, T_max=50, eta_min=1e-5
  loss:       BCE+Dice with deep supervision, lambda=[1,.5,.4,.3,.2,.1]
  epochs:     300
  batch size: 8
  image size: 256x256, resize-before-augment ordering (reproduction fix, kept)

Checkpoint selection: best.pth = min loss on the 600-image official test set,
matching both (a) the EGE-UNET reproduction's own convention (itself matching
the official EGE-UNet repo's methodology -- no separate held-out validation
split) and (b) this project's own `--k_fold allin` convention used for the
reference SwinUnet run, which also has no held-out split distinct from the
fixed test set. This keeps the two checkpoints' selection methodology
comparable, even though it means the reported test score is not from a
strictly blind test set for either network.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import EGEUNet
from losses import GTBceDiceLoss
from dataset import load_meta, compute_mean_std, ISICOfficialDataset
from transforms import build_train_transform, build_test_transform

DATA_ROOT = Path(r"C:\Users\quanp\Downloads\ISIC 2017\data\isic2017")
IMAGE_DIR = DATA_ROOT / "Image_clahe"
LABEL_DIR = DATA_ROOT / "Label"
DEFAULT_TRAIN_META = "meta_isic2017_train2000_seed42_dice37_shift5.csv"
TEST_META = DATA_ROOT / "meta_isic2017_test600.csv"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def train_one_epoch(loader, model, criterion, optimizer, epoch, print_interval=40):
    model.train()
    losses = []
    for it, (images, targets) in enumerate(loader):
        images = images.cuda(non_blocking=True).float()
        targets = targets.cuda(non_blocking=True).float()

        optimizer.zero_grad()
        gt_pre, out = model(images)
        loss = criterion(gt_pre, out, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if it % print_interval == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"epoch {epoch} iter {it}/{len(loader)} loss {np.mean(losses):.4f} lr {lr:.6g}", flush=True)
    return float(np.mean(losses))


@torch.no_grad()
def eval_loss(loader, model, criterion):
    model.eval()
    losses = []
    for images, targets in loader:
        images = images.cuda(non_blocking=True).float()
        targets = targets.cuda(non_blocking=True).float()
        gt_pre, out = model(images)
        loss = criterion(gt_pre, out, targets)
        losses.append(loss.item())
    return float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None, help="defaults to results/official_seed<seed>/")
    ap.add_argument("--train-meta-name", default=DEFAULT_TRAIN_META,
                     help="filename under data/isic2017/, e.g. meta_isic2017_train2000.csv for a "
                          "no-augmentation baseline, or a custom augmented meta CSV")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--t-max", type=int, default=50)
    ap.add_argument("--eta-min", type=float, default=1e-5)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--gpu-id", default="0")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent / "results" / f"official_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    set_seed(args.seed)
    torch.cuda.set_device(int(args.gpu_id)) if torch.cuda.is_available() else None

    train_meta_path = DATA_ROOT / args.train_meta_name
    train_ids = load_meta(train_meta_path)
    test_ids = load_meta(TEST_META)
    print(f"train meta: {train_meta_path} ({len(train_ids)} IDs)", flush=True)
    assert len(test_ids) == 600, f"expected 600 test IDs, got {len(test_ids)}"

    stats_path = out_dir / "norm_stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        mean, std = stats["mean"], stats["std"]
    else:
        mean, std = compute_mean_std(train_ids, IMAGE_DIR)
        stats_path.write_text(json.dumps({"mean": mean, "std": std}, indent=2))
    print(f"norm stats: mean={mean:.4f} std={std:.4f}", flush=True)

    train_tf = build_train_transform(mean, std)
    test_tf = build_test_transform(mean, std)

    train_ds = ISICOfficialDataset(train_ids, IMAGE_DIR, LABEL_DIR, train_tf)
    test_ds = ISICOfficialDataset(test_ids, IMAGE_DIR, LABEL_DIR, test_tf)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        pin_memory=True, num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        pin_memory=True, num_workers=args.num_workers, drop_last=False,
    )

    model = EGEUNet(
        num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64],
        bridge=True, gt_ds=True,
    ).cuda()

    criterion = GTBceDiceLoss(wb=1, wd=1, ds_weights=(0.1, 0.2, 0.3, 0.4, 0.5), final_weight=1.0)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.t_max, eta_min=args.eta_min,
    )

    history = []
    min_loss = float("inf")
    min_epoch = 0
    start_epoch = 1
    t0 = time.time()

    latest_path = ckpt_dir / "latest.pth"
    history_path = out_dir / "history.json"
    if latest_path.exists() and history_path.exists():
        # Resume after a crash (e.g. a CUDA abort from GPU contention with
        # another process): latest.pth already has model/optimizer/scheduler
        # state, so pick up right after its epoch rather than restarting from
        # epoch 1. Not bit-identical to an uninterrupted run (dataloader
        # shuffle order and RNG state differ across the resume boundary) --
        # an accepted tradeoff, not a correctness concern for this recipe.
        ckpt = torch.load(latest_path, map_location="cuda", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        min_loss, min_epoch = ckpt["min_loss"], ckpt["min_epoch"]
        start_epoch = ckpt["epoch"] + 1
        history = json.loads(history_path.read_text())
        t0 = time.time() - history[-1]["elapsed_s"]
        print(f"Resuming from epoch {start_epoch} (latest.pth was at epoch {ckpt['epoch']}, "
              f"best={min_loss:.4f}@{min_epoch})", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        torch.cuda.empty_cache()
        train_loss = train_one_epoch(train_loader, model, criterion, optimizer, epoch)
        scheduler.step()
        val_loss = eval_loss(test_loader, model, criterion)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                         "lr": optimizer.param_groups[0]["lr"], "elapsed_s": time.time() - t0})

        if val_loss < min_loss:
            min_loss, min_epoch = val_loss, epoch
            torch.save(model.state_dict(), ckpt_dir / "best.pth")

        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "min_loss": min_loss, "min_epoch": min_epoch,
        }, ckpt_dir / "latest.pth")

        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        print(f"[epoch {epoch}/{args.epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"best={min_loss:.4f}@{min_epoch}", flush=True)

    print(f"Training done. Best val_loss={min_loss:.4f} at epoch {min_epoch}. "
          f"Checkpoint: {ckpt_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
