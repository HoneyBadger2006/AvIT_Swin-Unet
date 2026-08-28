# Rebuilds meta_isic2017_train2000_seed42_dice37_shift5.csv: the official
# 2000-image training split plus the 185 validated hard-image augmentation
# variants (37 images x 5 techniques), matching the combined training set the
# reference SwinUnet run (results/isic2017_swinunet_clahe_seed42_dice37_shift5_
# SwinUnet_20260824_0145) was trained on.
#
# This file existed on disk when that SwinUnet run was launched but was not
# preserved. Rebuilding it here is pure data plumbing -- concatenating two
# already-validated, unmodified sources (meta_isic2017_train2000.csv and
# manifest_seed42_dice37_shift5.csv) -- not regenerating the augmentation
# itself, which is untouched (the 185 augmented .npy files already exist in
# Image_clahe/ and Label/, built by AViT/build_shift_augmentation.py).

from pathlib import Path

import pandas as pd

DATA_ROOT = Path(r"C:\Users\quanp\Downloads\ISIC 2017\data\isic2017")
MANIFEST = Path(
    r"C:\Users\quanp\Downloads\ISIC 2017\per_image_analysis_v2\bad_image_augmentation"
    r"\manifest_seed42_dice37_shift5.csv"
)
OUT_PATH = DATA_ROOT / "meta_isic2017_train2000_seed42_dice37_shift5.csv"


def main():
    train2000 = pd.read_csv(DATA_ROOT / "meta_isic2017_train2000.csv", dtype={"ID": str})
    manifest = pd.read_csv(MANIFEST, dtype={"ID": str, "source_id": str})

    aug_rows = manifest[["ID", "dataset", "diagnosis", "diagnosis_id"]].copy()

    combined = pd.concat([train2000, aug_rows], ignore_index=True)

    assert len(train2000) == 2000, f"expected 2000 base training rows, got {len(train2000)}"
    assert len(aug_rows) == 185, f"expected 185 augmented rows, got {len(aug_rows)}"
    assert len(combined) == 2185, f"expected 2185 combined rows, got {len(combined)}"
    assert combined["ID"].is_unique, "duplicate IDs in combined training meta"

    # Verify every ID actually has image+label .npy files on disk before writing
    # a meta CSV that claims otherwise.
    image_dir = DATA_ROOT / "Image_clahe"
    label_dir = DATA_ROOT / "Label"
    missing_img = [i for i in combined["ID"] if not (image_dir / f"{i}.npy").exists()]
    missing_lbl = [i for i in combined["ID"] if not (label_dir / f"{i}.npy").exists()]
    assert not missing_img, f"missing {len(missing_img)} image .npy files, e.g. {missing_img[:5]}"
    assert not missing_lbl, f"missing {len(missing_lbl)} label .npy files, e.g. {missing_lbl[:5]}"

    combined.to_csv(OUT_PATH, index=False)
    print(f"wrote {len(combined)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
