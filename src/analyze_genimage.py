import os
from pathlib import Path
import argparse
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# ============================================================
# 1. ARGUMENT PARSING
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Statistical analysis of a GenImage-like dataset "
            "(ai/nature, train/val, multiple generators)."
        )
    )

    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Path to the directory containing the generator folders (e.g. ./data/GenImage).",
    )

    parser.add_argument(
        "--generators",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Names of the generator folders inside root.\n"
            "Example:\n"
            "  --generators generador1_ADM_descomprimido "
            "generador2_BigGAN_descomprimido generador3_glide_descomprimido"
        ),
    )

    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val"],
        choices=["train", "val"],
        help="Dataset splits to analyze (default: train and val).",
    )

    parser.add_argument(
        "--sample_per_class",
        type=int,
        default=500,
        help=(
            "Maximum number of images per (generator, split, class) used to "
            "compute detailed statistics."
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Base directory where tables, figures and logs will be stored.",
    )

    return parser.parse_args()


# ============================================================
# 2. LOGGING
# ============================================================

def setup_logging(logs_dir: Path) -> None:
    """
    Configure logging to write:
      - to the console
      - to a file inside output/logs/
    """
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Timestamped log file so we don't overwrite previous runs
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"analysis_{ts}.log"

    log_format = "[%(asctime)s] [%(levelname)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Clear previous handlers if setup_logging is called multiple times
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    logging.info("Logging initialized.")
    logging.info(f"Log file: {log_file}")


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def list_images(folder: Path):
    """Iterate over all images in a directory recursively."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for root, _, files in os.walk(folder):
        for f in files:
            if Path(f).suffix.lower() in exts:
                yield Path(root) / f


def get_image_stats(path: Path) -> Optional[Dict[str, Any]]:
    """
    Compute basic statistics for a single image:
    - width, height, aspect_ratio
    - mean_r, mean_g, mean_b
    - std_r, std_g, std_b
    """
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            arr = np.asarray(im, dtype=np.float32) / 255.0  # H, W, 3

            mean_rgb = arr.mean(axis=(0, 1))
            std_rgb = arr.std(axis=(0, 1))
            aspect_ratio = w / max(h, 1)

            return {
                "width": w,
                "height": h,
                "aspect_ratio": aspect_ratio,
                "mean_r": float(mean_rgb[0]),
                "mean_g": float(mean_rgb[1]),
                "mean_b": float(mean_rgb[2]),
                "std_r": float(std_rgb[0]),
                "std_g": float(std_rgb[1]),
                "std_b": float(std_rgb[2]),
            }
    except Exception as e:
        logging.warning(f"Error reading image {path}: {e}")
        return None


def categorize_aspect_ratio(ar: float, tol: float = 0.05) -> str:
    """
    Roughly classify aspect ratio into categories:
    - 1:1
    - 4:3
    - 16:9
    - tall  (height > width)
    - wide  (width > height)
    """
    if abs(ar - 1.0) < tol:
        return "1:1"
    if abs(ar - 4 / 3) < tol:
        return "4:3"
    if abs(ar - 16 / 9) < tol:
        return "16:9"
    if ar < 1.0:
        return "tall"
    return "wide"


# ============================================================
# 4. MAIN ANALYSIS
# ============================================================

def analyze_dataset(
    root: Path,
    generators: List[str],
    splits: List[str],
    sample_per_class: int,
    out_tables_dir: Path,
    out_figures_dir: Path,
) -> None:
    """
    Iterate over all generators, splits and classes (ai/nature),
    collect counts and statistics, and generate tables + plots.
    """

    counts_rows: List[Dict[str, Any]] = []
    stats_rows: List[Dict[str, Any]] = []

    classes = ["ai", "nature"]

    for gen_name in generators:
        for split in splits:
            gen_split_root = root / gen_name / split

            for cls in classes:
                cls_root = gen_split_root / cls
                if not cls_root.exists():
                    logging.warning(f"Folder does not exist: {cls_root}")
                    continue

                logging.info(f"Scanning {cls_root} ...")
                paths = list(list_images(cls_root))
                n_images = len(paths)

                logging.info(
                    f"{n_images} images found in "
                    f"{gen_name} / {split} / {cls}"
                )

                counts_rows.append(
                    {
                        "generator": gen_name,
                        "split": split,
                        "class": cls,
                        "num_images": n_images,
                    }
                )

                if n_images == 0:
                    continue

                # Sampling for heavy statistics
                if n_images > sample_per_class:
                    rng = np.random.default_rng(seed=42)
                    idx = rng.choice(n_images, size=sample_per_class, replace=False)
                    sample_paths = [paths[i] for i in idx]
                    logging.info(
                        f"Sampling {sample_per_class} images "
                        f"out of {n_images}."
                    )
                else:
                    sample_paths = paths
                    logging.info(
                        f"Using all {n_images} images "
                        f"(below sample_per_class={sample_per_class})."
                    )

                for img_path in sample_paths:
                    st = get_image_stats(img_path)
                    if st is None:
                        continue

                    st["generator"] = gen_name
                    st["class"] = cls
                    st["split"] = split
                    st["path"] = str(img_path)
                    st["aspect_ratio_cat"] = categorize_aspect_ratio(
                        st["aspect_ratio"]
                    )
                    stats_rows.append(st)

    # ==============================
    # 4.1 TABLES (CSV)
    # ==============================

    counts_df = pd.DataFrame(counts_rows)
    stats_df = pd.DataFrame(stats_rows)

    counts_csv = out_tables_dir / "image_counts.csv"
    stats_csv = out_tables_dir / "image_stats_sample.csv"

    counts_df.to_csv(counts_csv, index=False)
    stats_df.to_csv(stats_csv, index=False)

    logging.info(f"Saved counts table to: {counts_csv}")
    logging.info(f"Saved statistics table to: {stats_csv}")

    # ==============================
    # 4.2 PLOTS
    # ==============================

    # ------------------------------------------------------------------
    # a) Image counts per generator / class / split
    # ------------------------------------------------------------------
    if not counts_df.empty:
        for split in counts_df["split"].unique():
            sub = counts_df[counts_df["split"] == split]
            if sub.empty:
                continue

            pivot_counts = sub.pivot_table(
                index="generator",
                columns="class",
                values="num_images",
                aggfunc="sum",
                fill_value=0,
            )

            ax = pivot_counts.plot(kind="bar")
            ax.set_ylabel("Number of images")
            ax.set_title(f"Image counts per generator ({split})")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            bar_path = out_figures_dir / f"counts_per_generator_{split}.png"
            plt.savefig(bar_path, dpi=200)
            plt.close()
            logging.info(f"Saved plot: {bar_path}")

    if stats_df.empty:
        logging.warning("No detailed statistics available (stats_df is empty).")
        return

    # ------------------------------------------------------------------
    # b) Histograms of width and height
    # ------------------------------------------------------------------
    plt.figure()
    stats_df["width"].hist(bins=30)
    plt.xlabel("Width (pixels)")
    plt.ylabel("Frequency")
    plt.title("Histogram of image widths")
    plt.tight_layout()
    w_hist = out_figures_dir / "width_hist.png"
    plt.savefig(w_hist, dpi=200)
    plt.close()
    logging.info(f"Saved plot: {w_hist}")

    plt.figure()
    stats_df["height"].hist(bins=30)
    plt.xlabel("Height (pixels)")
    plt.ylabel("Frequency")
    plt.title("Histogram of image heights")
    plt.tight_layout()
    h_hist = out_figures_dir / "height_hist.png"
    plt.savefig(h_hist, dpi=200)
    plt.close()
    logging.info(f"Saved plot: {h_hist}")

    # ------------------------------------------------------------------
    # c) Aspect ratio categories by class
    # ------------------------------------------------------------------
    plt.figure()
    crosstab_ar = pd.crosstab(stats_df["aspect_ratio_cat"], stats_df["class"])
    crosstab_ar.plot(kind="bar")
    plt.ylabel("Count")
    plt.title("Aspect ratio categories by class")
    plt.tight_layout()
    ar_plot = out_figures_dir / "aspect_ratio_by_class.png"
    plt.savefig(ar_plot, dpi=200)
    plt.close()
    logging.info(f"Saved plot: {ar_plot}")

    # ------------------------------------------------------------------
    # d) Luminance boxplot by class
    # ------------------------------------------------------------------
    stats_df["mean_luma"] = (
        0.2126 * stats_df["mean_r"]
        + 0.7152 * stats_df["mean_g"]
        + 0.0722 * stats_df["mean_b"]
    )

    plt.figure()
    stats_df.boxplot(column="mean_luma", by="class")
    plt.ylabel("Mean luminance (0-1)")
    plt.title("Luminance distribution by class")
    plt.suptitle("")
    plt.tight_layout()
    luma_box = out_figures_dir / "luminance_by_class.png"
    plt.savefig(luma_box, dpi=200)
    plt.close()
    logging.info(f"Saved plot: {luma_box}")

    # ------------------------------------------------------------------
    # e) PCA with simple features
    # ------------------------------------------------------------------
    feature_cols = [
        "width",
        "height",
        "aspect_ratio",
        "mean_r",
        "mean_g",
        "mean_b",
        "std_r",
        "std_g",
        "std_b",
    ]

    stats_df_clean = stats_df.dropna(subset=feature_cols).copy()
    if stats_df_clean.empty:
        logging.warning("No clean data for PCA (after dropping NaNs).")
        return

    features = stats_df_clean[feature_cols].values

    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-8
    features_norm = (features - mean) / std

    pca = PCA(n_components=2, random_state=42)
    pc = pca.fit_transform(features_norm)

    stats_df_clean["PC1"] = pc[:, 0]
    stats_df_clean["PC2"] = pc[:, 1]

    plt.figure()
    markers = {"ai": "o", "nature": "x"}
    for cls, marker in markers.items():
        subset = stats_df_clean[stats_df_clean["class"] == cls]
        if subset.empty:
            continue
        plt.scatter(
            subset["PC1"],
            subset["PC2"],
            marker=marker,
            alpha=0.5,
            label=cls,
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of simple image statistics (ai vs nature)")
    plt.legend()
    plt.tight_layout()
    pca_path = out_figures_dir / "pca_real_vs_ai.png"
    plt.savefig(pca_path, dpi=200)
    plt.close()
    logging.info(f"Saved plot: {pca_path}")

    logging.info("Analysis completed successfully.")


# ============================================================
# 5. MAIN ENTRY POINT
# ============================================================

def main() -> None:
    args = parse_args()

    root = Path(args.root)
    out_base = Path(args.output_dir)
    out_tables_dir = out_base / "tables"
    out_figures_dir = out_base / "figures"
    out_logs_dir = out_base / "logs"

    out_tables_dir.mkdir(parents=True, exist_ok=True)
    out_figures_dir.mkdir(parents=True, exist_ok=True)
    out_logs_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(out_logs_dir)

    logging.info(f"Dataset root: {root.resolve()}")
    logging.info(f"Generators: {args.generators}")
    logging.info(f"Splits: {args.splits}")
    logging.info(f"Output directory: {out_base.resolve()}")

    analyze_dataset(
        root=root,
        generators=args.generators,
        splits=args.splits,
        sample_per_class=args.sample_per_class,
        out_tables_dir=out_tables_dir,
        out_figures_dir=out_figures_dir,
    )


if __name__ == "__main__":
    main()
