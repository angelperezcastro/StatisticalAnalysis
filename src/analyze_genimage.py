import os
from pathlib import Path
import argparse
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# ============================================================
# 1. PARÁMETROS DE EJECUCIÓN
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Statistical analysis of GenImage-like dataset "
                    "(ai/nature, train/val, varios generadores)."
    )

    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Ruta al directorio que contiene los generadores "
             "(por ejemplo: ./data/GenImage)",
    )

    parser.add_argument(
        "--generators",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Nombres de las carpetas de generador dentro de root.\n"
            "Ejemplo para tu caso:\n"
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
        help="Splits a analizar (por defecto: train y val).",
    )

    parser.add_argument(
        "--sample_per_class",
        type=int,
        default=500,
        help=(
            "Número máximo de imágenes por (generator, split, class) para "
            "calcular estadísticas detalladas."
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directorio base donde se guardarán tablas y figuras.",
    )

    return parser.parse_args()


# ============================================================
# 2. FUNCIONES AUXILIARES
# ============================================================

def list_images(folder: Path):
    """Itera sobre todas las imágenes de un directorio."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for root, _, files in os.walk(folder):
        for f in files:
            if Path(f).suffix.lower() in exts:
                yield Path(root) / f


def get_image_stats(path: Path) -> Optional[Dict[str, Any]]:
    """
    Devuelve estadísticas básicas de una imagen:
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
    except Exception:
        # Si hay una imagen corrupta o similar, simplemente la saltamos.
        return None


def categorize_aspect_ratio(ar: float, tol: float = 0.05) -> str:
    """
    Clasifica la razón de aspecto en categorías aproximadas:
    - 1:1
    - 4:3
    - 16:9
    - tall (más alto que ancho)
    - wide (más ancho que alto)
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
# 3. ANÁLISIS PRINCIPAL
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
    Recorre todos los generadores, splits y clases (ai/nature),
    acumula conteos y estadísticas, y genera tablas + gráficos.
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
                    print(f"[WARNING] No existe la carpeta: {cls_root}")
                    continue

                print(f"[INFO] Escaneando {cls_root} ...")
                paths = list(list_images(cls_root))
                n_images = len(paths)

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

                # Muestreo para estadísticas pesadas
                if n_images > sample_per_class:
                    rng = np.random.default_rng(seed=42)
                    idx = rng.choice(n_images, size=sample_per_class, replace=False)
                    sample_paths = [paths[i] for i in idx]
                else:
                    sample_paths = paths

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
    # 3.1 TABLAS (CSV)
    # ==============================

    counts_df = pd.DataFrame(counts_rows)
    stats_df = pd.DataFrame(stats_rows)

    counts_csv = out_tables_dir / "image_counts.csv"
    stats_csv = out_tables_dir / "image_stats_sample.csv"

    counts_df.to_csv(counts_csv, index=False)
    stats_df.to_csv(stats_csv, index=False)

    print(f"[OK] Guardada tabla de conteos en: {counts_csv}")
    print(f"[OK] Guardada tabla de estadísticas en: {stats_csv}")

    # ==============================
    # 3.2 GRÁFICOS
    # ==============================

    # ------------------------------------------------------------------
    # a) Conteo de imágenes por generador / clase / split (bar plot)
    # ------------------------------------------------------------------
    if not counts_df.empty:
        # Un gráfico por split para que sea más claro
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
            print(f"[OK] Guardado gráfico: {bar_path}")

    if stats_df.empty:
        print("[WARNING] No hay estadísticas detalladas (stats_df vacío).")
        return

    # ------------------------------------------------------------------
    # b) Histogramas de width y height (todas las imágenes muestreadas)
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
    print(f"[OK] Guardado gráfico: {w_hist}")

    plt.figure()
    stats_df["height"].hist(bins=30)
    plt.xlabel("Height (pixels)")
    plt.ylabel("Frequency")
    plt.title("Histogram of image heights")
    plt.tight_layout()
    h_hist = out_figures_dir / "height_hist.png"
    plt.savefig(h_hist, dpi=200)
    plt.close()
    print(f"[OK] Guardado gráfico: {h_hist}")

    # ------------------------------------------------------------------
    # c) Distribución de aspect_ratio_cat por clase
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
    print(f"[OK] Guardado gráfico: {ar_plot}")

    # ------------------------------------------------------------------
    # d) Boxplot de luminancia por clase
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
    plt.suptitle("")  # Quita el título extra de pandas
    plt.tight_layout()
    luma_box = out_figures_dir / "luminance_by_class.png"
    plt.savefig(luma_box, dpi=200)
    plt.close()
    print(f"[OK] Guardado gráfico: {luma_box}")

    # ------------------------------------------------------------------
    # e) PCA con características simples
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

    # Filtramos posibles NaN/inf
    stats_df_clean = stats_df.dropna(subset=feature_cols).copy()
    if stats_df_clean.empty:
        print("[WARNING] No hay datos limpios para PCA.")
        return

    features = stats_df_clean[feature_cols].values

    # Normalización z-score
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
    print(f"[OK] Guardado gráfico: {pca_path}")


# ============================================================
# 4. MAIN
# ============================================================

def main() -> None:
    args = parse_args()

    root = Path(args.root)
    out_base = Path(args.output_dir)
    out_tables_dir = out_base / "tables"
    out_figures_dir = out_base / "figures"

    out_tables_dir.mkdir(parents=True, exist_ok=True)
    out_figures_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Root de dataset:", root.resolve())
    print("[INFO] Generadores:", args.generators)
    print("[INFO] Splits:", args.splits)
    print("[INFO] Directorio de salida:", out_base.resolve())

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
