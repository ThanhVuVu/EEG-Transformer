import argparse
import os
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from getData import SNOMED_MAPPING, parse_hea_label


CLASS_ID_TO_NAME = {
    0: "SR",
    1: "AFIB",
    2: "SB",
    3: "ST",
}


@dataclass
class RecordStats:
    mat_path: str
    label: int
    raw_shape: tuple
    raw_dtype: str
    raw_min: float
    raw_max: float
    raw_mean: float
    raw_std: float
    nan_frac: float


def _safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_val(mat_path: str) -> np.ndarray:
    mat_data = loadmat(mat_path)
    if "val" not in mat_data:
        raise KeyError(f"Missing 'val' key in mat file: {mat_path}")
    return mat_data["val"]


def _summarize_signal(x: np.ndarray) -> tuple[float, float, float, float, float]:
    x = np.asarray(x)
    nan_frac = float(np.isnan(x).mean()) if x.size else 0.0
    x_f = x.astype(np.float64, copy=False)
    return (
        float(np.nanmin(x_f)),
        float(np.nanmax(x_f)),
        float(np.nanmean(x_f)),
        float(np.nanstd(x_f)),
        nan_frac,
    )


def scan_dataset(root: str) -> tuple[list[str], list[int]]:
    hea_files = sorted([f for f in os.listdir(root) if f.endswith(".hea")])
    mat_paths: list[str] = []
    labels: list[int] = []

    for hea in hea_files:
        hea_path = os.path.join(root, hea)
        y = parse_hea_label(hea_path)
        if y is None:
            continue
        mat_path = hea_path[:-4] + ".mat"
        if not os.path.exists(mat_path):
            continue
        mat_paths.append(mat_path)
        labels.append(int(y))

    return mat_paths, labels


def plot_class_distribution(labels: list[int], out_dir: str) -> None:
    counts = {k: 0 for k in sorted(SNOMED_MAPPING.values())}
    for y in labels:
        counts[int(y)] += 1

    xs = list(counts.keys())
    ys = [counts[k] for k in xs]
    names = [f"{k}:{CLASS_ID_TO_NAME.get(k, str(k))}" for k in xs]

    plt.figure(figsize=(10, 5))
    plt.bar(names, ys)
    plt.title("Chapman 4-class label distribution (mapped SNOMED)")
    plt.ylabel("count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "class_distribution.png"), dpi=160)
    plt.close()


def plot_example_records(records: list[RecordStats], out_dir: str, max_per_class: int = 2) -> None:
    picked: dict[int, list[RecordStats]] = {0: [], 1: [], 2: [], 3: []}
    for r in records:
        if r.label in picked and len(picked[r.label]) < max_per_class:
            picked[r.label].append(r)
        if all(len(v) >= max_per_class for v in picked.values()):
            break

    for cls, recs in picked.items():
        for j, r in enumerate(recs):
            x = _load_val(r.mat_path)  # expected (12, 5000)
            t = np.arange(x.shape[1])
            fig, ax = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
            fig.suptitle(f"Example record | class={cls}:{CLASS_ID_TO_NAME.get(cls)} | file={os.path.basename(r.mat_path)}")

            # Plot a few standard leads for quick inspection
            lead_names = ["I", "II", "V1"]
            lead_idx = [0, 1, 6]
            for k in range(3):
                ax[k].plot(t, x[lead_idx[k]], linewidth=0.7)
                ax[k].set_ylabel(lead_names[k])
                ax[k].grid(True, alpha=0.2)
            ax[-1].set_xlabel("sample index (500Hz, 10s expected)")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"example_class{cls}_{j}.png"), dpi=160)
            plt.close()


def main():
    ap = argparse.ArgumentParser(description="EDA for WFDB_ChapmanShaoxing 12-lead ECG dataset.")
    ap.add_argument("--root", type=str, required=True, help="Path to WFDB_ChapmanShaoxing folder containing *.hea/*.mat")
    ap.add_argument("--out", type=str, default="./results/eda", help="Output directory for plots and summaries")
    ap.add_argument("--max_records", type=int, default=2000, help="Max number of records to load .mat for signal statistics")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for sampling records")
    args = ap.parse_args()

    _safe_makedirs(args.out)

    mat_paths, labels = scan_dataset(args.root)
    n_total = len(labels)
    print(f"Found {n_total} labeled records (mapped to 4 classes).")

    plot_class_distribution(labels, args.out)

    rng = np.random.default_rng(args.seed)
    idx = np.arange(n_total)
    rng.shuffle(idx)
    idx = idx[: min(args.max_records, n_total)]

    records: list[RecordStats] = []
    bad_files: list[tuple[str, str]] = []

    for i in idx:
        mp = mat_paths[int(i)]
        y = int(labels[int(i)])
        try:
            x = _load_val(mp)
            mn, mx, mean, std, nan_frac = _summarize_signal(x)
            records.append(
                RecordStats(
                    mat_path=mp,
                    label=y,
                    raw_shape=tuple(x.shape),
                    raw_dtype=str(x.dtype),
                    raw_min=mn,
                    raw_max=mx,
                    raw_mean=mean,
                    raw_std=std,
                    nan_frac=nan_frac,
                )
            )
        except Exception as e:
            bad_files.append((mp, str(e)))

    # Aggregate stats
    if records:
        shapes = {}
        for r in records:
            shapes[r.raw_shape] = shapes.get(r.raw_shape, 0) + 1
        print("Most common raw shapes (from sampled .mat):")
        for shp, c in sorted(shapes.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {shp}: {c}")

        mins = np.array([r.raw_min for r in records], dtype=np.float64)
        maxs = np.array([r.raw_max for r in records], dtype=np.float64)
        means = np.array([r.raw_mean for r in records], dtype=np.float64)
        stds = np.array([r.raw_std for r in records], dtype=np.float64)
        nans = np.array([r.nan_frac for r in records], dtype=np.float64)

        summary_path = os.path.join(args.out, "signal_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"records_sampled={len(records)}\n")
            f.write(f"bad_files={len(bad_files)}\n")
            f.write(f"min(min)={mins.min():.4f}\n")
            f.write(f"max(max)={maxs.max():.4f}\n")
            f.write(f"mean(mean)={means.mean():.4f}\n")
            f.write(f"mean(std)={stds.mean():.4f}\n")
            f.write(f"max(nan_frac)={nans.max():.6f}\n")

        print(f"Wrote summary to {summary_path}")

        # Example plots per class
        plot_example_records(records, args.out, max_per_class=2)

    if bad_files:
        bad_path = os.path.join(args.out, "bad_files.txt")
        with open(bad_path, "w") as f:
            for mp, err in bad_files[:500]:
                f.write(f"{mp}\t{err}\n")
        print(f"Wrote bad file list (first 500) to {bad_path}")

    print(f"EDA outputs saved under: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()

