#!/usr/bin/env python3
"""
visualize_results.py  -  ORB-SLAM3 trajectory evaluation
=========================================================
Usage:
    python visualize_results.py --est KeyFrameTrajectory.txt \
                                 --gt  groundtruth.txt \
                                 [--dataset "fr3_walking_xyz"] \
                                 [--out results.png]
"""

import argparse, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("TkAgg")          # interactive on Windows; change to Agg for headless
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ──────────────────────────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────────────────────────

def load_tum(path):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 8:
                data.append([float(v) for v in parts])
    if not data:
        raise ValueError(f"No valid poses in {path}")
    return np.array(data, dtype=np.float64)


def associate(est, gt, max_diff=0.02):
    matches_e, matches_g = [], []
    for i, t in enumerate(est[:, 0]):
        idx = np.argmin(np.abs(gt[:, 0] - t))
        if np.abs(gt[idx, 0] - t) < max_diff:
            matches_e.append(i); matches_g.append(idx)
    return est[matches_e], gt[matches_g]


# ──────────────────────────────────────────────────────────────────
# Alignment  (Umeyama - Horn's method, with optional scale)
# ──────────────────────────────────────────────────────────────────

def umeyama(src, dst, with_scale=True):
    """Align src -> dst.  Returns (R 3x3, t 3, s scalar)."""
    n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    var_s = np.mean(np.sum(sc**2, axis=1))
    cov   = dc.T @ sc / n
    U, D, Vt = np.linalg.svd(cov)
    sign = np.ones(3); sign[-1] = np.linalg.det(U) * np.linalg.det(Vt)
    R = U @ np.diag(sign) @ Vt
    s = (D * sign).sum() / var_s if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return R, t, s


def apply_tf(pts, R, t, s=1.0):
    return (s * R @ pts.T).T + t


# ──────────────────────────────────────────────────────────────────
# ATE
# ──────────────────────────────────────────────────────────────────

def ate(est_xyz, gt_xyz):
    """Full 3D ATE."""
    e = np.linalg.norm(est_xyz - gt_xyz, axis=1)
    return dict(errors=e, rmse=float(np.sqrt(np.mean(e**2))),
                mean=float(e.mean()), median=float(np.median(e)),
                std=float(e.std()), min=float(e.min()), max=float(e.max()))


def ate2d(est_xyz, gt_xyz):
    """2D ATE (XY only) — more meaningful for aerial monocular SLAM
    where Z (altitude) cannot be reliably estimated from a top-down camera."""
    e = np.linalg.norm(est_xyz[:, :2] - gt_xyz[:, :2], axis=1)
    return dict(errors=e, rmse=float(np.sqrt(np.mean(e**2))),
                mean=float(e.mean()), median=float(np.median(e)),
                std=float(e.std()), min=float(e.min()), max=float(e.max()))


# ──────────────────────────────────────────────────────────────────
# Style helpers
# ──────────────────────────────────────────────────────────────────

BLUE   = "#1565C0"
RED    = "#C62828"
ORANGE = "#E65100"
GREEN  = "#2E7D32"
GREY   = "#607D8B"

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor("#F8F9FA")
    ax.grid(color="#DEE2E6", lw=0.6, linestyle="--")
    ax.tick_params(colors="#333", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#CED4DA")
    if title:  ax.set_title(title, fontsize=9, fontweight="bold",
                            color="#212529", pad=5)
    if xlabel: ax.set_xlabel(xlabel, fontsize=8, color="#495057")
    if ylabel: ax.set_ylabel(ylabel, fontsize=8, color="#495057")


# ──────────────────────────────────────────────────────────────────
# Figure
# ──────────────────────────────────────────────────────────────────

def make_figure(est_aligned, gt_xyz, ate_info, ate2d_info, scale_factor,
                dataset_name, est_path, n_matched, timestamps):

    errors  = ate_info["errors"]
    rel_t   = timestamps - timestamps[0]

    # Error colourmap: green (low) -> yellow -> red (high)
    cmap = plt.cm.RdYlGn_r
    norm = mcolors.Normalize(vmin=0, vmax=np.percentile(errors, 95))

    fig = plt.figure(figsize=(20, 10), facecolor="white")
    fig.suptitle(
        f"ORB-SLAM3  |  Trajectory Evaluation  |  {dataset_name}",
        fontsize=14, fontweight="bold", color="#212529", y=0.98)

    # Layout: 3 rows x 4 cols
    # Col 0   : X/Y/Z vs Time (stacked)
    # Col 1   : 2D heatmap trajectory
    # Col 2   : 3D trajectory
    # Col 3   : ATE per frame (top) + stats card (bottom)
    gs = gridspec.GridSpec(
        3, 4, figure=fig,
        left=0.05, right=0.97, top=0.93, bottom=0.07,
        hspace=0.42, wspace=0.35,
        width_ratios=[1.1, 1.4, 1.3, 1.0],
    )

    # ── Col 0: X / Y / Z vs Time  ────────────────────────────────
    axes_t = []
    for row, (lbl, hide) in enumerate([("X (m)", True), ("Y (m)", True), ("Z (m)", False)]):
        ax = fig.add_subplot(gs[row, 0],
                             sharex=(axes_t[0] if axes_t else None))
        axes_t.append(ax)
        style_ax(ax,
                 title="Aligned Position vs Time" if row == 0 else "",
                 xlabel="" if hide else "Time (s)", ylabel=lbl)

        ax.plot(rel_t, gt_xyz[:, row],
                color=RED, lw=1.6, ls="--", label="Ground Truth", zorder=3)
        ax.plot(rel_t, est_aligned[:, row],
                color=BLUE, lw=1.3, label="Estimate", zorder=4)

        # Pink shading between estimate and GT to show error
        ax.fill_between(rel_t,
                        np.minimum(est_aligned[:, row], gt_xyz[:, row]),
                        np.maximum(est_aligned[:, row], gt_xyz[:, row]),
                        color="#EF9A9A", alpha=0.30)

        ax.legend(fontsize=7, loc="upper right",
                  framealpha=0.85, edgecolor="#CED4DA")
        if hide:
            plt.setp(ax.get_xticklabels(), visible=False)

    # ── Col 1: 2D trajectory with ATE heatmap  ───────────────────
    ax2 = fig.add_subplot(gs[:, 1])
    style_ax(ax2,
             title="Aligned Trajectory 2D  (coloured by ATE)",
             xlabel="X (m)", ylabel="Y (m)")

    # Ground truth dashed line
    ax2.plot(gt_xyz[:, 0], gt_xyz[:, 1],
             color=RED, lw=2.2, ls="--", label="Ground Truth", zorder=3, alpha=0.85)

    # Estimated trajectory coloured segment-by-segment by per-point ATE
    for i in range(len(est_aligned) - 1):
        c = cmap(norm(errors[i]))
        ax2.plot(est_aligned[i:i+2, 0], est_aligned[i:i+2, 1],
                 color=c, lw=2.0, zorder=4)

    # Start / End markers
    ax2.scatter(*est_aligned[0, :2],  s=90, color=GREEN,  zorder=7,
                marker="^", label="Est Start")
    ax2.scatter(*est_aligned[-1, :2], s=90, color=ORANGE, zorder=7,
                marker="s", label="Est End")
    ax2.scatter(*gt_xyz[0, :2],  s=90, color=GREEN,  zorder=7,
                marker="^", facecolors="none", linewidths=2)
    ax2.scatter(*gt_xyz[-1, :2], s=90, color=ORANGE, zorder=7,
                marker="s", facecolors="none", linewidths=2)

    ax2.set_aspect("equal")
    ax2.legend(fontsize=8, loc="upper right",
               framealpha=0.85, edgecolor="#CED4DA")

    # ATE colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax2, fraction=0.032, pad=0.03)
    cbar.set_label("ATE (m)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # ── Col 2: 3D trajectory  ────────────────────────────────────
    ax3 = fig.add_subplot(gs[:, 2], projection="3d")
    ax3.set_title("Aligned Trajectory 3D", fontsize=9, fontweight="bold",
                  color="#212529", pad=4)
    ax3.plot(*gt_xyz.T,      color=RED,  lw=1.8, ls="--", label="Ground Truth")
    ax3.plot(*est_aligned.T, color=BLUE, lw=1.3,           label="Estimate")
    ax3.scatter(*est_aligned[0],  s=50, color=GREEN,  zorder=5)
    ax3.scatter(*est_aligned[-1], s=50, color=ORANGE, zorder=5)
    ax3.set_xlabel("X (m)", fontsize=7)
    ax3.set_ylabel("Y (m)", fontsize=7)
    ax3.set_zlabel("Z (m)", fontsize=7)
    ax3.tick_params(labelsize=6)
    for pane in (ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane):
        pane.fill = False; pane.set_edgecolor("#CED4DA")
    ax3.legend(fontsize=7, loc="upper left",
               facecolor="white", edgecolor="#CED4DA")

    # ── Col 3 top: 3D ATE per frame  ──────────────────────────────
    ax_ate3 = fig.add_subplot(gs[0, 3])
    style_ax(ax_ate3, title="3D ATE per Frame (incl. Z)",
             xlabel="", ylabel="Error (m)")
    idx = np.arange(len(errors))
    ax_ate3.fill_between(idx, errors, color="#EF9A9A", alpha=0.5)
    ax_ate3.plot(idx, errors, color=RED, lw=0.9)
    ax_ate3.axhline(ate_info["rmse"], color=BLUE, lw=1.5, ls="--",
                    label=f'RMSE {ate_info["rmse"]:.2f} m')
    ax_ate3.legend(fontsize=7, framealpha=0.85, edgecolor="#CED4DA")
    plt.setp(ax_ate3.get_xticklabels(), visible=False)

    # ── Col 3 mid: 2D ATE per frame (XY only) ────────────────────
    errors2d = ate2d_info["errors"]
    ax_ate2 = fig.add_subplot(gs[1, 3], sharex=ax_ate3)
    style_ax(ax_ate2, title="2D ATE per Frame (XY only)",
             xlabel="Frame #", ylabel="Error (m)")
    ax_ate2.fill_between(idx, errors2d, color="#BBDEFB", alpha=0.6)
    ax_ate2.plot(idx, errors2d, color=BLUE, lw=0.9)
    ax_ate2.axhline(ate2d_info["rmse"],   color=GREEN, lw=1.5, ls="--",
                    label=f'RMSE {ate2d_info["rmse"]:.2f} m')
    ax_ate2.axhline(ate2d_info["median"], color=GREY,  lw=1.2, ls=":")
    ax_ate2.legend(fontsize=7, framealpha=0.85, edgecolor="#CED4DA")

    # ── Col 3 bottom: stats card  ─────────────────────────────────
    ax_s = fig.add_subplot(gs[2, 3])
    ax_s.set_facecolor("#F1F3F5"); ax_s.axis("off")

    est_len = float(np.sum(np.linalg.norm(np.diff(est_aligned, axis=0), axis=1)))
    gt_len  = float(np.sum(np.linalg.norm(np.diff(gt_xyz,      axis=0), axis=1)))

    rows = [
        ("RESULTS SUMMARY",  "",                            "#212529", "#212529", True),
        ("─" * 20,           "",                            "#ADB5BD", "#ADB5BD", False),
        ("Dataset",          dataset_name,                  GREY,      "#212529", False),
        ("Matched frames",   str(n_matched),                GREY,      "#212529", False),
        ("Scale factor",     f"x{scale_factor:.4f}",        GREY,      ORANGE,    False),
        ("─" * 20,           "",                            "#ADB5BD", "#ADB5BD", False),
        ("3D ATE RMSE",      f'{ate_info["rmse"]:.3f} m',   GREY,      RED,       False),
        ("3D ATE Mean",      f'{ate_info["mean"]:.3f} m',   GREY,      "#212529", False),
        ("3D ATE Max",       f'{ate_info["max"]:.3f} m',    GREY,      "#212529", False),
        ("─" * 20,           "",                            "#ADB5BD", "#ADB5BD", False),
        ("2D ATE RMSE",      f'{ate2d_info["rmse"]:.3f} m', GREY,      BLUE,      False),
        ("2D ATE Mean",      f'{ate2d_info["mean"]:.3f} m', GREY,      "#212529", False),
        ("2D ATE Max",       f'{ate2d_info["max"]:.3f} m',  GREY,      "#212529", False),
        ("─" * 20,           "",                            "#ADB5BD", "#ADB5BD", False),
        ("Est. path",        f"{est_len:.1f} m",            GREY,      "#212529", False),
        ("GT  path",         f"{gt_len:.1f} m",             GREY,      "#212529", False),
    ]

    rh = 0.068
    for k, (label, value, lc, vc, bold) in enumerate(rows):
        y  = 0.97 - k * rh
        fw = "bold" if bold else "normal"
        ax_s.text(0.03, y, label, transform=ax_s.transAxes,
                  fontsize=8, color=lc, fontweight=fw,
                  fontfamily="monospace", va="top")
        if value:
            ax_s.text(0.97, y, value, transform=ax_s.transAxes,
                      fontsize=8, color=vc, fontweight=fw,
                      fontfamily="monospace", va="top", ha="right")

    return fig


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--est",      required=True,  help="KeyFrameTrajectory.txt")
    ap.add_argument("--gt",       required=True,  help="groundtruth.txt (TUM)")
    ap.add_argument("--dataset",  default="TUM fr3_walking_xyz")
    ap.add_argument("--out",      default=None,   help="Save to PNG (optional)")
    ap.add_argument("--max_diff", type=float, default=0.02)
    args = ap.parse_args()

    print(f"[+] Loading estimated : {args.est}")
    est_raw = load_tum(args.est);  print(f"    {len(est_raw)} poses")

    print(f"[+] Loading GT        : {args.gt}")
    gt_raw  = load_tum(args.gt);   print(f"    {len(gt_raw)} poses")

    print("[+] Associating ...")
    est_m, gt_m = associate(est_raw, gt_raw, args.max_diff)
    n = len(est_m);  print(f"    {n} matched pairs")

    if n < 3:
        sys.exit("ERROR: fewer than 3 matched pairs – check timestamps or raise --max_diff")

    est_xyz    = est_m[:, 1:4]
    gt_xyz     = gt_m[:, 1:4]
    timestamps = est_m[:, 0]

    print("[+] Aligning WITH scale (monocular) ...")
    R, t, s = umeyama(est_xyz, gt_xyz, with_scale=True)
    est_al  = apply_tf(est_xyz, R, t, s)
    print(f"    scale factor = {s:.4f}")

    print("[+] Computing ATE ...")
    info    = ate(est_al, gt_xyz)
    info_2d = ate2d(est_al, gt_xyz)
    print(f"    3D  RMSE={info['rmse']:.4f} m   Mean={info['mean']:.4f} m   Max={info['max']:.4f} m")
    print(f"    2D  RMSE={info_2d['rmse']:.4f} m   Mean={info_2d['mean']:.4f} m   Max={info_2d['max']:.4f} m  (XY only)")

    print("[+] Building figure ...")
    fig = make_figure(est_al, gt_xyz, info, info_2d, s,
                      args.dataset, args.est, n, timestamps)

    if args.out:
        fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"[+] Saved -> {args.out}")

    plt.show()
    print("Done.")


if __name__ == "__main__":
    main()