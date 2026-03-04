from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def save_corr_surface_png(out: Path, corr: np.ndarray, title: str = "phase correlation"):
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.imshow(corr, origin="lower")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def save_overlay_png(out: Path, ref: np.ndarray, cur_warp: np.ndarray, title: str):
    """
    Quick presentation-friendly overlay:
      - show ref in grayscale
      - add cur_warp as contours
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.imshow(ref, cmap="gray", origin="lower")
    cs = plt.contour(cur_warp, levels=6, linewidths=0.8)
    plt.clabel(cs, inline=True, fontsize=6)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
