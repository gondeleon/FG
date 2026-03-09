#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manual OSM ↔ RADAR calibrator (AOI_GPKG-only), with polygon/line overlays + GUI buttons.

This is the "old-style" calibrator, focused on visual alignment of OSM layers to RADAR in the AOI. It does not use any track points or GPS data, and is meant for quick manual adjustments when the initial misalignment is large.:
- RADAR as background image (cartesian)
- OSM layers as real geometries (water fill, buildings fill+red edge, coastline line)
- Buttons + keyboard shortcuts to adjust Sim(2) and save JSON

- AOI is in UTM absolute coordinates;
- RADAR display is local meters around (0,0).
- We localize OSM by subtracting an origin (E0,N0), default from first point in 'track'.

Sim(2) model:
  x_radar = s * R(theta) * x_osm_local + t
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import matplotlib
try:
    matplotlib.use("QtAgg")
except Exception:
    pass
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from PIL import Image

import geopandas as gpd
from shapely.affinity import affine_transform
from shapely.ops import unary_union


# ----------------------------
# Sim(2)
# ----------------------------
@dataclass
class Sim2:
    s: float = 1.0
    rot_deg: float = 0.0
    tx: float = 0.0
    ty: float = 0.0

    def affine_params(self) -> Tuple[float, float, float, float, float, float]:
        """Params for shapely.affinity.affine_transform: [a, b, d, e, xoff, yoff]."""
        th = math.radians(self.rot_deg)
        c, sn = math.cos(th), math.sin(th)
        a = self.s * c
        b = -self.s * sn
        d = self.s * sn
        e = self.s * c
        return (a, b, d, e, self.tx, self.ty)

    def M3(self) -> np.ndarray:
        a, b, d, e, xoff, yoff = self.affine_params()
        return np.array([[a, b, xoff],
                         [d, e, yoff],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    @staticmethod
    def from_json(d: dict) -> "Sim2":
        return Sim2(
            s=float(d.get("s", 1.0)),
            rot_deg=float(d.get("rot_deg", d.get("yaw_deg", 0.0))),
            tx=float(d.get("tx", d.get("tx_m", 0.0))),
            ty=float(d.get("ty", d.get("ty_m", 0.0))),
        )


# ----------------------------
# Radar polar -> cart (single frame)
# ----------------------------
def load_radar_polar_u8(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    I = np.asarray(img, dtype=np.uint8)
    if I.ndim != 2:
        raise ValueError(f"Expected grayscale radar PNG, got shape {I.shape}")
    return I

def polar_to_cart(I_pol: np.ndarray, r_max: float, cart_res: int, az_cw: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """I_pol: (Naz, Nrg) -> I_cart uint8 (cart_res, cart_res) with extent [-r_max, r_max]."""
    Naz, Nrg = I_pol.shape
    xs = np.linspace(-r_max, r_max, cart_res, dtype=np.float32)
    ys = np.linspace(-r_max, r_max, cart_res, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)

    R = np.sqrt(X*X + Y*Y)
    TH = np.mod(np.arctan2(Y, X), 2.0*np.pi)
    t_idx = (TH / (2.0*np.pi)) * (Naz - 1)
    if az_cw:
        t_idx = (Naz - 1) - t_idx

    r_idx = (R / r_max) * (Nrg - 1)


    # --- clamp indices to avoid sampling outside polar image (corners have R > r_max) ---
    oor = (R > r_max)
    r_idx = np.clip(r_idx, 0.0, float(Nrg - 1))
    t_idx = np.clip(t_idx, 0.0, float(Naz - 1))
    # (optional but clean) send OOR to a safe index; will be zeroed after
    r_idx[oor] = 0.0
    t_idx[oor] = 0.0

    # bilinear sampling
    r0 = np.floor(r_idx).astype(np.int32)
    t0 = np.floor(t_idx).astype(np.int32)
    r1 = np.clip(r0 + 1, 0, Nrg - 1)
    t1 = np.clip(t0 + 1, 0, Naz - 1)
    fr = (r_idx - r0).astype(np.float32)
    ft = (t_idx - t0).astype(np.float32)


    I = I_pol.astype(np.float32)
    v00 = I[t0, r0]
    v01 = I[t0, r1]
    v10 = I[t1, r0]
    v11 = I[t1, r1]
    I_cart = (v00*(1-fr)*(1-ft) + v01*(fr)*(1-ft) + v10*(1-fr)*(ft) + v11*(fr)*(ft))

    I_cart[R > r_max] = 0.0

    # robust normalization for display
    nz = I_cart[I_cart > 0]
    if nz.size:
        lo, hi = np.percentile(nz, 1.0), np.percentile(nz, 99.5)
        if hi > lo + 1e-6:
            I_cart = np.clip((I_cart - lo) / (hi - lo), 0.0, 1.0) * 255.0
    return I_cart.astype(np.uint8), xs, ys


# ----------------------------
# OSM load + localize + union
# ----------------------------
def _safe_read(gpkg: Path, layer: str, available_layers: Optional[list]) -> Optional[gpd.GeoDataFrame]:
    if available_layers is not None and layer not in available_layers:
        return None
    try:
        g = gpd.read_file(gpkg, layer=layer)
        g = g[g.geometry.notna() & ~g.geometry.is_empty]
        return None if g.empty else g
    except Exception:
        return None

def first_track_point_xy(track_gdf: Optional[gpd.GeoDataFrame]) -> Optional[np.ndarray]:
    if track_gdf is None or track_gdf.empty:
        return None
    geom = track_gdf.geometry.iloc[0]
    if geom is None or geom.is_empty:
        return None
    try:
        coords = np.asarray(geom.coords, dtype=np.float64)
        if coords.shape[0] >= 1:
            return coords[0, :2]
    except Exception:
        pass
    return None

def centroid_xy(gdf: Optional[gpd.GeoDataFrame]) -> Optional[np.ndarray]:
    if gdf is None or gdf.empty:
        return None
    geom = gdf.geometry.unary_union
    if geom is None or geom.is_empty:
        return None
    c = geom.centroid
    return np.array([c.x, c.y], dtype=np.float64)

def localize_gdf(gdf: Optional[gpd.GeoDataFrame], E0: float, N0: float) -> Optional[gpd.GeoDataFrame]:
    if gdf is None or gdf.empty:
        return gdf
    out = gdf.copy()
    out["geometry"] = out.geometry.apply(lambda geom: affine_transform(geom, [1, 0, 0, 1, -E0, -N0]) if geom and not geom.is_empty else geom)
    return out

def union_layer(gdf: Optional[gpd.GeoDataFrame]):
    if gdf is None or gdf.empty:
        return None
    try:
        return unary_union(gdf.geometry.values)
    except Exception:
        return unary_union(gdf.geometry)

def apply_sim2_flip_geom(geom, sim2: Sim2, flip_x: bool, flip_y: bool):
    if geom is None:
        return None
    if geom.is_empty:
        return geom

    # First: flip in local OSM frame (before Sim2)
    a = -1.0 if flip_x else 1.0
    e = -1.0 if flip_y else 1.0
    g = affine_transform(geom, [a, 0, 0, e, 0, 0])

    # Then: Sim2 (scale+rot+trans)
    return affine_transform(g, list(sim2.affine_params()))



# ----------------------------
# GUI
# ----------------------------
class CalibGUI:
    def __init__(self,
                 I_cart: np.ndarray,
                 xs: np.ndarray,
                 ys: np.ndarray,
                 water_geom,
                 coast_geom,
                 buildings_geom,
                 track_geom,
                 out_json: Path,
                 in_json: Optional[Path],
                 out_png: Path,
                 meta: Dict,
                 auto_zoom: bool):
        self.I = I_cart
        self.extent = [xs[0], xs[-1], ys[0], ys[-1]]
        self.out_json = out_json
        self.in_json = in_json
        self.out_png = out_png
        self.meta = meta
        self.auto_zoom = auto_zoom
        self.flip_x = False
        self.flip_y = False


        self.sim = Sim2()
        if in_json and in_json.exists():
            try:
                self.sim = Sim2.from_json(json.loads(in_json.read_text()))
                print(f"[INFO] Loaded init: {in_json}")
            except Exception as e:
                print(f"[WARN] Could not load {in_json}: {e}")

        self.water = water_geom
        self.coast = coast_geom
        self.bldg = buildings_geom
        self.track = track_geom

        self.fig = plt.figure(figsize=(12, 8))
        # NEW: start maximized / full-screen (QtAgg)
        mgr = plt.get_current_fig_manager()
        try:
            # Maximize window (recomendado, no cambia el WM)
            mgr.window.showMaximized()
        except Exception:
            try:
                # Full screen real (más agresivo)
                mgr.full_screen_toggle()
            except Exception:
                pass

        try:
            self.fig.canvas.manager.set_window_title("Manual OSM↔RADAR Calibrator (AOI only)")
        except Exception:
            pass

        self.ax = self.fig.add_axes([0.05, 0.10, 0.70, 0.85])
        self.ax_info = self.fig.add_axes([0.78, 0.78, 0.20, 0.18])
        self.ax_info.axis("off")

        # Keep references to widgets (otherwise they can be garbage-collected and stop responding)
        self._buttons = []
        self._textboxes = []

        # Buttons layout
        left = 0.78
        right_w = 0.20
        cols = 4
        gapx = 0.008
        w = (right_w - (cols - 1) * gapx) / cols
        h = 0.055
        gy = 0.010
        y0 = 0.38

        def add_btn(label, x_idx, row, cb):
            axb = self.fig.add_axes([left + x_idx*(w+gapx), y0 + row*(h+gy), w, h])
            b = Button(axb, label)
            b.on_clicked(cb)
            self._buttons.append(b)
            return b
        
        
        add_btn("Rot -10°", 0, 4, lambda e: self.nudge(rot=-10.0))
        add_btn("Rot +10°", 1, 4, lambda e: self.nudge(rot=+10.0))
        add_btn("Tx -10",   2, 4, lambda e: self.nudge(tx=-10.0))
        add_btn("Tx +10",   3, 4, lambda e: self.nudge(tx=+10.0))

        # Row 5: Ty±10 + Flip X/Y (misma fila)
        add_btn("Ty -10", 0, 5, lambda e: self.nudge(ty=-10.0))
        add_btn("Ty +10", 1, 5, lambda e: self.nudge(ty=+10.0))
        add_btn("Flip X", 2, 5, lambda e: self.toggle_flip_x())
        add_btn("Flip Y", 3, 5, lambda e: self.toggle_flip_y())


        add_btn("Rot -1°", 0, 0, lambda e: self.nudge(rot=-1.0))
        add_btn("Rot +1°", 1, 0, lambda e: self.nudge(rot=+1.0))
        add_btn("Rot -0.1°", 2, 0, lambda e: self.nudge(rot=-0.1))
        add_btn("Rot +0.1°", 3, 0, lambda e: self.nudge(rot=+0.1))

        add_btn("Tx -1.0", 0, 1, lambda e: self.nudge(tx=-1.0))
        add_btn("Tx +1.0", 1, 1, lambda e: self.nudge(tx=+1.0))
        add_btn("Tx -0.1", 2, 1, lambda e: self.nudge(tx=-0.1))
        add_btn("Tx +0.1", 3, 1, lambda e: self.nudge(tx=+0.1))

        add_btn("Ty -1.0", 0, 2, lambda e: self.nudge(ty=-1.0))
        add_btn("Ty +1.0", 1, 2, lambda e: self.nudge(ty=+1.0))
        add_btn("Ty -0.1", 2, 2, lambda e: self.nudge(ty=-0.1))
        add_btn("Ty +0.1", 3, 2, lambda e: self.nudge(ty=+0.1))

        add_btn("S ×0.99", 0, 3, lambda e: self.scale_by(0.99))
        add_btn("S ×1.01", 1, 3, lambda e: self.scale_by(1.01))
        add_btn("S ×0.999", 2, 3, lambda e: self.scale_by(0.999))
        add_btn("S ×1.001", 3, 3, lambda e: self.scale_by(1.001))

        # ---------------------------
        # Bottom control panel (NO overlaps)
        # y: 0.02 .. 0.24 reservado para controles
        # ---------------------------
        bw = (right_w - gapx) / 2
        bh = 0.055

        # Row A (0.24): Save / Screenshot
        rowA = 0.30
        btn_save = Button(self.fig.add_axes([left, rowA, bw, bh]), "Save JSON"); btn_save.on_clicked(lambda e: self.save_json()); self._buttons.append(btn_save)
        btn_shot = Button(self.fig.add_axes([left + bw + gapx, rowA, bw, bh]), "Screenshot"); btn_shot.on_clicked(lambda e: self.save_png()); self._buttons.append(btn_shot)

        # Row B (0.18): Reset / Load
        rowB = 0.24
        btn_reset = Button(self.fig.add_axes([left, rowB, bw, bh]), "Reset"); btn_reset.on_clicked(lambda e: self.reset()); self._buttons.append(btn_reset)
        btn_load = Button(self.fig.add_axes([left + bw + gapx, rowB, bw, bh]), "Load JSON"); btn_load.on_clicked(lambda e: self.load_json()); self._buttons.append(btn_load)

        # Row C (0.12): Apply
        rowC = 0.15
        btn_apply = Button(self.fig.add_axes([left, rowC, right_w, bh]), "Apply Tx/Ty/Rot/s"); btn_apply.on_clicked(lambda e: self.apply_textboxes()); self._buttons.append(btn_apply)

        # Row D (0.02..0.10): Textboxes (2x2)
        tb_h = 0.07
        ax_rt = self.fig.add_axes([left,               0.07, bw, tb_h])
        ax_sc = self.fig.add_axes([left + bw + gapx,   0.07, bw, tb_h])
        ax_tx = self.fig.add_axes([left,               0.02, bw, tb_h])
        ax_ty = self.fig.add_axes([left + bw + gapx,   0.02, bw, tb_h])

        self.tb_rt = TextBox(ax_rt, "Rot", initial=f"{self.sim.rot_deg:.3f}"); self._textboxes.append(self.tb_rt)
        self.tb_sc = TextBox(ax_sc, "s",   initial=f"{self.sim.s:.6f}"); self._textboxes.append(self.tb_sc)
        self.tb_tx = TextBox(ax_tx, "Tx",  initial=f"{self.sim.tx:.3f}"); self._textboxes.append(self.tb_tx)
        self.tb_ty = TextBox(ax_ty, "Ty",  initial=f"{self.sim.ty:.3f}"); self._textboxes.append(self.tb_ty)

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.draw()

    def _auto_limits(self):
        x0, x1, y0, y1 = self.extent[0], self.extent[1], self.extent[2], self.extent[3]
        radar_view = max(abs(x0), abs(x1), abs(y0), abs(y1))
        if not self.auto_zoom:
            return (-radar_view, radar_view), (-radar_view, radar_view)
        view = radar_view
        return (-view, view), (-view, view)

    def draw(self):
        self.ax.cla()
        self.ax.imshow(self.I, extent=self.extent, origin="lower", cmap="gray", interpolation="nearest", alpha=0.95)
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.25)
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.ax.set_title("RADAR (background) + OSM overlay (water/buildings/coastline)")

        water_t = apply_sim2_flip_geom(self.water, self.sim, self.flip_x, self.flip_y) if self.water is not None else None
        coast_t = apply_sim2_flip_geom(self.coast, self.sim, self.flip_x, self.flip_y) if self.coast is not None else None
        bldg_t  = apply_sim2_flip_geom(self.bldg,  self.sim, self.flip_x, self.flip_y) if self.bldg  is not None else None
        trk_t   = apply_sim2_flip_geom(self.track, self.sim, self.flip_x, self.flip_y) if self.track is not None else None

        # Plot order
        if water_t is not None and (not water_t.is_empty):
            gpd.GeoSeries([water_t]).plot(ax=self.ax, color="#7EC8E3", alpha=0.35, edgecolor="none", zorder=2)
        if bldg_t is not None and (not bldg_t.is_empty):
            gpd.GeoSeries([bldg_t]).plot(ax=self.ax, color="#FFD54A", alpha=0.45, edgecolor="red", linewidth=0.6, zorder=3)
        if coast_t is not None and (not coast_t.is_empty):
            gpd.GeoSeries([coast_t]).plot(ax=self.ax, color="cyan", linewidth=1.2, zorder=4)
        if trk_t is not None and (not trk_t.is_empty):
            gpd.GeoSeries([trk_t]).plot(ax=self.ax, color="magenta", linewidth=1.0, alpha=0.8, zorder=5)

        xlim, ylim = self._auto_limits()
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)

        self.ax_info.clear()
        self.ax_info.axis("off")
        hud = (
            f"s        = {self.sim.s:.6f}\n"
            f"rot_deg  = {self.sim.rot_deg:.3f}\n"
            f"tx [m]   = {self.sim.tx:.3f}\n"
            f"ty [m]   = {self.sim.ty:.3f}\n"
            f"flip_x   = {self.flip_x}\n"
            f"flip_y   = {self.flip_y}\n\n"

            "Keys:\n"
            "  ←/→ rot ±0.1°\n"
            "  ↑/↓ scale ×1.001/×0.999\n"
            "  W/A/S/D move 0.1m\n"
            "  R reset, L load, K save, P png, Q quit\n"
        )
        self.ax_info.text(0.0, 1.0, hud, va="top", family="monospace", fontsize=10)
        # if hasattr(self, "tb_tx"):
        #     self.tb_tx.set_val(f"{self.sim.tx:.3f}")
        #     self.tb_ty.set_val(f"{self.sim.ty:.3f}")
        #     self.tb_rt.set_val(f"{self.sim.rot_deg:.3f}")
        #     self.tb_sc.set_val(f"{self.sim.s:.6f}")

        self.fig.canvas.draw_idle()

    def nudge(self, rot=0.0, tx=0.0, ty=0.0):
        self.sim.rot_deg += rot
        self.sim.tx += tx
        self.sim.ty += ty
        self.draw()

    def scale_by(self, f: float):
        self.sim.s *= f
        self.draw()

    def reset(self):
        self.sim = Sim2()
        self.flip_x = False
        self.flip_y = False
        self.sync_textboxes()
        self.draw()

    def load_json(self):
        p = self.in_json if (self.in_json and self.in_json.exists()) else self.out_json
        try:
            d = json.loads(p.read_text())
            self.sim = Sim2.from_json(d)
            self.flip_x = bool(d.get("flip_x", False))
            self.flip_y = bool(d.get("flip_y", False))
            print(f"[INFO] Loaded {p}")
        except Exception as e:
            print(f"[WARN] Could not load {p}: {e}")
        self.sync_textboxes()
        self.draw()

    def sync_textboxes(self):
        if not hasattr(self, "tb_tx"):
            return
        # set_val es caro; sólo llamarlo cuando realmente cambia algo “de golpe”
        self.tb_tx.set_val(f"{self.sim.tx:.3f}")
        self.tb_ty.set_val(f"{self.sim.ty:.3f}")
        self.tb_rt.set_val(f"{self.sim.rot_deg:.3f}")
        self.tb_sc.set_val(f"{self.sim.s:.6f}")

    def save_json(self):
        payload = {
            "model": "Sim2+Flip",
            "s": float(self.sim.s),
            "rot_deg": float(self.sim.rot_deg),
            "tx": float(self.sim.tx),
            "ty": float(self.sim.ty),
            "flip_x": bool(self.flip_x),
            "flip_y": bool(self.flip_y),
            "M3x3": self.sim.M3().tolist(),
            "meta": self.meta,
        }

        self.out_json.parent.mkdir(parents=True, exist_ok=True)
        self.out_json.write_text(json.dumps(payload, indent=2))
        print(f"[OK] Saved {self.out_json}")

    def save_png(self):
        self.out_png.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(self.out_png, dpi=200)
        print(f"[OK] Saved {self.out_png}")

    def on_key(self, e):
        k = (e.key or "").lower()
        if k == "left":
            self.nudge(rot=-0.1)
        elif k == "right":
            self.nudge(rot=+0.1)
        elif k == "up":
            self.scale_by(1.001)
        elif k == "down":
            self.scale_by(0.999)
        elif k == "w":
            self.nudge(ty=+0.1)
        elif k == "s":
            self.nudge(ty=-0.1)
        elif k == "a":
            self.nudge(tx=-0.1)
        elif k == "d":
            self.nudge(tx=+0.1)
        elif k == "r":
            self.reset()
        elif k == "l":
            self.load_json()
        elif k == "k":
            self.save_json()
        elif k == "p":
            self.save_png()
        elif k == "q":
            plt.close(self.fig)

    def toggle_flip_x(self):
        self.flip_x = not self.flip_x
        self.draw()

    def toggle_flip_y(self):
        self.flip_y = not self.flip_y
        self.draw()

    def apply_textboxes(self):
        try:
            self.sim.tx = float(self.tb_tx.text)
            self.sim.ty = float(self.tb_ty.text)
            self.sim.rot_deg = float(self.tb_rt.text)
            self.sim.s = float(self.tb_sc.text)
            self.sync_textboxes()
            self.draw()
        except Exception as ex:
            print(f"[WARN] Invalid textbox values: {ex}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radar_png", type=Path, required=True)
    ap.add_argument("--aoi_gpkg", type=Path, required=True)
    ap.add_argument("--out_json", type=Path, default=Path("outputs/T_OSM_to_RADAR.json"))
    ap.add_argument("--in_json", type=Path, default=None)
    ap.add_argument("--out_png", type=Path, default=Path("outputs/osm_radar_overlay.png"))

    ap.add_argument("--H", type=int, default=400, help="expected azimuth bins")
    ap.add_argument("--W", type=int, default=3424, help="expected range bins")
    ap.add_argument("--r_max", type=float, default=600.0)
    ap.add_argument("--cart_res", type=int, default=1024)
    ap.add_argument("--az_cw", action="store_true", help="Assume azimuth increases clockwise (invert azimuth index)")
    
    ap.add_argument("--water_layer", type=str, default="water")
    ap.add_argument("--coast_layer", type=str, default="coastline")
    ap.add_argument("--buildings_layer", type=str, default="buildings")
    ap.add_argument("--origin_layer", type=str, default="track")
    ap.add_argument("--auto_zoom", action="store_true")
    

    args = ap.parse_args()

    if not args.aoi_gpkg.exists():
        raise FileNotFoundError(f"AOI gpkg not found: {args.aoi_gpkg}")

    # Radar
    I_pol = load_radar_polar_u8(args.radar_png)
    if I_pol.shape == (args.W, args.H):
        I_pol = I_pol.T
    elif I_pol.shape != (args.H, args.W):
        print(f"[WARN] Radar polar shape {I_pol.shape}. Expected (H,W)=({args.H},{args.W}) or transposed. Proceeding anyway.")
    I_cart, xs, ys = polar_to_cart(I_pol, r_max=args.r_max, cart_res=args.cart_res, az_cw=args.az_cw)

    # AOI layers list
    layers = None
    try:
        import fiona
        layers = list(fiona.listlayers(str(args.aoi_gpkg)))
    except Exception:
        pass

    g_water = _safe_read(args.aoi_gpkg, args.water_layer, layers)
    g_coast = _safe_read(args.aoi_gpkg, args.coast_layer, layers)
    g_bldg  = _safe_read(args.aoi_gpkg, args.buildings_layer, layers)
    g_track = _safe_read(args.aoi_gpkg, args.origin_layer, layers)
    g_aoi   = _safe_read(args.aoi_gpkg, "aoi", layers)

    # CRS (informational)
    crs = None
    for g in (g_water, g_coast, g_bldg, g_track, g_aoi):
        if g is not None and g.crs is not None:
            crs = g.crs
            break

    # Origin
    origin = first_track_point_xy(g_track)
    if origin is None:
        origin = centroid_xy(g_aoi)
    if origin is None:
        origin = centroid_xy(g_water) or centroid_xy(g_coast) or centroid_xy(g_bldg)
    if origin is None:
        origin = np.array([0.0, 0.0], dtype=np.float64)
        print("[WARN] No origin found; using (0,0). OSM may be off-screen.")

    E0, N0 = float(origin[0]), float(origin[1])
    print(f"[INFO] AOI layers present: {layers}")
    print(f"[INFO] AOI CRS: {crs}")
    print(f"[INFO] Local origin (E0,N0): [{E0:.3f}, {N0:.3f}]")

    # Localize & union for fast plotting
    water_u = union_layer(localize_gdf(g_water, E0, N0)) if g_water is not None else None
    coast_u = union_layer(localize_gdf(g_coast, E0, N0)) if g_coast is not None else None
    bldg_u  = union_layer(localize_gdf(g_bldg,  E0, N0)) if g_bldg  is not None else None
    trk_u   = union_layer(localize_gdf(g_track, E0, N0)) if g_track is not None else None
    
    simp = 1.5  # meters (ajustá 0.5–5.0)
    if water_u is not None: water_u = water_u.simplify(simp, preserve_topology=True)
    if coast_u is not None: coast_u = coast_u.simplify(simp, preserve_topology=True)
    if bldg_u  is not None: bldg_u  = bldg_u.simplify(simp, preserve_topology=True)
    if trk_u   is not None: trk_u   = trk_u.simplify(simp, preserve_topology=False)

    meta = {
        "radar_png": str(args.radar_png),
        "aoi_gpkg": str(args.aoi_gpkg),
        "layers": {"water": args.water_layer, "coast": args.coast_layer, "buildings": args.buildings_layer, "origin": args.origin_layer},
        "crs": str(crs),
        "origin_E0N0": [E0, N0],
        "r_max": float(args.r_max),
        "cart_res": int(args.cart_res),
        "radar_expected_HW": [int(args.H), int(args.W)],
        "radar_actual_shape": [int(I_pol.shape[0]), int(I_pol.shape[1])],
    }

    gui = CalibGUI(
        I_cart=I_cart,
        xs=xs,
        ys=ys,
        water_geom=water_u,
        coast_geom=coast_u,
        buildings_geom=bldg_u,
        track_geom=trk_u,
        out_json=args.out_json,
        in_json=args.in_json,
        out_png=args.out_png,
        meta=meta,
        auto_zoom=args.auto_zoom,
    )

    print("[INFO] GUI ready. Close the window to exit.")
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
