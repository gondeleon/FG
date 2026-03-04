from __future__ import annotations

from dataclasses import dataclass
import numpy as np

try:
    from scipy.ndimage import map_coordinates
    _HAS_SCIPY = True
except Exception:
    map_coordinates = None
    _HAS_SCIPY = False


@dataclass
class PhaseCorrResult:
    dx: float
    dy: float
    yaw_rad: float
    psr: float          # peak-to-sidelobe ratio (confidence proxy)
    peak: float         # peak value of correlation
    corr: np.ndarray    # correlation surface (translation stage)


def _hann2d(h: int, w: int) -> np.ndarray:
    wy = np.hanning(h)
    wx = np.hanning(w)
    return (wy[:, None] * wx[None, :]).astype(np.float32)


def _phase_correlation(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> tuple[np.ndarray, tuple[int, int], float, float]:
    """
    Returns:
      corr: shifted correlation surface
      (dy, dx): integer peak location relative to center
      psr, peak
    """
    a = a.astype(np.float32)
    b = b.astype(np.float32)

    # FFT
    A = np.fft.fft2(a)
    B = np.fft.fft2(b)
    R = A * np.conj(B)
    R /= (np.abs(R) + eps)
    corr = np.fft.ifft2(R).real

    # Shift zero-lag to center
    corr = np.fft.fftshift(corr)

    # Peak
    peak_idx = np.unravel_index(np.argmax(corr), corr.shape)
    cy, cx = corr.shape[0] // 2, corr.shape[1] // 2
    dy = int(peak_idx[0] - cy)
    dx = int(peak_idx[1] - cx)
    peak = float(corr[peak_idx])

    # PSR: peak / std of sidelobe region excluding a small window around peak
    mask = np.ones_like(corr, dtype=bool)
    r = 7
    y0, y1 = max(0, peak_idx[0]-r), min(corr.shape[0], peak_idx[0]+r+1)
    x0, x1 = max(0, peak_idx[1]-r), min(corr.shape[1], peak_idx[1]+r+1)
    mask[y0:y1, x0:x1] = False
    sidelobe = corr[mask]
    psr = peak / (float(np.std(sidelobe)) + 1e-9)

    return corr, (dy, dx), psr, peak


def _log_polar_sample(mag: np.ndarray, n_rho: int = 256, n_ang: int = 360) -> np.ndarray:
    """
    Magnitude spectrum -> log-polar image.
    Requires SciPy. If not available, raises.
    """
    if not _HAS_SCIPY:
        raise RuntimeError("SciPy is required for log-polar sampling (scipy.ndimage.map_coordinates).")

    h, w = mag.shape
    cy, cx = (h - 1) * 0.5, (w - 1) * 0.5

    # radial limits
    r_max = min(cy, cx)
    rho = np.linspace(0.0, np.log(r_max + 1e-6), n_rho, dtype=np.float32)  # log radius
    ang = np.linspace(0.0, 2*np.pi, n_ang, endpoint=False, dtype=np.float32)

    rr = np.exp(rho)[:, None]
    aa = ang[None, :]
    yy = cy + rr * np.sin(aa)
    xx = cx + rr * np.cos(aa)

    coords = np.vstack([yy.ravel(), xx.ravel()])
    out = map_coordinates(mag.astype(np.float32), coords, order=1, mode="reflect").reshape(n_rho, n_ang)
    return out


def estimate_yaw_xy_phasecorr(
    I_ref: np.ndarray,
    I_cur: np.ndarray,
    px_size_m: float,
    window: bool = True,
    n_rho: int = 256,
    n_ang: int = 360,
) -> PhaseCorrResult:
    """
    1) Rotation from log-polar phase correlation on Fourier magnitude
    2) Translation from phase correlation after rotating cur

    Conventions:
      - dx,dy in meters, in image x/y axes (x right, y down). Convert to ENU later if needed.
    """
    I0 = I_ref.astype(np.float32)
    I1 = I_cur.astype(np.float32)
    if window:
        W = _hann2d(I0.shape[0], I0.shape[1])
        I0 = I0 * W
        I1 = I1 * W

    # --- Rotation via log-polar on FFT magnitude ---
    F0 = np.fft.fftshift(np.fft.fft2(I0))
    F1 = np.fft.fftshift(np.fft.fft2(I1))
    M0 = np.log1p(np.abs(F0)).astype(np.float32)
    M1 = np.log1p(np.abs(F1)).astype(np.float32)

    LP0 = _log_polar_sample(M0, n_rho=n_rho, n_ang=n_ang)
    LP1 = _log_polar_sample(M1, n_rho=n_rho, n_ang=n_ang)

    corr_rot, (d_ang, _), psr_rot, peak_rot = _phase_correlation(LP0, LP1)

    # d_ang is shift in "rows" (rho) or "cols" (ang) depending on implementation:
    # Here LP is (rho, ang), so yaw shift is along ang dimension => dx in correlation.
    # Our _phase_correlation returns (dy, dx) relative to center. We need dx (ang shift).
    _, (dy_lp, dx_lp), psr_rot, peak_rot = _phase_correlation(LP0, LP1)
    yaw = - (dx_lp * (2*np.pi / n_ang))  # negative due to correlation direction

    # --- Rotate I1 by yaw (nearest/bilinear via FFT shift is heavy; do spatial rotate) ---
    # Use simple rotation with scipy if present, else do a coarse rotation via nearest mapping.
    if _HAS_SCIPY:
        from scipy.ndimage import rotate
        I1r = rotate(I1, np.rad2deg(yaw), reshape=False, order=1, mode="nearest")
    else:
        # crude fallback: no refinement
        I1r = I1

    # --- Translation via phase correlation ---
    corr_xy, (dy, dx), psr_xy, peak_xy = _phase_correlation(I0, I1r)

    dx_m = float(dx) * float(px_size_m)
    dy_m = float(dy) * float(px_size_m)

    # Combine confidences: keep translation corr surface but PSR from translation
    return PhaseCorrResult(
        dx=dx_m,
        dy=dy_m,
        yaw_rad=float(yaw),
        psr=float(psr_xy),
        peak=float(peak_xy),
        corr=corr_xy.astype(np.float32),
    )
