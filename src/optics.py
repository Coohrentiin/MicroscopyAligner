# Copyright (c) 2025 Corentin Soubeiran
# SPDX-License-Identifier: MIT
"""Qt-free optics helpers for the Wavefront Focusing & Alignment tool.

This module is intentionally free of any Qt / GUI imports so it can be unit
tested headlessly (mirroring the Qt-free functions in
``keyPointsDetectionAndSelection.py``).  It provides:

- **Sampling / scale** helpers (sample-plane pixel size, magnification scale,
  depth of field) used to relate two objectives' optics.
- **Angular Spectrum Method (ASM)** free-space propagation of a complex field
  (:func:`propagate_asm`) plus tiny complex<->(phase, amp) converters.
- **Autofocus** metrics + sweep (:func:`autofocus`, Gini / Gouy-phase).
- A **focus-consistency** curve (:func:`focus_consistency_curve`) comparing two
  fields propagated over a z range (NCC / L2 / SSIM).
- Alignment **estimators**: :func:`estimate_scale_translation` (scale +
  translation only, no rotation) and :func:`estimate_distortion` (non-centered
  distortion via thin-plate spline / polynomial / free-center radial /
  piecewise-affine), with :func:`warp_field_with_distortion` to apply a global
  similarity + distortion to a complex field channel-consistently.
- :func:`read_wavefront_wavelengths` to surface the per-Z ``wavelengths_nm``
  written into the ImageJ ``Info`` tag by :func:`utils_images.save_stack`.

Length units are SI **metres** internally; callers convert nm / µm at the UI
boundary.

Conventions
-----------
The wavefront complex field is ``field = amp * exp(i * phase)`` where
``phase = OPD / lambda * 2*pi + delta_phi`` (an optical-path-difference term
plus a residual).  Both are reconstructed by the same ``exp(i*phase)``.

The keypoint matrix convention follows
``keyPointsSelection.estimate_transform_keypoints``: pairs are
``[((tx, ty), (mx, my)), ...]`` with ``pair[0]`` the **template** point and
``pair[1]`` the **moving** point, and the returned 3x3 matrix maps a **moving**
point to its **template** point (``template ~= M @ moving``).  ``ImageAligner``
then warps the moving image with ``transform.inverse`` into template space.
"""
import json

import numpy as np
import tifffile
from skimage import transform as tf
from skimage.restoration import unwrap_phase as _unwrap_phase

try:  # SSIM is optional; degrade gracefully if unavailable.
    from skimage.metrics import structural_similarity as _ssim
except Exception:  # pragma: no cover
    _ssim = None

try:  # Available in recent skimage (>=0.24); fall back to polynomial otherwise.
    from skimage.transform import ThinPlateSplineTransform as _TPS
except Exception:  # pragma: no cover
    _TPS = None

try:  # Optional GPU acceleration for the (batched) ASM propagation.
    import torch as _torch
    _TORCH_DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"
    # Only worth the host<->device transfer when there's a GPU.
    _TORCH_OK = _torch.cuda.is_available()
except Exception:  # pragma: no cover
    _torch = None
    _TORCH_DEVICE = "cpu"
    _TORCH_OK = False


def gpu_available():
    """True when a CUDA torch backend will be used for batched propagation."""
    return bool(_TORCH_OK)


# --------------------------------------------------------------- sampling/scale
def sample_pixel_size(camera_pixel_size_m, magnification):
    """Sample-plane pixel size = camera pixel size / magnification (metres)."""
    return float(camera_pixel_size_m) / float(magnification)


def magnification_scale(mag_template, mag_moving):
    """Resize factor bringing the MOVING sampling onto the TEMPLATE sampling.

    Sample pixel size scales as ``1 / magnification``: a higher-mag objective
    has a *smaller* sample-plane pixel, so a feature spans more pixels there.
    A 20x template vs 10x moving therefore needs the moving image **enlarged**
    by ``mag_template / mag_moving = 2`` (then center-cropped to the template
    shape, showing the central FOV).
    """
    return float(mag_template) / float(mag_moving)


def depth_of_field(wavelength_m, na, n=1.0):
    """Wave-optical depth of field ``n * lambda / NA**2`` (metres)."""
    na = float(na)
    if na <= 0:
        return float("inf")
    return float(n) * float(wavelength_m) / (na * na)


def suggest_focus_range(wavelength_m, na, n=1.0, span_factor=5.0, min_step_m=50e-9):
    """Suggest ``(half_range_m, step_m)`` for an autofocus sweep.

    ``step ~ DOF / 4`` clamped to ``>= min_step_m``; ``half_range ~
    span_factor * DOF``.  Used only to seed the UI; the focus-consistency check
    uses the user-set range.
    """
    dof = depth_of_field(wavelength_m, na, n)
    if not np.isfinite(dof):
        return 5e-6, max(min_step_m, 100e-9)
    step = max(min_step_m, dof / 4.0)
    half = span_factor * dof
    return float(half), float(step)


def focus_z_values(half_range_m, step_m):
    """Symmetric z sweep ``arange(-half, +half + step, step)`` (metres)."""
    half = abs(float(half_range_m))
    step = abs(float(step_m))
    if step <= 0:
        return np.array([0.0])
    return np.arange(-half, half + step, step)


# --------------------------------------------------------------- field helpers
def field_from_phase_amp(phase, amp):
    """Build a complex field ``amp * exp(i * phase)`` (phase in radians)."""
    phase = np.asarray(phase, dtype=np.float64)
    amp = np.asarray(amp, dtype=np.float64)
    return amp * np.exp(1j * phase)


def center_crop(arr, frac):
    """Return the centered ``frac`` (0<frac<=1) sub-array of ``arr`` (H, W).

    Used to run the focus search on a fast central ROI; the chosen z is then
    applied to the full frame. ``frac >= 1`` returns the array unchanged.
    """
    arr = np.asarray(arr)
    if frac is None or frac >= 1.0:
        return arr
    H, W = arr.shape[:2]
    h = max(1, int(round(H * frac))); w = max(1, int(round(W * frac)))
    y0 = (H - h) // 2; x0 = (W - w) // 2
    return arr[y0:y0 + h, x0:x0 + w]


def load_field(path, frame_index=0):
    """Load a wavefront TIFF frame as a complex field ``amp*exp(i*phase)``.

    Thin wrapper over :func:`utils_images.load_wavefront_tif` so callers (the
    focusing tool, batch mode's propagation map) get a ready complex field.
    Returns ``(field, n_frames)``.
    """
    from utils_images import load_wavefront_tif  # local import: avoid cycle at module load
    phase, amp, n_frames = load_wavefront_tif(path, frame_index=frame_index)
    return field_from_phase_amp(phase, amp), int(n_frames)


def unwrapped_phase(field, subtract_median=False):
    """Spatially-unwrapped phase (radians, float32) of a complex field.

    Uses :func:`skimage.restoration.unwrap_phase` so the phase image is
    continuous (no 2π/±π seams) -- required for display and for any
    phase-based feature detection / correlation. NaNs (e.g. from out-of-hull
    warps) are zeroed before unwrapping. When ``subtract_median`` is set the
    median of the unwrapped phase is removed (``phase - median(phase)``), which
    centers the wavefront and removes a global piston offset.
    """
    wrapped = np.angle(np.asarray(field))
    wrapped = np.nan_to_num(wrapped, nan=0.0, posinf=0.0, neginf=0.0)
    phase = _unwrap_phase(wrapped).astype(np.float32)
    if subtract_median:
        phase = phase - np.float32(np.median(phase))
    return phase


def phase_amp_from_field(field, unwrap=False, subtract_median=False):
    """Return ``(phase float32, amp float32)`` of a complex field.

    By default the phase is the raw (wrapped) ``np.angle``; pass
    ``unwrap=True`` to spatially unwrap it via :func:`unwrapped_phase`
    (optionally with ``subtract_median``).
    """
    field = np.asarray(field)
    if unwrap:
        phase = unwrapped_phase(field, subtract_median=subtract_median)
    else:
        phase = np.angle(field).astype(np.float32)
    return phase, np.abs(field).astype(np.float32)


# ----------------------------------------------------------------- ASM
def propagate_asm(field, z, wavelength, pixel_size, n=1.0, band_limit=True):
    """Angular Spectrum Method free-space propagation of a complex field.

    Parameters
    ----------
    field : complex ndarray (H, W)
        Complex field ``amp * exp(i*phase)``.
    z : float
        Signed propagation distance (metres).
    wavelength : float
        Vacuum wavelength (metres); medium wavelength is ``wavelength / n``.
    pixel_size : float
        Sample-plane pixel size (metres).
    n : float
        Refractive index of the propagation medium.
    band_limit : bool
        Apply Matsushima band-limiting to suppress aliasing for large ``|z|``.

    Returns
    -------
    complex ndarray (H, W) — the propagated field.

    Transfer function (unshifted FFT order)::

        H = exp( i * (2*pi/lam) * z * sqrt(1 - (lam*fx)^2 - (lam*fy)^2) )

    with evanescent components (negative root argument) set to zero.
    """
    field = np.asarray(field, dtype=np.complex128)
    if z == 0:
        return field.copy()

    lam = float(wavelength) / float(n)
    k = 2.0 * np.pi / lam
    if _TORCH_OK:
        # GPU single propagation (matches the batched/numpy math to ~1e-15).
        dev = _TORCH_DEVICE
        H, W = field.shape
        fx = _torch.fft.fftfreq(W, d=float(pixel_size), device=dev, dtype=_torch.float64)
        fy = _torch.fft.fftfreq(H, d=float(pixel_size), device=dev, dtype=_torch.float64)
        FY, FX = _torch.meshgrid(fy, fx, indexing="ij")
        arg = 1.0 - (lam * FX) ** 2 - (lam * FY) ** 2
        prop = arg >= 0.0
        Hf = _torch.where(prop, _torch.exp(1j * (k * float(z)) * _torch.sqrt(_torch.clamp(arg, min=0.0))),
                          _torch.zeros((), dtype=_torch.complex128, device=dev))
        if band_limit:
            dfx = 1.0 / (W * float(pixel_size)); dfy = 1.0 / (H * float(pixel_size))
            fx_max = 1.0 / (lam * _torch.sqrt(_torch.tensor((2.0 * dfx * z) ** 2 + 1.0, device=dev)))
            fy_max = 1.0 / (lam * _torch.sqrt(_torch.tensor((2.0 * dfy * z) ** 2 + 1.0, device=dev)))
            Hf = _torch.where((FX.abs() <= fx_max) & (FY.abs() <= fy_max), Hf,
                              _torch.zeros((), dtype=_torch.complex128, device=dev))
        spec = _torch.fft.fft2(_torch.as_tensor(field, device=dev, dtype=_torch.complex128))
        return _torch.fft.ifft2(spec * Hf).cpu().numpy()

    FX, FY = _freq_grids(field.shape, pixel_size)
    Hf = _asm_transfer(FX, FY, float(z), lam, k, pixel_size, band_limit)
    return np.fft.ifft2(np.fft.fft2(field) * Hf)


class Propagator:
    """Cached ASM propagator for a FIXED field over varying z (GPU when available).

    Precomputes the field's FFT and the frequency grids once; each
    :meth:`at` only builds the transfer function and runs one inverse FFT. This
    is the hot path for the interactive distance slider / live overlay, where
    the field is constant and only z changes. ``band_limit`` defaults off (the
    slider's z range is small).
    """

    def __init__(self, field, wavelength, pixel_size, n=1.0, band_limit=False):
        self.lam = float(wavelength) / float(n)
        self.k = 2.0 * np.pi / self.lam
        self.pixel_size = float(pixel_size)
        self.band_limit = band_limit
        self.shape = np.asarray(field).shape
        if _TORCH_OK:
            dev = _TORCH_DEVICE
            H, W = self.shape
            fx = _torch.fft.fftfreq(W, d=self.pixel_size, device=dev, dtype=_torch.float64)
            fy = _torch.fft.fftfreq(H, d=self.pixel_size, device=dev, dtype=_torch.float64)
            self._FY, self._FX = _torch.meshgrid(fy, fx, indexing="ij")
            self._arg = 1.0 - (self.lam * self._FX) ** 2 - (self.lam * self._FY) ** 2
            self._root = _torch.sqrt(_torch.clamp(self._arg, min=0.0))
            self._prop = self._arg >= 0.0
            self._spec = _torch.fft.fft2(_torch.as_tensor(field, device=dev, dtype=_torch.complex128))
        else:
            self._FX, self._FY = _freq_grids(self.shape, self.pixel_size)
            self._spec = np.fft.fft2(np.asarray(field, dtype=np.complex128))

    def at(self, z):
        """Return the field propagated to distance ``z`` (metres), as numpy."""
        z = float(z)
        if z == 0:
            if _TORCH_OK:
                return _torch.fft.ifft2(self._spec).cpu().numpy()
            return np.fft.ifft2(self._spec)
        if _TORCH_OK:
            dev = _TORCH_DEVICE
            Hf = _torch.where(self._prop,
                              _torch.exp(1j * (self.k * z) * self._root),
                              _torch.zeros((), dtype=_torch.complex128, device=dev))
            if self.band_limit:
                H, W = self.shape
                dfx = 1.0 / (W * self.pixel_size); dfy = 1.0 / (H * self.pixel_size)
                fx_max = 1.0 / (self.lam * _torch.sqrt(_torch.tensor((2.0 * dfx * z) ** 2 + 1.0, device=dev)))
                fy_max = 1.0 / (self.lam * _torch.sqrt(_torch.tensor((2.0 * dfy * z) ** 2 + 1.0, device=dev)))
                Hf = _torch.where((self._FX.abs() <= fx_max) & (self._FY.abs() <= fy_max), Hf,
                                  _torch.zeros((), dtype=_torch.complex128, device=dev))
            return _torch.fft.ifft2(self._spec * Hf).cpu().numpy()
        Hf = _asm_transfer(self._FX, self._FY, z, self.lam, self.k, self.pixel_size, self.band_limit)
        return np.fft.ifft2(self._spec * Hf)


def _freq_grids(shape, pixel_size):
    """``(FX, FY)`` spatial-frequency meshgrids (cycles/m, unshifted order)."""
    H, W = shape
    fx = np.fft.fftfreq(W, d=float(pixel_size))
    fy = np.fft.fftfreq(H, d=float(pixel_size))
    return np.meshgrid(fx, fy)


def _asm_transfer(FX, FY, z, lam, k, pixel_size, band_limit):
    """ASM transfer function ``H`` for a single distance ``z`` (numpy)."""
    arg = 1.0 - (lam * FX) ** 2 - (lam * FY) ** 2
    prop = arg >= 0.0
    Hf = np.where(prop, np.exp(1j * k * z * np.sqrt(np.maximum(arg, 0.0))), 0.0)
    if band_limit:
        # Matsushima & Shimobaba (2009): limit the local fringe frequency so the
        # transfer function is sampled correctly over the (finite) aperture.
        H, W = FX.shape
        dfx = 1.0 / (W * float(pixel_size))
        dfy = 1.0 / (H * float(pixel_size))
        fx_max = 1.0 / (lam * np.sqrt((2.0 * dfx * z) ** 2 + 1.0))
        fy_max = 1.0 / (lam * np.sqrt((2.0 * dfy * z) ** 2 + 1.0))
        Hf = np.where((np.abs(FX) <= fx_max) & (np.abs(FY) <= fy_max), Hf, 0.0)
    return Hf


def propagate_asm_stack(field, z_values, wavelength, pixel_size, n=1.0,
                        band_limit=False):
    """Propagate one complex field to many distances; return a complex stack.

    Returns ``(len(z_values), H, W)``.  Uses a **batched FFT on the GPU** when
    torch+CUDA are available (a single ``fft2`` of the field is reused across
    all z, the transfer functions stacked along a new axis), otherwise loops
    :func:`propagate_asm` in numpy.  This is the hot path for autofocus and the
    focus-consistency map.
    """
    field = np.asarray(field, dtype=np.complex128)
    z_values = np.asarray(z_values, dtype=np.float64)
    lam = float(wavelength) / float(n)
    k = 2.0 * np.pi / lam
    if _TORCH_OK:
        dev = _TORCH_DEVICE
        H, W = field.shape
        fx = _torch.fft.fftfreq(W, d=float(pixel_size), device=dev, dtype=_torch.float64)
        fy = _torch.fft.fftfreq(H, d=float(pixel_size), device=dev, dtype=_torch.float64)
        FY, FX = _torch.meshgrid(fy, fx, indexing="ij")
        arg = 1.0 - (lam * FX) ** 2 - (lam * FY) ** 2
        prop = arg >= 0.0
        root = _torch.sqrt(_torch.clamp(arg, min=0.0))               # (H, W)
        z = _torch.as_tensor(z_values, device=dev, dtype=_torch.float64)[:, None, None]
        phase = k * z * root[None]                                   # (Z, H, W)
        Hf = _torch.where(prop[None], _torch.exp(1j * phase),
                          _torch.zeros((), dtype=_torch.complex128, device=dev))
        if band_limit:
            dfx = 1.0 / (W * float(pixel_size)); dfy = 1.0 / (H * float(pixel_size))
            zabs = z.abs()
            fx_max = 1.0 / (lam * _torch.sqrt((2.0 * dfx * zabs) ** 2 + 1.0))
            fy_max = 1.0 / (lam * _torch.sqrt((2.0 * dfy * zabs) ** 2 + 1.0))
            band = (FX.abs()[None] <= fx_max) & (FY.abs()[None] <= fy_max)
            Hf = _torch.where(band, Hf, _torch.zeros((), dtype=_torch.complex128, device=dev))
        # z==0 is identity (no propagation), matching propagate_asm's short-circuit:
        # force the transfer function to all-ones for any zero plane.
        if _torch.any(z == 0):
            ones = _torch.ones((), dtype=_torch.complex128, device=dev)
            Hf = _torch.where((z == 0), ones, Hf)
        spec = _torch.fft.fft2(_torch.as_tensor(field, device=dev, dtype=_torch.complex128))
        out = _torch.fft.ifft2(spec[None] * Hf)
        return out.cpu().numpy()

    # numpy fallback
    out = np.empty((z_values.size,) + field.shape, dtype=np.complex128)
    FX, FY = _freq_grids(field.shape, pixel_size)
    spec = np.fft.fft2(field)
    for i, z in enumerate(z_values):
        if z == 0:
            out[i] = field
        else:
            Hf = _asm_transfer(FX, FY, float(z), lam, k, pixel_size, band_limit)
            out[i] = np.fft.ifft2(spec * Hf)
    return out


# ----------------------------------------------------------------- autofocus
def gini_index(values):
    """Gini index of a 1-D non-negative array (sparsity; higher = sparser).

    ``G = sum_i (2i - N - 1) * v_(i) / (N * sum(v))`` over the ascending sort.
    Returns 0 for an all-zero / degenerate input.
    """
    v = np.sort(np.asarray(values, dtype=np.float64).ravel())
    v = np.clip(v, 0.0, None)
    n = v.size
    total = v.sum()
    if n == 0 or total <= 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float(np.sum((2 * idx - n - 1) * v) / (n * total))


def _roi_crop(arr, roi):
    if roi is None:
        return arr
    y0, y1, x0, x1 = roi
    return arr[y0:y1, x0:x1]


def focus_metric_gini(field, roi=None):
    """Gini index of the amplitude (maximize -> focused, sparse amplitude)."""
    a = np.abs(_roi_crop(np.asarray(field), roi))
    a = a - a.min()
    return gini_index(a.ravel())


def focus_metric_gouy(field, roi=None):
    """Gouy-phase focus proxy: variance of the phase, negated (maximize).

    Near focus the wavefront is flattest; minimizing phase variance (i.e.
    maximizing ``-var``) locates that plane. The phase is spatially unwrapped
    first so the variance reflects the actual wavefront shape, not ±π seams.
    """
    p = unwrapped_phase(_roi_crop(np.asarray(field), roi))
    return float(-np.var(p))


_FOCUS_METRICS = {"gini": focus_metric_gini, "gouy": focus_metric_gouy}


def autofocus(field, z_values, wavelength, pixel_size, n=1.0, method="gini",
              roi=None, post=None):
    """Sweep ``z_values``, propagate, score, and return ``(best_z, scores, z_values)``.

    The full field is propagated each step on its native grid (so no padded
    edge is ever propagated); an optional ``post`` callable is applied to the
    propagated field (e.g. crop to the template shape) **before** the ``roi``
    crop, so the ROI matches what the user sees.  Only the ``roi`` crop is
    scored so FFT wrap-around at the borders does not pollute the metric.
    ``method`` in ``{"gini" (amplitude), "gouy" (phase)}``; both are maximized.
    """
    metric = _FOCUS_METRICS.get(method, focus_metric_gini)
    z_values = np.asarray(z_values, dtype=np.float64)
    stack = propagate_asm_stack(field, z_values, wavelength, pixel_size, n=n)
    scores = np.empty(z_values.shape, dtype=np.float64)
    for i in range(z_values.size):
        prop = stack[i]
        if post is not None:
            prop = post(prop)
        scores[i] = metric(prop, roi=roi)
    best_z = float(z_values[int(np.argmax(scores))]) if z_values.size else 0.0
    return best_z, scores, z_values


def autofocus_gini(field, z_values, wavelength, pixel_size, n=1.0, roi=None):
    return autofocus(field, z_values, wavelength, pixel_size, n=n, method="gini", roi=roi)


def autofocus_gouy(field, z_values, wavelength, pixel_size, n=1.0, roi=None):
    return autofocus(field, z_values, wavelength, pixel_size, n=n, method="gouy", roi=roi)


# ------------------------------------------------------- focus consistency
def _norm(a):
    a = np.asarray(a, dtype=np.float64)
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo + 1e-12)


def _similarity(a, b, metric):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.size == 0:
        return float("nan")
    if metric == "l2":
        return float(-np.mean((_norm(a) - _norm(b)) ** 2))
    if metric == "ssim":
        if _ssim is None:
            return float("nan")
        na, nb = _norm(a), _norm(b)
        return float(_ssim(na, nb, data_range=1.0))
    # default: normalized cross-correlation (Pearson)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def focus_consistency_curve(field_a, field_b, z_values,
                            wavelength_a, wavelength_b,
                            pixel_size_a, pixel_size_b,
                            n=1.0, metric="ncc", roi=None):
    """Similarity between two fields propagated over ``z_values``.

    For each z both fields are propagated by that distance, their amplitudes
    cropped to ``roi`` and compared with ``metric`` in ``{"ncc", "l2",
    "ssim"}`` (all returned "higher = better").  A single sharp peak near z=0
    confirms the two fields are co-focused and aligned.

    Returns ``(z_values, metric_array)``.
    """
    z_values = np.asarray(z_values, dtype=np.float64)
    out = np.empty(z_values.shape, dtype=np.float64)
    for i, z in enumerate(z_values):
        pa = np.abs(propagate_asm(field_a, z, wavelength_a, pixel_size_a, n=n))
        pb = np.abs(propagate_asm(field_b, z, wavelength_b, pixel_size_b, n=n))
        out[i] = _similarity(_roi_crop(pa, roi), _roi_crop(pb, roi), metric)
    return z_values, out


def _overlap_bbox(mask, border=0):
    """Bounding box (y0, y1, x0, x1) of ``mask`` shrunk by ``border`` px each side.

    Returns ``None`` if nothing is left. Used to restrict the focus-check metric
    to the common valid region and avoid warp/propagation edge artifacts.
    """
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    y0, y1 = int(ys.min()) + border, int(ys.max()) + 1 - border
    x0, x1 = int(xs.min()) + border, int(xs.max()) + 1 - border
    if y1 - y0 < 1 or x1 - x0 < 1:
        return None
    return (y0, y1, x0, x1)


def focus_consistency_map(field_a, field_b, z_a_values, z_b_values,
                          wavelength_a, wavelength_b,
                          pixel_size_a, pixel_size_b,
                          n=1.0, metric="ncc", roi=None, fit_b=None,
                          align_b=None, border=16, progress=None):
    """2-D focus-consistency map over (z_a, z_b), alignment- & overlap-aware.

    Propagates ``field_a`` (template) over ``z_a_values`` and ``field_b``
    (moving) over ``z_b_values`` (each stack computed once, GPU-batched), then
    scores every pair with ``metric`` in ``{"ncc","l2","ssim"}`` (amplitude).

    - ``align_b(complex_frame) -> complex_frame``: applied to each propagated
      moving frame to bring it into the template frame using any prior
      alignment (transform + distortion). Supersedes ``fit_b`` when given.
    - The comparison is restricted to the **common non-zero overlap** of the
      template and the aligned moving frame, with the border eroded by
      ``border`` px (default 16) to drop warp/FFT edge artifacts. ``roi``, when
      given, further restricts the region.
    - ``progress(done, total)``: optional callback for a progress bar.

    Returns ``(z_a_values, z_b_values, map2d)`` with ``map2d`` shape
    ``(len(z_b_values), len(z_a_values))`` (rows = z_b, cols = z_a). Cells with
    too little overlap are NaN.
    """
    z_a_values = np.asarray(z_a_values, dtype=np.float64)
    z_b_values = np.asarray(z_b_values, dtype=np.float64)
    transform_b = align_b if align_b is not None else fit_b

    stack_a = propagate_asm_stack(field_a, z_a_values, wavelength_a, pixel_size_a, n=n)
    stack_b = propagate_asm_stack(field_b, z_b_values, wavelength_b, pixel_size_b, n=n)

    amps_a = [np.abs(stack_a[i]) for i in range(z_a_values.size)]
    amps_b = []
    for j in range(z_b_values.size):
        fb = stack_b[j]
        if transform_b is not None:
            fb = transform_b(fb)
        amps_b.append(np.abs(fb))

    # Common valid region: where the template and the aligned moving are both
    # non-zero (the warp zero-fills outside the moving's mapped support). The
    # alignment is z-independent so one representative aligned frame defines it.
    tpl_valid = amps_a[len(amps_a) // 2] > 0
    mov_valid = amps_b[len(amps_b) // 2] > 0
    valid = tpl_valid & mov_valid
    bbox = _overlap_bbox(valid, border=border)
    if bbox is None:
        bbox = (0, amps_a[0].shape[0], 0, amps_a[0].shape[1])
    # Intersect with an explicit ROI if provided.
    if roi is not None:
        ry0, ry1, rx0, rx1 = roi
        by0, by1, bx0, bx1 = bbox
        bbox = (max(by0, ry0), min(by1, ry1), max(bx0, rx0), min(bx1, rx1))

    y0, y1, x0, x1 = bbox
    amps_a = [a[y0:y1, x0:x1] for a in amps_a]
    amps_b = [b[y0:y1, x0:x1] for b in amps_b]

    total = z_b_values.size * z_a_values.size
    out = np.empty((z_b_values.size, z_a_values.size), dtype=np.float64)
    done = 0
    for j in range(z_b_values.size):
        for i in range(z_a_values.size):
            out[j, i] = _similarity(amps_a[i], amps_b[j], metric)
            done += 1
        if progress is not None:
            progress(done, total)
    return z_a_values, z_b_values, out


def optimal_indices(map2d, mode="global", ref_col=None, ref_row=None):
    """Indices ``(i_col=z_a, j_row=z_b)`` of the optimum (max) of ``map2d``.

    ``mode``:
    - ``"global"``   : argmax over the whole map.
    - ``"vs_template"``: best moving z at the template column ``ref_col`` (fix
      template focus -> search the vertical line).
    - ``"vs_moving"``  : best template z at the moving row ``ref_row`` (fix
      moving focus -> search the horizontal line).

    Returns ``(i_col, j_row)`` or ``None`` if all-NaN.
    """
    if not np.any(np.isfinite(map2d)):
        return None
    nrow, ncol = map2d.shape
    if mode == "vs_template" and ref_col is not None:
        col = np.clip(int(ref_col), 0, ncol - 1)
        line = map2d[:, col]
        if not np.any(np.isfinite(line)):
            return None
        return col, int(np.nanargmax(line))
    if mode == "vs_moving" and ref_row is not None:
        row = np.clip(int(ref_row), 0, nrow - 1)
        line = map2d[row, :]
        if not np.any(np.isfinite(line)):
            return None
        return int(np.nanargmax(line)), row
    j, i = np.unravel_index(np.nanargmax(map2d), map2d.shape)
    return int(i), int(j)


# ------------------------------------------------------------ metadata reader
def read_wavefront_wavelengths(path):
    """Return the per-Z ``wavelengths_nm`` list from an ImageJ TIFF, or ``None``.

    :func:`utils_images.save_stack` stores acquisition metadata as JSON in the
    ImageJ ``Info`` tag; this reads it back so the UI can label frames.
    """
    try:
        with tifffile.TiffFile(path) as tif:
            meta = tif.imagej_metadata or {}
        info = meta.get("Info")
        if not info:
            return None
        parsed = json.loads(info)
        wl = parsed.get("wavelengths_nm")
        if wl:
            return [float(x) for x in wl]
    except Exception:
        return None
    return None


# ------------------------------------------------------------ alignment math
def _pairs_to_arrays(pairs):
    """Split keypoint pairs into ``(template_pts, moving_pts)`` (N, 2) arrays.

    Matches :func:`keyPointsSelection.estimate_transform_keypoints`: ``pair[0]``
    is the template point, ``pair[1]`` the moving point.
    """
    template_pts = np.asarray([p[0] for p in pairs], dtype=np.float64)
    moving_pts = np.asarray([p[1] for p in pairs], dtype=np.float64)
    return template_pts, moving_pts


def estimate_scale_translation(pairs):
    """Isotropic scale + translation (NO rotation) from keypoint pairs.

    Returns a 3x3 matrix mapping a **moving** point to its **template** point
    (``template ~= M @ moving``), matching the convention of
    :func:`keyPointsSelection.estimate_transform_keypoints`.

    Closed form with moving ``M`` and template ``T``::

        s = sum<T - Tbar, M - Mbar> / sum||M - Mbar||^2
        t = Tbar - s * Mbar
    """
    if not pairs:
        raise ValueError("At least one point pair is required")
    template_pts, moving_pts = _pairs_to_arrays(pairs)

    matrix = np.eye(3)
    if len(pairs) == 1:
        # Translation only.
        translation = (template_pts - moving_pts)[0]
        matrix[0, 2], matrix[1, 2] = translation[0], translation[1]
        return matrix

    t_bar = template_pts.mean(axis=0)
    m_bar = moving_pts.mean(axis=0)
    t_c = template_pts - t_bar
    m_c = moving_pts - m_bar

    denom = float(np.sum(m_c * m_c))
    s = float(np.sum(t_c * m_c) / denom) if denom > 1e-12 else 1.0
    t = t_bar - s * m_bar

    matrix[0, 0] = s
    matrix[1, 1] = s
    matrix[0, 2] = t[0]
    matrix[1, 2] = t[1]
    return matrix


def estimate_distortion(pairs, template_shape, model="tps", order=2,
                        radial_grid=9, residual_matrix=None):
    """Estimate a non-rigid distortion warp from keypoint correspondences.

    The returned object maps **template (output) coords -> input coords**, i.e.
    it is suitable as the ``inverse_map`` argument of
    :func:`skimage.transform.warp` (the convention used throughout this app).

    Parameters
    ----------
    pairs : list of ((tx, ty), (mx, my))
        Correspondences (template, moving), ideally the RANSAC inliers.
    template_shape : tuple
        ``(H, W)`` of the template / output, used by the piecewise model.
    model : {"tps", "poly", "radial", "piecewise"}
        ``"tps"`` (default) thin-plate spline; ``"poly"`` polynomial of
        ``order``; ``"radial"`` free-center radial (cx, cy, k1, k2);
        ``"piecewise"`` piecewise-affine (NaN outside the convex hull).
    residual_matrix : 3x3 array, optional
        When given (the linear moving->template fit ``M``), the moving points are
        first mapped by ``M`` into template space, so the returned warp is the
        **residual** template->template remap (the non-rigid part *after* the
        linear transform). Composing ``current_transform.inverse(residual(...))``
        then keeps the linear transform live/adjustable instead of the distortion
        subsuming it. Without it, the warp is the full template->moving map.
    """
    template_pts, moving_pts = _pairs_to_arrays(pairs)
    if len(pairs) < 3:
        raise ValueError("Distortion estimation needs at least 3 point pairs")

    # src = output (template); dst = input. By default the input is the raw
    # moving points (full template->moving map). With residual_matrix, dst is the
    # moving points mapped into template space, giving a template->template
    # residual that composes with the linear transform.
    src = template_pts
    if residual_matrix is not None:
        M = np.asarray(residual_matrix, dtype=float)
        ones = np.ones((len(moving_pts), 1))
        hom = np.hstack([moving_pts, ones])           # (N, 3)
        mapped = (M @ hom.T).T                          # (N, 3)
        dst = mapped[:, :2] / mapped[:, 2:3]
    else:
        dst = moving_pts

    if model == "tps":
        if _TPS is None:
            model = "poly"  # graceful fallback on older skimage
        else:
            tps = _TPS()
            tps.estimate(src, dst)
            return tps

    if model == "poly":
        poly = tf.PolynomialTransform()
        # PolynomialTransform.estimate(src, dst) fits src -> dst.
        poly.estimate(src, dst, order=order)
        return poly

    if model == "piecewise":
        pw = tf.PiecewiseAffineTransform()
        pw.estimate(src, dst)
        return pw

    if model == "radial":
        return _RadialDistortion.fit(src, dst, grid=radial_grid)

    if model == "spherical":
        return _SphericalDistortion.fit(src, dst, grid=radial_grid)

    raise ValueError(f"Unknown distortion model: {model!r}")


class _RadialDistortion:
    """Free-center radial distortion mapping template -> moving.

    ``p_out = c + (p_in - c) * (1 + k1 r^2 + k2 r^4)`` with ``r = ||p_in - c||``.
    Provides a ``__call__(coords)`` compatible with :func:`skimage.transform.warp`.
    """

    def __init__(self, center, k1, k2):
        self.center = np.asarray(center, dtype=np.float64)
        self.k1 = float(k1)
        self.k2 = float(k2)

    def __call__(self, coords):
        coords = np.asarray(coords, dtype=np.float64)
        d = coords - self.center
        r2 = np.sum(d * d, axis=1)
        factor = 1.0 + self.k1 * r2 + self.k2 * r2 * r2
        return self.center + d * factor[:, None]

    @classmethod
    def fit(cls, src, dst, grid=9):
        """Coarse grid search over the center; linear least-squares for k1,k2."""
        src = np.asarray(src, dtype=np.float64)
        dst = np.asarray(dst, dtype=np.float64)
        cx_candidates = np.linspace(src[:, 0].min(), src[:, 0].max(), grid)
        cy_candidates = np.linspace(src[:, 1].min(), src[:, 1].max(), grid)

        best = None
        for cx in cx_candidates:
            for cy in cy_candidates:
                c = np.array([cx, cy])
                d = src - c
                r2 = np.sum(d * d, axis=1)
                # Solve d*(1 + k1 r2 + k2 r4) = (dst - c) for k1, k2:
                #   d*(k1 r2 + k2 r4) = (dst - c) - d
                # stacking the x and y residual equations.
                rhs = (dst - c) - d
                Ax = np.column_stack([d[:, 0] * r2, d[:, 0] * r2 * r2])
                Ay = np.column_stack([d[:, 1] * r2, d[:, 1] * r2 * r2])
                A = np.vstack([Ax, Ay])
                b = np.concatenate([rhs[:, 0], rhs[:, 1]])
                coef, *_ = np.linalg.lstsq(A, b, rcond=None)
                k1, k2 = coef
                model = cls(c, k1, k2)
                pred = model(src)
                err = float(np.mean(np.sum((pred - dst) ** 2, axis=1)))
                if best is None or err < best[0]:
                    best = (err, model)
        return best[1]


class _SphericalDistortion:
    """Free-center spherical curvature mapping template -> moving.

    A spherical (cap) deformation makes the apparent in-plane radial position
    grow with a leading cubic term: ``p_out = c + (p_in - c) * (1 + k * r^2)``
    with ``r = ||p_in - c||`` so the displacement ``|p_out - p_in| ~ k * r^3``.
    Fitting ``k`` (and the center ``c``) from keypoint correspondences estimates
    the curvature; ``k > 0`` = barrel-like (moving stretched outward), ``k < 0``
    = pincushion. Compatible with :func:`skimage.transform.warp` via ``__call__``.
    """

    def __init__(self, center, k):
        self.center = np.asarray(center, dtype=np.float64)
        self.k = float(k)

    def __call__(self, coords):
        coords = np.asarray(coords, dtype=np.float64)
        d = coords - self.center
        r2 = np.sum(d * d, axis=1)
        return self.center + d * (1.0 + self.k * r2)[:, None]

    def effective_radius(self):
        """Rough effective sphere radius (px): ``r`` where displacement ~ r/... .
        Returns ``1/sqrt(|k|)`` as a curvature length scale, or inf if flat."""
        return float("inf") if abs(self.k) < 1e-18 else float(1.0 / np.sqrt(abs(self.k)))

    @classmethod
    def fit(cls, src, dst, grid=9):
        """Free-center fit: grid-search the center, linear least-squares for k.

        ``src``/``dst`` are template/moving (or residual) points. Returns the
        best :class:`_SphericalDistortion`."""
        src = np.asarray(src, dtype=np.float64); dst = np.asarray(dst, dtype=np.float64)
        cxs = np.linspace(src[:, 0].min(), src[:, 0].max(), grid)
        cys = np.linspace(src[:, 1].min(), src[:, 1].max(), grid)
        best = None
        for cx in cxs:
            for cy in cys:
                c = np.array([cx, cy])
                d = src - c
                r2 = np.sum(d * d, axis=1)
                # (dst - c) - d = d * (k * r2)  -> solve scalar k by least squares
                rhs = (dst - c) - d
                A = np.concatenate([d[:, 0] * r2, d[:, 1] * r2])
                b = np.concatenate([rhs[:, 0], rhs[:, 1]])
                denom = float(A @ A)
                k = float(A @ b / denom) if denom > 1e-18 else 0.0
                model = cls(c, k)
                err = float(np.mean(np.sum((model(src) - dst) ** 2, axis=1)))
                if best is None or err < best[0]:
                    best = (err, model)
        return best[1]


def distortion_fit_quality(distortion_tf, pairs, residual_matrix=None):
    """Quality metrics of a fitted distortion against the correspondences.

    Maps the template points through ``distortion_tf`` and compares to the
    (optionally residual_matrix-mapped) moving points. Returns a dict with
    ``rms`` (px), ``r2`` (1 - SS_res/SS_tot of the displacement), ``max_err``,
    ``n`` and, for spherical/radial models, ``k`` / ``effective_radius_px``.
    """
    template_pts, moving_pts = _pairs_to_arrays(pairs)
    dst = moving_pts
    if residual_matrix is not None:
        M = np.asarray(residual_matrix, dtype=float)
        hom = np.hstack([moving_pts, np.ones((len(moving_pts), 1))])
        mapped = (M @ hom.T).T
        dst = mapped[:, :2] / mapped[:, 2:3]
    pred = np.asarray(distortion_tf(template_pts), dtype=np.float64)
    res = pred - dst
    sq = np.sum(res * res, axis=1)
    rms = float(np.sqrt(np.mean(sq)))
    # R^2 on the displacement field (how much of template->dst motion is explained).
    disp = dst - template_pts
    ss_tot = float(np.sum((disp - disp.mean(axis=0)) ** 2))
    ss_res = float(np.sum(res * res))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
    out = {"rms": rms, "r2": r2, "max_err": float(np.sqrt(sq.max())), "n": len(pairs)}
    if hasattr(distortion_tf, "k"):
        out["k"] = float(distortion_tf.k)
    if hasattr(distortion_tf, "effective_radius"):
        out["effective_radius_px"] = distortion_tf.effective_radius()
    return out


def deformation_grid(distortion_tf, shape, n=16):
    """Return (xs_src, ys_src, xs_dst, ys_dst) of a regular grid mapped through
    ``distortion_tf`` (template -> moving) for a quiver/grid visualization.

    ``shape`` = (H, W); ``n`` grid lines per axis. Each is an (n, n) array."""
    H, W = shape
    gx, gy = np.meshgrid(np.linspace(0, W - 1, n), np.linspace(0, H - 1, n))
    coords = np.column_stack([gx.ravel(), gy.ravel()])
    mapped = np.asarray(distortion_tf(coords), dtype=np.float64)
    xs_dst = mapped[:, 0].reshape(n, n); ys_dst = mapped[:, 1].reshape(n, n)
    return gx, gy, xs_dst, ys_dst


def warp_field_with_distortion(field, global_matrix, distortion_tf, output_shape,
                               input_offset=(0.0, 0.0)):
    """Warp a COMPLEX field by a global similarity (+ optional distortion).

    The real and imaginary parts are warped **separately** with the same
    inverse map and recombined, avoiding phase-wrap interpolation artifacts.

    ``global_matrix`` maps moving -> template (the ImageAligner convention), so
    the inverse map fed to :func:`skimage.transform.warp` is its inverse,
    composed with the distortion (which already maps template -> moving).

    ``input_offset`` ``(dx, dy)`` is added to the final moving (input) coords.
    Use it to sample a LARGER, un-cropped moving field with a matrix that was
    estimated on a centered crop: pass the crop's ``(x0, y0)`` so output borders
    are filled from the moving periphery instead of going black.
    """
    field = np.asarray(field, dtype=np.complex128)

    global_tf = tf.AffineTransform(matrix=np.asarray(global_matrix, dtype=float))
    ox, oy = float(input_offset[0]), float(input_offset[1])

    def inverse_map(coords):
        # skimage.warp passes OUTPUT (template) coords and expects INPUT
        # (moving) coords. global_tf maps moving->template, so its inverse maps
        # template->moving; the distortion already maps template->moving.
        moving_coords = global_tf.inverse(coords)
        if distortion_tf is not None:
            moving_coords = distortion_tf(moving_coords)
        if ox or oy:
            moving_coords = moving_coords + np.array([ox, oy])
        return moving_coords

    re = tf.warp(field.real, inverse_map, output_shape=output_shape, preserve_range=True)
    im = tf.warp(field.imag, inverse_map, output_shape=output_shape, preserve_range=True)
    return re + 1j * im
