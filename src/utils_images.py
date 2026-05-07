# Copyright (c) 2025 Corentin Soubeiran
# SPDX-License-Identifier: MIT
# General imports
import os
import json
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, Optional, Tuple

# Imaging imports
import numpy as np
import cv2
import tifffile


def load_imgfile(filename):
    """

    """
    # print("filename", filename)
    isfdacx = False
    if (isinstance(filename, str)) and (not os.path.isfile(filename)) and (not filename.endswith('.map')):
        raise ValueError("'%s' is not a file." % filename)

    if not os.path.isfile(filename) and (not filename.endswith('.map')):
        img = filename

    elif filename.endswith('.tiff') or filename.endswith('.tif'):
        img = cv2.imread(filename, flags=cv2.IMREAD_ANYDEPTH)
        img = np.float32(img)
    elif filename.endswith('.jpg') or filename.endswith(".png"):
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif filename.endswith('.holo.npy'):
        # Holograms have 2 channels: Amplitude and Phase
        img = np.load(filename)
        if img.shape[-1] != 2:
            raise ValueError('holo images must have 2 channels')
    elif filename.endswith('.phy.npy'):
        # Wave front as2 channels: Amplitude and Phase
        img = np.load(filename)
        if img.shape[-1] != 2:
            if img.shape[0] == 2:
                img = img.transpose((1, 2, 0))
            else: 
                raise ValueError('phy wavefront images must have 2 channels: Amplitude and Phase')
        add_feat_axis = False
    elif filename.endswith('.wf.npy'):
        img = np.load(filename)
        re = np.real(img)
        im = np.imag(img)
        img = np.concatenate([re[..., np.newaxis], im[..., np.newaxis]], axis=-1)
        add_feat_axis = False
    elif filename.endswith('.npy'):
        img = np.load(filename)
        re = np.real(img)
        im = np.imag(img)
        img = np.concatenate([re[..., np.newaxis], im[..., np.newaxis]], axis=-1)
    else:
        raise ValueError('unknown filetype for %s' % filename)

    return img

def make_PHY(amp: np.array,opd: np.array,wavelenght:float=None,from_phase = False):
    # two channels: amplitude and phase
    if from_phase:
        phase = opd
    else:
        phase = opd/wavelenght*(2*np.pi)
    amplitude = amp
    phy = np.stack([amplitude,phase],axis=0)
    return phy

def make_WF(amp: np.array,opd: np.array,wavelenght:float=None,from_phase = False):
    # Complex wavefront
    if from_phase:
        phase = opd
    else:
        phase = opd/wavelenght*(2*np.pi)  
    amplitude = amp
    wf = amplitude*np.exp(1j*phase)
    return wf

def save_npy(obj:np.array,path:str):
    np.save(path,obj)

def load_npy(path:str):
    data = np.load(path)
    return data


def load_wavefront_tif(path: str, frame_index: int = 0):
    """Load a 2-channel wavefront TIFF.

    Channel 0 = phase (rad), channel 1 = amplitude. Supports the four
    common layouts (H, W, 2), (2, H, W), (T, 2, H, W), (T, H, W, 2).

    Returns
    -------
    phase, amp : float32 arrays of shape (H, W)
    n_frames : int
        Number of time frames present on disk (1 for 3-D files).
    """
    raw = np.asarray(tifffile.imread(path))

    if raw.ndim == 3:
        n_frames = 1
        if raw.shape[0] == 2:
            phase, amp = raw[0], raw[1]
        elif raw.shape[-1] == 2:
            phase, amp = raw[..., 0], raw[..., 1]
        else:
            raise ValueError(
                f"3-D wavefront TIFF must have a size-2 channel axis; "
                f"got shape {raw.shape}"
            )
    elif raw.ndim == 4:
        if raw.shape[1] == 2:
            n_frames = raw.shape[0]
            t = max(0, min(int(frame_index), n_frames - 1))
            phase, amp = raw[t, 0], raw[t, 1]
        elif raw.shape[-1] == 2:
            n_frames = raw.shape[0]
            t = max(0, min(int(frame_index), n_frames - 1))
            phase, amp = raw[t, ..., 0], raw[t, ..., 1]
        else:
            raise ValueError(
                f"4-D wavefront TIFF must have a size-2 channel axis at "
                f"position 1 (TCYX) or -1 (TYXC); got shape {raw.shape}"
            )
    else:
        raise ValueError(
            f"Unsupported wavefront TIFF rank {raw.ndim} (shape {raw.shape}); "
            f"expected 3-D or 4-D."
        )

    return phase.astype(np.float32), amp.astype(np.float32), n_frames


def save_stack(path: str,
               stack: np.ndarray,
               wavelengths: Optional[Tuple[float, ...]] = None,
               source_path: Optional[str] = None,
               metas: Optional[Dict[str, object]] = None,
               input_format: Optional[str] = None,
               timestamps: Optional[Tuple[float, ...]] = None) -> None:
    """Save a real-valued stack as an ImageJ hyperstack TIFF.

    Accepts arrays of shape ``(N, H, W)``, ``(N, H, W, C)`` or
    ``(N, nWL, H, W, C)``. Always writes axes ``"TZCYX"``, with float32
    precision. Optical / acquisition metadata supplied via ``metas`` is
    stored as JSON in the ImageJ ``Info`` tag. Per-frame timestamps are
    written as OME-XML ``DeltaT`` planes when ``timestamps`` is given.
    """
    if stack.ndim == 3:
        N, H, W = stack.shape
        nwl, C = 1, 1
        stack = stack[:, None, :, :, None]
    elif stack.ndim == 4:
        N, H, W, C = stack.shape
        nwl = 1
        stack = stack[:, None, :, :, :]
    elif stack.ndim == 5:
        N, nwl, H, W, C = stack.shape
    else:
        raise ValueError(f"stack must be 3-D, 4-D or 5-D; got {stack.shape}")

    if timestamps is not None and len(timestamps) != N:
        raise ValueError("timestamps length must match the time dimension")

    if input_format == "TZCYX":
        out = stack.astype(np.float32)
    else:
        out = stack.transpose((0, 1, 4, 2, 3)).astype(np.float32)

    meta: Dict[str, object] = {}
    if wavelengths is not None:
        meta["wavelengths_nm"] = list(map(float, wavelengths))
    if source_path is not None:
        meta["source_path"] = str(source_path)
    if metas is not None:
        meta.update(metas)

    ij_metadata = {"axes": "TZCYX", "Info": json.dumps(meta)}

    description = None
    if timestamps is not None:
        ome = ET.Element(
            "OME",
            xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06",
        )
        image = ET.SubElement(ome, "Image", ID="Image:0")
        pixels = ET.SubElement(
            image, "Pixels",
            DimensionOrder="TZCYX", Type="float",
            SizeT=str(N), SizeZ=str(nwl), SizeC=str(C),
            SizeY=str(H), SizeX=str(W),
        )
        for t, dt in enumerate(timestamps):
            ET.SubElement(
                pixels, "Plane",
                TheT=str(t), TheZ="0", TheC="0",
                DeltaT=str(float(dt)),
            )
        description = ET.tostring(ome, encoding="unicode")

    dir_name = os.path.dirname(path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    tifffile.imwrite(
        path, out, imagej=True,
        metadata=ij_metadata, description=description,
    )
