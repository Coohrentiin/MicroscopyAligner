# Copyright (c) 2025 Corentin Soubeiran
# SPDX-License-Identifier: MIT
# General imports
import os
from datetime import datetime

# Imaging imports
import numpy as np
import cv2


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
