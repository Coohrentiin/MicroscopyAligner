# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['Z:\\home\\coohrentiin\\workspace\\Manual_Registration_App\\src\\main.py'],
    pathex=['Z:\\home\\coohrentiin\\workspace\\Manual_Registration_App/src'],
    binaries=[],
    datas=[],
    hiddenimports=['scipy.ndimage', 'scipy.fft._pocketfft', 'scipy.fft', 'skimage.transform', 'skimage.registration', 'tifffile', 'PIL', 'cv2', 'matplotlib'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['pytest', 'numpy.tests', 'scipy.tests', 'skimage.tests', 'matplotlib.tests', 'PyQt5', 'tkinter', 'skimage.data', 'PIL.ImageTk', 'tensorflow', 'tensorboard', 'keras', 'torch', 'nvidia'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MicroscopyAligner',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['src\\resources\\icon.png'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=True,
    upx=True,
    upx_exclude=[],
    name='MicroscopyAligner',
)
