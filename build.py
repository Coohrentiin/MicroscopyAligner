#!/usr/bin/env python3
"""
Build script for creating executable using PyInstaller
Run: python build_exe.py
"""
# Copyright (c) 2025 Corentin Soubeiran
# SPDX-License-Identifier: MIT

import PyInstaller.__main__
import sys
import os
from pathlib import Path
import pathlib

root = pathlib.Path(__file__).parent

def build_exe():
    """Build the executable using PyInstaller"""
    
    # Get the directory of this script
    current_dir = Path(__file__).parent.absolute()
    
    # Define PyInstaller arguments
    args = [
        str(root / "src/main.py"),  # Main script
        '--onedir',          
        '--name=MicroscopyAligner',  
        '--icon=src/resources/icon.png',      
        f"--paths={root}/src",   
        
        # Include necessary packages
        '--hidden-import=scipy.ndimage',
        # '--hidden-import=scipy.special._ufuncs',
        # '--hidden-import=scipy.special._ufuncs_cxx',
        '--hidden-import=scipy.fft._pocketfft',
        # '--hidden-import=scipy.fft._fftlog_backend',
        # '--hidden-import=scipy.linalg.cython_blas',
        # '--hidden-import=scipy.linalg.cython_lapack',
        # '--hidden-import=scipy.spatial.transform._rotation_groups',
        "--hidden-import=scipy.fft",
        # "--hidden-import=scipy.special",
        # "--hidden-import=scipy.sparse.csgraph._validation",
        '--hidden-import=skimage.transform',
        '--hidden-import=skimage.registration',
        '--hidden-import=tifffile',
        '--hidden-import=PIL',
        '--hidden-import=cv2',
        '--hidden-import=matplotlib',
        # '--hidden-import=matplotlib.backends.backend_qt5agg',


        # --- Exclusions to reduce size ---
        '--exclude-module=pytest',
        '--exclude-module=numpy.tests',
        '--exclude-module=scipy.tests',
        '--exclude-module=skimage.tests',
        '--exclude-module=matplotlib.tests',
        '--exclude-module=PyQt5',
        '--exclude-module=tkinter',
        '--exclude-module=skimage.data',
        '--exclude-module=PIL.ImageTk',
        '--exclude-module=tensorflow',
        '--exclude-module=tensorboard',
        '--exclude-module=keras',
        '--exclude-module=torch',
        '--exclude-module=nvidia',

        # Collect all data from packages
        # '--collect-all=scipy',
        # '--collect-all=skimage',
        # '--collect-all=tifffile',
        
        # Additional options
        '--clean',           # Clean cache
        '--noconfirm',       # Replace output without asking
        
        # Paths
        f'--distpath={current_dir / "dist"}',
        f'--workpath={current_dir / "build"}',
        f'--specpath={current_dir}',
    ]
    
    # Add optimization for smaller file size (optional)
    if '--optimize' in sys.argv:
        args.extend(['--strip', '--upx-dir=/usr/local/share/'])
    
    print("Building executable with PyInstaller...")
    print("This may take a few minutes...")
    
    # Run PyInstaller
    PyInstaller.__main__.run(args)
    
    print("\n" + "="*50)
    print("Build complete!")
    print(f"Executable location: {current_dir / 'dist' / 'MicroscopyAligner'}")
    print("="*50)

def create_spec_file():
    """Alternative: Create a spec file for more control"""
    
    spec_content = """
# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['image_aligner.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'scipy.ndimage',
        'skimage.transform',
        'skimage.registration',
        'tifffile',
        'PIL',
        'cv2',
        'scipy.spatial.transform',
        'scipy.special._ufuncs_cxx',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='MicroscopyAligner',
    debug=False,
    bootloader_ignore_signals=False,
    icon='src/resources/icon.ico',
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

if sys.platform == 'darwin':
    app = BUNDLE(
        exe,
        name='MicroscopyAligner.app',
        icon=None,
        bundle_identifier='com.microscopy.aligner',
        info_plist={
            'NSHighResolutionCapable': 'True',
            'NSRequiresAquaSystemAppearance': 'False',
        },
    )
"""
    
    with open('MicroscopyAligner.spec', 'w') as f:
        f.write(spec_content)
    
    print("Spec file created: MicroscopyAligner.spec")
    print("You can now build with: pyinstaller MicroscopyAligner.spec")

if __name__ == "__main__":
    # Check if user wants to create spec file instead
    if '--spec' in sys.argv:
        create_spec_file()
    else:
        build_exe()
        
    # Additional instructions
    print("\n" + "="*50)
    print("INSTRUCTIONS FOR DEPLOYMENT:")
    print("="*50)
    print("""
1. Install requirements:
   pip install -r requirements.txt

2. Build executable:
   python build.py

3. For Windows:
   - The executable will be in dist/MicroscopyAligner.exe
   - Can be distributed as a single file

4. For macOS:
   - Run: python build_exe.py
   - The app bundle will be in dist/MicroscopyAligner.app

5. For Linux:
   - Run: python build_exe.py
   - The executable will be in dist/MicroscopyAligner

TROUBLESHOOTING:

If you encounter issues with missing modules:
1. Add them to hidden imports in build.py
2. Or create a spec file: python build.py --spec
3. Then edit and run: pyinstaller MicroscopyAligner.spec

For smaller file size:
- Install UPX compression tool
- Run: python build.py --optimize
""")
