#!/usr/bin/env python3
"""Quick verification script to test if backend dependencies are installed correctly."""

import sys

def test_imports():
    """Test all required imports."""
    errors = []
    
    print("Testing imports...")
    
    try:
        import cv2
        print(f"✓ OpenCV: {cv2.__version__}")
    except ImportError as e:
        errors.append(f"✗ OpenCV: {e}")
    
    try:
        import dlib
        print(f"✓ dlib: {dlib.__version__}")
    except ImportError as e:
        errors.append(f"✗ dlib: {e}")
    
    try:
        import numpy as np
        print(f"✓ NumPy: {np.__version__}")
    except ImportError as e:
        errors.append(f"✗ NumPy: {e}")
    
    try:
        import scipy
        print(f"✓ SciPy: {scipy.__version__}")
    except ImportError as e:
        errors.append(f"✗ SciPy: {e}")
    
    try:
        import imutils
        print(f"✓ imutils: OK")
    except ImportError as e:
        errors.append(f"✗ imutils: {e}")
    
    try:
        import keyboard
        print(f"✓ keyboard: OK")
    except ImportError as e:
        errors.append(f"✗ keyboard: {e}")
    
    try:
        import fastapi
        print(f"✓ FastAPI: {fastapi.__version__}")
    except ImportError as e:
        errors.append(f"✗ FastAPI: {e}")
    
    try:
        import uvicorn
        print(f"✓ uvicorn: {uvicorn.__version__}")
    except ImportError as e:
        errors.append(f"✗ uvicorn: {e}")
    
    try:
        from blink_morse import BlinkMorseDetector
        print(f"✓ blink_morse module: OK")
    except ImportError as e:
        errors.append(f"✗ blink_morse module: {e}")
    
    try:
        from backend.main import app
        print(f"✓ backend.main module: OK")
    except ImportError as e:
        errors.append(f"✗ backend.main module: {e}")
    
    return errors

def test_shape_predictor():
    """Test if shape predictor file exists."""
    from pathlib import Path
    
    print("\nTesting shape predictor file...")
    
    base_dir = Path(__file__).parent
    predictor = base_dir / "shape_predictor_68_face_landmarks.dat"
    
    if predictor.exists():
        size_mb = predictor.stat().st_size / (1024 * 1024)
        print(f"✓ Found: {predictor} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"✗ Not found: {predictor}")
        print(f"  Please download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Backend Verification Script")
    print("=" * 60)
    
    errors = test_imports()
    predictor_ok = test_shape_predictor()
    
    print("\n" + "=" * 60)
    if errors:
        print("❌ ERRORS FOUND:")
        for error in errors:
            print(f"  {error}")
        print("\nPlease install missing packages:")
        print("  python -m pip install <package-name>")
        sys.exit(1)
    elif not predictor_ok:
        print("⚠️  WARNING: Shape predictor file missing")
        print("   Backend will fail to start without it.")
        sys.exit(1)
    else:
        print("✅ All checks passed! Backend should work.")
        print("\nTo start the backend, run:")
        print("  python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000")
    print("=" * 60)

