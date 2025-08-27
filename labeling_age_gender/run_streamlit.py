#!/usr/bin/env python3
"""
Launcher script for the Automatic Facial Image Labelling System (Streamlit GUI)
"""

import sys
import subprocess
import os

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit', 'tensorflow', 'cv2', 'numpy', 'PIL', 'pandas', 'plotly', 'openpyxl'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                from PIL import Image
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies():
    """Install missing dependencies"""
    print("Installing required dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_streamlit.txt"])
        print("Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("Failed to install dependencies. Please install manually:")
        print("pip install -r requirements_streamlit.txt")
        return False

def main():
    print("🤖 Automatic Facial Image Labelling System - Streamlit GUI")
    print("=" * 60)
    
    # Check if model file exists
    if not os.path.exists("age_gender_pseudolabel.h5"):
        print("❌ Error: Model file 'age_gender_pseudolabel.h5' not found!")
        print("Please ensure the model file is in the current directory.")
        input("Press Enter to exit...")
        return
    
    # Check dependencies
    missing = check_dependencies()
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("Installing dependencies...")
        
        if not install_dependencies():
            input("Press Enter to exit...")
            return
    
    print("✅ All dependencies are available!")
    print("🚀 Launching Streamlit GUI application...")
    print()
    
    try:
        # Launch Streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_gui.py"])
    except KeyboardInterrupt:
        print("\n👋 Application closed by user.")
    except Exception as e:
        print(f"❌ Error launching Streamlit: {str(e)}")
        print("Please check the error message above and ensure all files are present.")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
