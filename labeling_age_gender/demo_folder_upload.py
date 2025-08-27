#!/usr/bin/env python3
"""
Demo script for testing the enhanced Streamlit GUI with folder upload functionality
"""

import os
import zipfile
import tempfile
import glob
from PIL import Image
import numpy as np

def create_sample_dataset():
    """Create a sample dataset for testing"""
    print("🔄 Creating sample dataset...")
    
    # Create sample images with proper naming convention
    sample_data = [
        ("25_0_person1.jpg", 25, 0),  # 25-year-old male
        ("30_1_person2.jpg", 30, 1),  # 30-year-old female
        ("45_0_person3.jpg", 45, 0),  # 45-year-old male
        ("22_1_person4.jpg", 22, 1),  # 22-year-old female
        ("60_0_person5.jpg", 60, 0),  # 60-year-old male
        ("35_1_person6.jpg", 35, 1),  # 35-year-old female
        ("28_0_person7.jpg", 28, 0),  # 28-year-old male
        ("40_1_person8.jpg", 40, 1),  # 40-year-old female
    ]
    
    # Create sample directory
    sample_dir = "sample_dataset"
    os.makedirs(sample_dir, exist_ok=True)
    
    # Create sample images
    for filename, age, gender in sample_data:
        # Create a simple colored image
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Add some color variation based on age/gender
        if gender == 0:  # Male
            img_array[:, :, 0] = np.clip(img_array[:, :, 0] + 50, 0, 255)  # More blue
        else:  # Female
            img_array[:, :, 2] = np.clip(img_array[:, :, 2] + 50, 0, 255)  # More red
        
        # Age affects brightness
        age_factor = age / 100.0
        img_array = np.clip(img_array * (0.5 + age_factor * 0.5), 0, 255).astype(np.uint8)
        
        # Save image
        img = Image.fromarray(img_array)
        img_path = os.path.join(sample_dir, filename)
        img.save(img_path)
        print(f"✅ Created: {filename}")
    
    # Create some unlabeled images
    unlabeled_names = ["unlabeled1.jpg", "unlabeled2.jpg", "unlabeled3.jpg", "unlabeled4.jpg"]
    for filename in unlabeled_names:
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img_path = os.path.join(sample_dir, filename)
        img.save(img_path)
        print(f"✅ Created: {filename}")
    
    print(f"✅ Sample dataset created in '{sample_dir}' directory")
    return sample_dir

def create_zip_file(dataset_dir, zip_filename="sample_dataset.zip"):
    """Create a ZIP file from the dataset directory"""
    print(f"🔄 Creating ZIP file: {zip_filename}")
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, dataset_dir)
                zipf.write(file_path, arcname)
                print(f"📁 Added to ZIP: {arcname}")
    
    print(f"✅ ZIP file created: {zip_filename}")
    return zip_filename

def test_folder_upload_functionality():
    """Test the folder upload functionality"""
    print("🧪 Testing folder upload functionality...")
    
    # Create sample dataset
    dataset_dir = create_sample_dataset()
    
    # Create ZIP file
    zip_file = create_zip_file(dataset_dir)
    
    # Test file detection
    print("\n🔍 Testing file detection...")
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff')
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(dataset_dir, ext)))
    
    print(f"📊 Found {len(image_files)} image files:")
    for file_path in image_files:
        filename = os.path.basename(file_path)
        print(f"  - {filename}")
    
    # Test labeled vs unlabeled classification
    print("\n🏷️ Testing labeled vs unlabeled classification...")
    labeled_files = []
    unlabeled_files = []
    
    for file_path in image_files:
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        
        # Check if filename follows the labeled format (age_gender_*.jpg)
        if len(parts) >= 2:
            try:
                age = int(parts[0])
                gender = int(parts[1])
                if 0 <= age <= 100 and gender in [0, 1]:
                    labeled_files.append(file_path)
                    print(f"  ✅ Labeled: {filename} (Age: {age}, Gender: {'Male' if gender == 0 else 'Female'})")
                    continue
            except ValueError:
                pass
        
        # If not labeled, consider it unlabeled
        unlabeled_files.append(file_path)
        print(f"  🔍 Unlabeled: {filename}")
    
    print(f"\n📊 Summary:")
    print(f"  - Labeled images: {len(labeled_files)}")
    print(f"  - Unlabeled images: {len(unlabeled_files)}")
    print(f"  - Total images: {len(image_files)}")
    
    # Cleanup
    print(f"\n🧹 Cleaning up sample files...")
    import shutil
    shutil.rmtree(dataset_dir)
    os.remove(zip_file)
    print("✅ Cleanup complete!")

def main():
    print("🤖 Enhanced Streamlit GUI - Folder Upload Demo")
    print("=" * 60)
    
    print("This demo creates a sample dataset to test the folder upload functionality.")
    print("The enhanced Streamlit GUI now supports:")
    print("  📁 ZIP file uploads")
    print("  📂 Manual folder path input")
    print("  🏷️ Automatic labeled/unlabeled detection")
    print("  📊 Real-time progress tracking")
    print("  🎨 Enhanced user interface")
    
    print("\n" + "=" * 60)
    
    try:
        test_folder_upload_functionality()
        print("\n🎉 Demo completed successfully!")
        print("\n🚀 To test the enhanced GUI:")
        print("  1. Run: streamlit run streamlit_gui.py")
        print("  2. Go to 'Load Dataset' page")
        print("  3. Choose 'Upload Folder (Recommended)'")
        print("  4. Upload a ZIP file or enter folder path")
        print("  5. Watch the automatic detection work!")
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        print("Please check your environment and try again.")

if __name__ == "__main__":
    main()
