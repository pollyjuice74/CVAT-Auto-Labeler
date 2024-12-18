# Automatic Image Labeling and Integration with CVAT

This README provides instructions for setting up automatic labeling for images using YOLO models and CVAT.

---

## 1. **Overview**
The script automates the process of labeling unlabeled images using a trained YOLO model. The pipeline includes:
1. Exporting images from CVAT.
2. Running YOLO inference to predict labels.
3. Organizing labeled files.
4. Re-importing labeled files back into CVAT.

This process ensures that unlabeled images can be quickly annotated and integrated into your dataset.

---

## 2. **Prerequisites**
Ensure you have the following installed:
- Python >= 3.8
- Ultralytics YOLO library
- CVAT environment for annotation management
- Required Python modules:
  - `shutil`
  - `os`
  - `scripts` package (custom scripts like `img_labeler`, `cvat_importer`)

Ensure your YOLO model weights (`.pt`) and configuration files (`.yaml`) are available.

---

## 3. **Directory Structure**
Organize your project as follows:
```plaintext
project_root/
|-- datasets/
|   |-- tst_pose_4_key_points/
|       |-- import_zips/              # Exported unlabeled images from CVAT
|       |-- frames/                   # Unlabeled and labeled images folder
|       |-- runs/
|           |-- pose/train/weights/   # YOLO model weights
|       |-- output/                   # Directory for labeled images
|-- scripts/
    |-- img_labeler.py                # Automatic labeling script
    |-- cvat_importer.py              # Script to re-import to CVAT
```

---

## 4. **Automatic Labeling Pipeline**

### Step 1: Exporting Images from CVAT
Export images as a ZIP file from CVAT with human-supervised labeled images.

```python
from scripts.export_from_cvat import *

# Export images from CVAT
export_from_cvat_zip(
    file_path_zip="/path/to/your/exported_unlabeled_images.zip",
    dataset_output_dir="/path/to/dataset_output/",
    train_split=0.6, val_split=0.2, test_split=0.2,
    export_format='yolo_pose'
)
```

This splits the dataset into `train`, `val`, and `test` folders.

---

### Step 2: Automatic Labeling with YOLO
Run automatic labeling on the exported images using a trained YOLO model.

```python
from scripts.img_labeler import label_files

# Paths to YOLO weights and unlabeled images
model_paths = [
    "path/to/weights/best.pt",  # Main model weights
    "path/to/weights/last.pt"   # Backup weights
]
unlabeled_files_dir = "/path/to/unlabeled/images"

# Run labeling process
label_files(bbox_model_path=model_paths[0],
            unlabeled_files_dir=unlabeled_files_dir,
            label_format='yolo_detect')
```
- **Input:**
   - Trained YOLO weights (`.pt` files).
   - Folder containing unlabeled images.
- **Output:**
   - Labeled files in the same directory.

---

### Step 3: Re-importing Labeled Files to CVAT
Once labeling is complete, re-import the labeled files back into CVAT for further review.

```python
from scripts.import_to_cvat import export_to_cvat_yolo

# Re-import to CVAT
export_to_cvat_yolo(
    labeled_files_dir="/path/to/labeled_images",
    img_output_path="/path/to/output_folder",
    export_format='yolo_detect',
    export_filename='detect_tst.zip',
    labeled=True
)
```
- **Export Format:** `yolo_detect` ensures compatibility with CVAT's YOLO import format.
- **Output File:** A `.zip` file ready for upload to CVAT.

---

## 5. **Importing Labeled Files to CVAT**
To upload labeled files to CVAT:
1. Open your CVAT workspace.
2. Go to **Actions > Upload Annotations**.
3. Upload the `detect_tst.zip` file generated in Step 3.
4. Confirm the upload to see bounding box annotations.

---

## 6. **Notes**
1. Ensure your YOLO model is well-trained to avoid generating noisy annotations.
2. Backup manual annotations before overwriting them with auto-generated labels.
3. Use the `export_format='yolo_pose'` or `yolo_detect` as required for CVAT.

---

## 7. **Troubleshooting**
- **Missing Labels:** Verify the YOLO weights path and dataset folder structure.
- **CVAT Import Errors:** Ensure the ZIP file format and YOLO compatibility.

---

## 8. **Credits**
This script utilizes:
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [CVAT Annotation Tool](https://github.com/openvinotoolkit/cvat)
- Custom scripts for image handling and labeling.

---

## 9. **License**
This project is licensed under the MIT License. For details, see the LICENSE file.

---

## 10. **Contact**
For any issues, feel free to contact the project maintainer.
