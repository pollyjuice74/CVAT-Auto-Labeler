# Automatic Image Labeling Pipeline Using CVAT and YOLO

This README provides instructions for the end-to-end workflow of exporting, refining, sampling, labeling, and importing images using **CVAT**, **YOLO models**, and custom scripts.


## **Summary of Steps**
1. Move labels zip file exported from CVAT **either as YOLO 1.1 or YOLOv8 pose 1.0** to `import_zips/` folder.
2. Split labels from CVAT using `import_from_cvat()` to train/val/test.
3. Train/refine the YOLO model using labeled data.
4. Sample new images in `unlabeled/` for labeling.
5. Automate labeling of `unlabeled/` images using the trained model.
6. Package and export labeled data back into CVAT using `export_to_cvat()` as **pose or box labels** using the same import format.
7. Loop.

**Important Notes:**
- Create a **new task on CVAT** any time you are **editing generated labels** to avoid deleting previous labels.
- When running `import_from_cvat()` the function will check `unlabeled/` folder for **images with the label in the imported zip file**. If it has a label it will move it to train/val/test split and **remove generated label** resulting in a "cleaned" `unlabeled/` folder. 

### **WARNING:** 
Ensure that the **imported ZIP file** contains **human-supervised labeled images**. If not, it may result in a model-generated/human-labeled mixed dataset.

---

## **Directory Structure**

- Recommended workflow includes creating a `datasets/` folder containig all `dataset/` folders, rename 'dataset/' to most appropriate name.

### **Bounding Box Detection**
```plaintext
dataset/
├── annotations/
│   └── annotations.xml
├── export_zips/
│   ├── upload_box_labels.zip
│   └── predicted_labels.zip
├── import_zips/
├── images/       # Each train/val/test has .jpg and .txt file in the same folder
│   ├── test/
│   ├── train/
│   └── val/
└── unlabeled/
    ├── frame_1.jpg
    └── frame_1.txt
```

### **Pose Detection**
```plaintext
dataset/
├── export_zips/
│   └── pose_labels.zip
├── images/       # .jpgs 
│   ├── train/
│   ├── val/
│   └── test/
├── labels/       # .txt 
│   ├── train/
│   ├── val/
│   └── test/
├── runs/
│   └── pose/train/weights/
├── unlabeled/
|    ├── frame_1.jpg
|    └── frame_1.txt
└── yamls/
     └── dataset.yaml

```

---

## **1. Importing labels from CVAT**

Export images and labels from CVAT as a ZIP file. The script splits images into `train`, `val`, and `test` folders while ensuring labels are organized.

```python
from scripts.import_from_cvat import *

# Define paths
zip_file_path = 'path/to/dataset/import_zips/import_file.zip'
dataset_output_dir = 'path/to/dataset' # `dataset/` folder path

# Run export
import_from_cvat(zip_file_path, dataset_output_dir, train_split=0.6, val_split=0.2, test_split=0.2, export_format='yolo_pose') # 'yolo_pose' or 'yolo_detect' for pose, box respectively
```

**Usage:**
1. Export images from CVAT in YOLO-compatible format.
2. Place the `.zip` file into the `import_zips/` folder.
3. The script automatically organizes:
   - **Images** into `train`, `val`, and `test`.
   - **Labels** into `labels/train`, `labels/val`, and `labels/test` folders.

---

## **2. Refining a YOLO Model**
Train a YOLO model using exported data.

```python
from ultralytics import YOLO

# Define paths
model_path = 'path/to/model/weights/best.pt'
yaml_path = 'dataset/yamls/dataset.yaml' # for pose

# Train model
model = YOLO(model_path)
results = model.train(data=yaml_path, epochs=3000, imgsz=(2560, 1440), batch=4)
```

---

## **3. Sampling New Image Files**
Generate samples of images for further labeling.

```python
from scripts.sample_Gurabo import sample_Gurabo

# Define paths
base_path = 'path/to/source/videos'
dataset_output_dir = 'path/to/dataset/unlabeled/'

# Run sampling
sample_Gurabo(base_path, dataset_output_dir, sample_colonie_times=2, sample_files_times=1000)
```

**Usage:**
- Extract new frames or files for labeling.

---

## **4. Automating Image Labeling**
Automate the labeling of unlabeled images using a trained YOLO model.

```python
from scripts.img_labeler import label_files

# Define paths
pose_model_path = 'path/to/pose_model/weights/best.pt'
box_model_path = 'path/to/box_model/weights/best.pt'

unlabeled_files_dir = 'path/to/dataset/unlabeled/'

# Run labeling
label_files(box_model_path, unlabeled_files_dir, label_format='yolo_detect')
```

**Usage:**
- Input: Folder containing `unlabeled/` images.
- Output: Labels saved in the same `unlabeled/` folder.

---

## **5. Importing Images Back to CVAT**
Package the labeled images and prepare them for upload to CVAT.

```python
from scripts.export2cvat import *

# Define paths
labeled_files_dir = 'path/to/dataset/unlabeled' # generated labels
zip_output_path = 'path/to/dataset/export_zips'

# Export to CVAT
export_to_cvat_yolo(labeled_files_dir, zip_output_path, export_format='yolo_detect', classes=['bee'], export_filename='detect_tst.zip', labeled=True)
```

**Usage:**
1. Export the labeled files into a ZIP archive.
2. Move the file to your local machine:
   ```bash
   scp username@server:path/to/dataset/export_zips/export_file.zip /local/path/to/downloads
   ```
3. Create a new task in CVAT.
4. Go to **Actions > Upload Annotations** and select:
   - `YOLO 1.1` for bounding box labels.
   - `YOLOv8 Pose 1.0` for pose detection.

---