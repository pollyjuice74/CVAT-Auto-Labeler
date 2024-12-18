import os
from zipfile import ZipFile
from PIL import Image  # for image dimension normalization
import shutil


def export_to_cvat_yolo(labeled_files_dir, zip_output_path, export_format='pose', subset='train', classes=[], export_filename='cvat_upload.zip', labeled=True):
    """
    Exports labeled data in CVAT-compatible YOLO format, supporting 'yolo_detect' and 'yolo_pose' formats.

    Parameters:
    labeled_files_dir (str): Directory containing .jpg images and their corresponding .txt annotation files.
    zip_output_path (str): Directory to save the exported .zip file.
    export_format (str): Format to export: 'yolo_detect' or 'yolo_pose'.
    subset (str): Subset label (e.g., 'train', 'val', 'test').
    classes (list): List of class names for the dataset.
    export_filename (str): Name of the output .zip file.
    labeled (bool): Whether to include labels in the export.

    Raises:
    AssertionError: If export_format is not 'yolo_detect' or 'yolo_pose'.

    Output:
    - Creates a .zip archive conforming to YOLO format, suitable for CVAT import.
    """
    assert export_format in ['yolo_detect', 'yolo_pose'], " Export format must be one of: detect, pose. "

    if export_format=='yolo_detect':
        yolo_detect_export(labeled_files_dir, zip_output_path, subset=subset, classes=classes, export_filename=export_filename, labeled=labeled)

    elif export_format=='yolo_pose':
        yolo_pose_export(labeled_files_dir, zip_output_path, subset=subset, classes=classes, export_filename=export_filename, labeled=labeled)


def yolo_pose_export(labeled_files_dir, zip_output_path,subset, classes, export_filename, labeled):
    """
    Exports labeled data in YOLOv8 pose format, including label normalization and dataset packaging.

    Parameters:
    labeled_files_dir (str): Directory containing .jpg images and their corresponding .txt annotations.
    zip_output_path (str): Directory to save the exported .zip file.
    subset (str): Subset label (e.g., 'train', 'val', 'test').
    classes (list): List of class names for the dataset.
    export_filename (str): Name of the output .zip file.
    labeled (bool): Whether to include labels in the export.

    Output:
    - Creates a .zip archive containing:
        - Normalized pose labels in 'labels/' directory.
        - Images in 'images/' directory.
        - A `data.yaml` file for YOLOv8 configuration.
        - A text file listing image paths (`<subset>.txt`).
    """
    # Prepare output directories for images and labels
    images_output_dir = os.path.join(zip_output_path, 'images', subset)
    labels_output_dir = os.path.join(zip_output_path, 'labels', subset)
    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(labels_output_dir, exist_ok=True)

    # Process images and normalize labels
    for file_name in os.listdir(labeled_files_dir):
        if file_name.endswith('.jpg'):
            # Copy image files to the images directory
            shutil.copy(os.path.join(labeled_files_dir, file_name), images_output_dir)
        elif file_name.endswith('.txt') and labeled:
            # Normalize labels and write to labels directory
            label_path = os.path.join(labeled_files_dir, file_name)
            with open(label_path, 'r') as label_file:
                labels = label_file.readlines()

            # Get image dimensions for normalization
            img_path = os.path.join(labeled_files_dir, file_name.replace('.txt', '.jpg'))
            with Image.open(img_path) as img:
                img_width, img_height = img.size

            # Normalize labels for pose format
            normalized_labels = normalize_pose_labels(labels, img_width, img_height)

            # Write normalized labels to the labels output directory
            normalized_label_path = os.path.join(labels_output_dir, file_name)
            with open(normalized_label_path, 'w') as normalized_label_file:
                normalized_label_file.writelines(normalized_labels)

    # Create data.yaml for YOLOv8 format
    data_yaml_path = os.path.join(zip_output_path, 'data.yaml')
    with open(data_yaml_path, 'w') as yaml_file:
        yaml_file.write(f"path: ./\n")
        yaml_file.write(f"train: train.txt\n")
        yaml_file.write(f"kpt_shape: [4, 3]\n")  # Update to match the number of keypoints in your skeleton
        yaml_file.write("names:\n")
        for i, class_name in enumerate(classes):
            yaml_file.write(f"  {i}: {class_name}\n")

    # Create subset file (train.txt) listing image paths
    subset_txt_path = os.path.join(zip_output_path, f"{subset}.txt")
    with open(subset_txt_path, 'w') as subset_file:
        for file_name in os.listdir(images_output_dir):
            subset_file.write(f"images/{subset}/{file_name}\n")

    # Package everything into a zip file
    zip_file_path = os.path.join(zip_output_path, export_filename)
    with ZipFile(zip_file_path, 'w') as zipf:
        # Add images and labels
        for root, _, files in os.walk(os.path.join(zip_output_path, 'images')):
            for file in files:
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), zip_output_path))
        for root, _, files in os.walk(os.path.join(zip_output_path, 'labels')):
            for file in files:
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), zip_output_path))

        # Add data.yaml and subset file
        zipf.write(data_yaml_path, 'data.yaml')
        zipf.write(subset_txt_path, f"{subset}.txt")

    # Cleanup temporary directories and files
    shutil.rmtree(os.path.join(zip_output_path, 'images'))
    shutil.rmtree(os.path.join(zip_output_path, 'labels'))
    os.remove(data_yaml_path)
    os.remove(subset_txt_path)

    print(f"CVAT-compatible YOLOv8 Pose zip created at {zip_file_path}")


def normalize_pose_labels(labels, img_width, img_height):
    """
    Normalizes pose labels for YOLOv8 format, converting absolute keypoints
    to relative coordinates and replacing confidence scores with visibility flags.

    Parameters:
    labels (list): List of strings where each line represents an annotation in the format:
                   <class_id> <x_center> <y_center> <width> <height> <kpx1> <kpy1> <kp_conf1> ...
    img_width (int): Width of the image.
    img_height (int): Height of the image.

    Returns:
    list: Normalized pose labels with visibility flags (2) in the following format:
          <class_id> <x_center> <y_center> <width> <height> <kp_x1> <kp_y1> 2 <kp_x2> <kp_y2> 2 ...
    """
    normalized_labels = []

    for line in labels:
        parts = line.strip().split()
        class_id = parts[0]  # The class ID
        bbox = [float(x) for x in parts[1:5]]  # Bounding box: x_center, y_center, width, height
        keypoints = [float(x) for x in parts[5:]]  # Keypoints: x1, y1, conf1, x2, y2, conf2, ...

        # Normalize bounding box
        x_center = bbox[0]
        y_center = bbox[1]
        width = bbox[2]
        height = bbox[3]

        # Normalize keypoints and replace confidence with 2
        normalized_keypoints = []
        for i in range(0, len(keypoints), 3):  # Keypoints are grouped as (x, y, conf)
            kp_x = keypoints[i] / img_width  # Normalize x
            kp_y = keypoints[i + 1] / img_height  # Normalize y
            kp_visibility = 2  # Replace confidence with visibility flag '2'
            normalized_keypoints.extend([kp_x, kp_y, kp_visibility])

        # Construct the normalized line
        normalized_line = f"{class_id} {x_center} {y_center} {width} {height} " + \
                          " ".join(map(str, normalized_keypoints)) + "\n"
        normalized_labels.append(normalized_line)

    return normalized_labels
    

def yolo_detect_export(labeled_files_dir, zip_output_path, subset, classes, export_filename, labeled):
    """
    Creates a YOLO-detect format zip file for CVAT, containing .jpg images, .txt annotations, and metadata files.
    Generates obj.names and obj.data for the dataset, organized into subsets like 'train' and 'valid'.

    Parameters:
    labeled_files_dir (str): Directory with .jpg images and their corresponding .txt annotation files.
    zip_output_path (str): Directory to save the generated zip file.
    subset (str): Dataset subset label (e.g., 'train', 'valid') for directory organization in zip.
    classes (list): List of class names for the dataset, to be included in obj.names.
    export_filename (str): Name of the output zip file.
    labeled (bool): If True, includes .txt annotation files in the zip; otherwise, only images are included.

    Output:
    - A zip file in the specified output path, organized per YOLO format for CVAT import.
    """
    # Define output paths for the structure within the zip
    images_folder = f"obj_{subset}_data"
    labels_folder = "labels"
    
    # Generate list of (img.jpg, img.txt) tuples for images with annotations
    files = [(img, img.replace('.jpg', '.txt')) for img in os.listdir(labeled_files_dir)
             if img.endswith('.jpg') and os.path.exists(os.path.join(labeled_files_dir, img.replace('.jpg', '.txt')))]

    # Path for the zip file
    zip_file_path = os.path.join(zip_output_path, export_filename)
    
    # Write the zip file
    with ZipFile(zip_file_path, 'w') as zipf:
        # Create and add obj.names
        obj_names_path = os.path.join(zip_output_path, 'obj.names')
        with open(obj_names_path, 'w') as f:
            f.write('\n'.join(classes))
        zipf.write(obj_names_path, 'obj.names')
        
        # Create and add obj.data
        obj_data_content = f"classes = {len(classes)}\nnames = obj.names\ntrain = train.txt"
        obj_data_path = os.path.join(zip_output_path, 'obj.data')
        with open(obj_data_path, 'w') as f:
            f.write(obj_data_content)
        zipf.write(obj_data_path, 'obj.data')
        
        # Add images and labels to the zip
        for img, txt in files:
            img_path = os.path.join(labeled_files_dir, img)
            txt_path = os.path.join(labeled_files_dir, txt)
            
            # Add the image to the correct subset folder
            zipf.write(img_path, os.path.join(images_folder, img))
            
            if labeled:
                # Normalize the YOLO .txt annotation and add it to the labels folder
                with open(txt_path, 'r') as f:
                    labels = f.readlines()
                
                # Get image dimensions
                with Image.open(img_path) as im:
                    img_width, img_height = im.size
                
                # Normalize labels
                normalized_labels = normalize_bbox_labels(labels, img_width, img_height)
                
                # Write normalized annotations
                normalized_txt_content = ''.join(normalized_labels)
                zipf.writestr(os.path.join(images_folder, txt), normalized_txt_content)
        
        # Generate subset file list (e.g., train.txt)
        subset_file_path = os.path.join(zip_output_path, f"{subset}.txt")
        with open(subset_file_path, 'w') as f:
            for img, _ in files:
                f.write(f"{images_folder}/{img}\n")
        zipf.write(subset_file_path, f"{subset}.txt")
    
    # Cleanup: Remove temporary files
    os.remove(obj_names_path)
    os.remove(obj_data_path)
    os.remove(subset_file_path)

    print(f"CVAT-compatible YOLO detect zip created at {zip_file_path}")


def normalize_bbox_labels(labels, img_width, img_height):
    """
    Normalizes bounding box labels in YOLO format to be relative to the image size.

    Parameters:
    labels (list): List of annotation lines from .txt file.
    img_width (int): Image width.
    img_height (int): Image height.

    Returns:
    list: List of normalized label strings in YOLO format.
    """
    normalized_labels = []
    for label in labels:
        parts = label.strip().split()
        class_id = parts[0]
        x_center = float(parts[1]) / img_width
        y_center = float(parts[2]) / img_height
        width = float(parts[3]) / img_width
        height = float(parts[4]) / img_height
        normalized_label = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
        normalized_labels.append(normalized_label)
    
    return normalized_labels