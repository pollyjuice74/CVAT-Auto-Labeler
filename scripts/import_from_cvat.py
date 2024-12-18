import os
import zipfile
import random
import shutil  # Import shutil for rmtree
from scripts.xml2yolo import convert_xml_to_yolo  # Ensure this function is available
# from ultralytics.data.converter import convert_coco


def import_from_cvat(zip_file_path, dataset_output_dir, train_split=0.6, val_split=0.2, test_split=0.2, export_format='yolo'):
    """
    Extracts a zip file containing YOLO or CVAT formats, organizes images into train, val, test, and unlabeled folders,
    and cleans up extracted files after processing. 
    
    Parameters:
    - zip_file_path (str): Path to the zip file containing dataset in YOLO or CVAT format.
    - dataset_output_dir (str): Directory to save the organized files (train, val, test, unlabeled).
    - train_split, val_split, test_split (float): Proportions for dataset splitting.
    - box_pred (bool): If True, convert CVAT XML to YOLO format.
    - pose_pred (bool): Placeholder for pose predictions (currently not implemented).
    - export_format (str): Format of the input data ('yolo' or 'cvat').
    """
    assert train_split + val_split + test_split == 1.0, "Splits must equal 1.0"
    assert export_format in ['yolo_detect', 'yolo_pose', 'cvat'], "export_format must be 'yolo' or 'cvat'"

    # Set up extraction path within the import_zips folder
    import_zips_dir = os.path.join(dataset_output_dir, 'import_zips')
    unlabeled_dir = os.path.join(dataset_output_dir, 'unlabeled')

    # Extract the zip file to the import_zips directory
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(import_zips_dir)

    # Extract a list of tuples of the form (.jpg img path, .txt img path)
    if export_format == 'yolo_detect':
        image_paths = yolo_format_extraction(import_zips_dir, unlabeled_dir) # RECOMMENDED EXTRACTION FORMAT
        # Move files to train, val and test folders
        move_files_yolo(image_paths, dataset_output_dir, train_split, val_split, export_format)

    elif export_format == 'yolo_pose':
        image_paths = yolo_pose_format_extraction(import_zips_dir, unlabeled_dir) # YOLO pose extractor
        # Move files to train, val and test folders
        move_files_yolo_pose(image_paths, dataset_output_dir, train_split, val_split, export_format)

    elif export_format == 'cvat':
        print('*** Un-tested cvat exporter format. ****')
        image_paths = cvat_format_extraction(import_zips_dir) # Unsupported
    print(image_paths)

    clean_up(import_zips_dir, unlabeled_dir) # Remove non .zip files from import_zips dir
    
    print("Files have been organized into train, val, test folders.")


def yolo_format_extraction(import_zips_dir, unlabeled_dir):
    """ Handle YOLO format: expects obj.names, obj.data, images, and annotations in 'obj_<subset>_data' folders """
    image_paths = [] # tuple list of (.jpg path, .txt path) paths

    # Iterate over 'obj_<subset>_data' directories
    for subset in ['train', 'valid']:
        subset_folder = os.path.join(import_zips_dir, f'obj_{subset}_data', f'obj_{subset}_data')

        # Check 'obj_<subset>_data' directory
        if os.path.isdir(subset_folder):
            # Get .txt labels in subset folder
            label_filenames = [label_filename for label_filename in os.listdir(subset_folder) if label_filename.endswith('.txt')]

            # Get corresponding .jpg filepath in unlabeled directory for each .txt label located in import_zips directory
            for label_filename in label_filenames:
                jpg_filename = label_filename.replace('.txt', '.jpg')
                jpg_filepath = os.path.join(unlabeled_dir, jpg_filename)

                if os.path.exists(jpg_filepath):
                    image_paths.append((jpg_filepath, os.path.join(subset_folder, label_filename))) # list of tuples to be passed to split_and_move_files()
                
    return image_paths


def move_files_yolo(image_paths, dataset_output_dir, train_split, val_split, export_format):
    """ Splits image paths into train, val, test sets, moves images and annotations to respective folders. """
    # Create train, val, test directories and outputs paths
    train_dir, val_dir, test_dir = make_split_dirs(dataset_output_dir)

    unique_image_paths = []
    # Check if image already exists in one of train/val/test to ensure no duplicates 
    for jpg_filepath, label_filepath in image_paths:
        # Get .jpg and .txt filenames
        jpg_filename = os.path.basename(jpg_filepath)
        label_filename = os.path.basename(label_filepath)
        
        # Check if both the image and label files already exist in train/val/test folders
        if not any(
            os.path.exists(os.path.join(split_path, jpg_filename)) and 
            os.path.exists(os.path.join(split_path, label_filename))
            for split_path in [train_dir, val_dir, test_dir]
        ):
            unique_image_paths.append((jpg_filepath, label_filepath))

    split_and_place_files(unique_image_paths, 
                          train_dir, val_dir, test_dir,
                          train_split, val_split,
                          export_format)


def yolo_pose_format_extraction(import_zips_dir, unlabeled_dir):
    """Extracts YOLO pose format images and label file paths from specified directories."""
    image_paths = [] # tuple list of (.jpg path, .txt path) paths

    # Labels file path
    labels_folder_path = os.path.join(import_zips_dir, 'labels', 'train')

    # Get .txt labels in subset folder
    label_filenames = [label_filename for label_filename in os.listdir(labels_folder_path) 
                       if label_filename.endswith('.txt')]
    
    for label_filename in label_filenames:
        jpg_filename = label_filename.replace('.txt', '.jpg')
        jpg_filepath = os.path.join(unlabeled_dir, jpg_filename) # Path to .jpg file

        # If .jpg file path exists append (jpg_filepath, label_filepath) tuple
        if os.path.isfile(jpg_filepath):
            label_filepath = os.path.join(labels_folder_path, label_filename)
            image_paths.append( (jpg_filepath, label_filepath) )

    return image_paths


def move_files_yolo_pose(image_paths, dataset_output_dir, train_split, val_split, export_format):
    """ Splits image paths into train, val, test sets, moves images and annotations to respective folders. Specific for YOLO pose format. """
    # Images/labels directories
    images_output_dir = os.path.join(dataset_output_dir, 'images')
    label_output_dir = os.path.join(dataset_output_dir, 'labels')

    # Create train, val, test directories and outputs paths
    img_train_dir, img_val_dir, img_test_dir = make_split_dirs(images_output_dir)
    label_train_dir, label_val_dir, label_test_dir = make_split_dirs(label_output_dir)

    unique_paths = []
    # Check if image already exists in one of train/val/test to ensure no duplicates 
    for jpg_filepath, label_filepath in image_paths:
        # Get .jpg and .txt filenames
        jpg_filename = os.path.basename(jpg_filepath)
        label_filename = os.path.basename(label_filepath)
        
        # Check if both the image and label files already exist in img/label train/val/test folders
        if not any(
            os.path.exists(os.path.join(image_split_path, jpg_filename)) and 
            os.path.exists(os.path.join(label_split_path, label_filename))
            for image_split_path, label_split_path in [(img_train_dir, label_train_dir),
                                                       (img_val_dir, label_val_dir), 
                                                       (img_test_dir, label_test_dir)] ):
            unique_paths.append((jpg_filepath, label_filepath))
    
    random.shuffle(unique_paths) # Shuffle

    # Split the shuffled list back into two lists
    unique_image_paths, unique_label_paths = map(list, zip(*unique_paths))

    # Split images
    split_and_place_files(unique_image_paths, 
                          img_train_dir, img_val_dir, img_test_dir,
                          train_split, val_split,
                          export_format)
    # Split labels
    split_and_place_files(unique_label_paths, 
                          label_train_dir, label_val_dir, label_test_dir,
                          train_split, val_split,
                          export_format)


def make_split_dirs(dataset_output_dir):
    """ Make train, val, test, unlabeled dirs if they don't exist and return their paths. """
    train_dir = os.path.join(dataset_output_dir, 'train')
    val_dir = os.path.join(dataset_output_dir, 'val')
    test_dir = os.path.join(dataset_output_dir, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    return train_dir, val_dir, test_dir
        

def split_and_place_files(unique_image_paths, 
                          train_dir, val_dir, test_dir,
                          train_split, val_split,
                          export_format):
    """ Splits and places files into train/val/test directories. """
    # Shuffle and split labeled images into train, val, test
    n = len(unique_image_paths)
    train_images = unique_image_paths[:int(train_split * n)]
    val_images = unique_image_paths[int(train_split * n):int((train_split + val_split) * n)]
    test_images = unique_image_paths[int((train_split + val_split) * n):]

    # Move labeled files to train, val, test directories
    move_files(train_images, train_dir, export_format)
    move_files(val_images, val_dir, export_format)
    move_files(test_images, test_dir, export_format)


def move_files(image_label_paths, target_dir, export_format):
    """
    Moves image and label files to the specified target directory.
    
    Args:
        image_label_paths (list of tuples): A list of (image path, label path) tuples.
        target_dir (str): The directory to move files into.
    """
    if export_format=='yolo_detect':
        for jpg_filepath, label_filepath in image_label_paths:
            shutil.move(jpg_filepath, os.path.join(target_dir,
                                                os.path.basename(jpg_filepath)))
            
            shutil.move(label_filepath, os.path.join(target_dir, 
                                                os.path.basename(label_filepath)))
            
    elif export_format=='yolo_pose':
        for filepath in image_label_paths:
            shutil.move(filepath, os.path.join(target_dir,
                                                os.path.basename(filepath)))

        
def clean_up(import_zips_dir, unlabeled_dir):
    """ Clean up after zip extraction and remove model labeled labels from unlabeled directory. """
    # Remove all non .zip files in import_zips directory
    for item in os.listdir(import_zips_dir):
        item_path = os.path.join(import_zips_dir, item)

        # Remove all non .zip file items in import_zips directory
        if not item.endswith('.zip'):
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)  # Use shutil.rmtree to remove non-empty directories
    
    # Removes all .txt model labeled labels in unlabeled directory
    for item in os.listdir(unlabeled_dir):
        if item.endswith('.txt'):
            item_path = os.path.join(unlabeled_dir, item)    
            os.remove(item_path)


def cvat_format_extraction(import_zips_dir, box_pred):
    """ Handle CVAT format: convert XML to YOLO format if needed """
    annotations_path = os.path.join(import_zips_dir, 'annotations.xml')

    if box_pred and os.path.exists(annotations_path):
        convert_xml_to_yolo(annotations_path, import_zips_dir)
        image_paths = [
            (os.path.join(import_zips_dir, img), os.path.join(import_zips_dir, img.replace('.jpg', '.txt')))
            for img in os.listdir(import_zips_dir)
            if img.endswith('.jpg') and os.path.exists(os.path.join(import_zips_dir, img.replace('.jpg', '.txt')))
        ]
    return image_paths