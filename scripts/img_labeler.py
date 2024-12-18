from ultralytics import YOLO
import os


def label_files(model_weights_path, files_dir, label_format):
    """ Automatically labels imgs based on a given model's predictions and outputs it to the same file where the imgs are located, files_dir. """
    assert label_format in ['yolo_detect', 'yolo_pose']
    model = YOLO(model_weights_path)
    
    preds = model.predict(files_dir) # make the model pass through the data and label unseen images
    print(preds)

    for i, pred in enumerate(preds):
        # Extract the base name of the image without file extension
        img_filename = os.path.basename(pred.path).split('.')[0]

        # Define the output path for the .txt file
        txt_output_path = os.path.join(files_dir, f"{img_filename}.txt")

        # output img preds to files_dir as .txt files
        with open(txt_output_path, 'w') as f:

            # Handle box predictions
            if label_format=='yolo_detect':
                for box in pred.boxes: # Iterate over box predictions
                    # YOLO format requires: <class> <x_center> <y_center> <width> <height>
                    class_id = int(box.cls)  # Class ID
                    x_center, y_center, width, height = box.xywh[0]  # Normalized center and size
                    
                    # Write the prediction to the .txt file
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            # Handle pose predictions
            elif label_format=='yolo_pose':
                # Iterate over keypoints and bounding boxes
                for i, (box, keypoints) in enumerate(zip(pred.boxes.xywhn, pred.keypoints.data)):
                    # Format bounding box
                    class_id = int(pred.boxes.cls[i])  # Class ID
                    bbox_str = " ".join(f"{v:.6f}" for v in box)  # x_center, y_center, width, height
                    
                    # Format keypoints
                    keypoints_str = " ".join(f"{k:.6f}" for k in keypoints.flatten())  # Keypoints and visibility
                    
                    # Write to file in YOLO pose format
                    f.write(f"{class_id} {bbox_str} {keypoints_str}\n")

    print(f"Predictions saved in YOLO format to {files_dir}.")