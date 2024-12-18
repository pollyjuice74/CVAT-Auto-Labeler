import os
import xml.etree.ElementTree as ET

def convert_xml_to_yolo(xml_file, output_dir):
    """ Writes .txt files with labelings from the xml_file to the output directory. """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    for image in root.findall('image'):
        file_name = image.get('name')
        image_width = int(image.get('width'))
        image_height = int(image.get('height'))
        boxes = []
        
        for box in image.findall('box'):
            label = box.get('label')
            class_id = 0  # Assuming Bee class, class ID is 0
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))
            
            # Convert to YOLO format (normalized)
            x_center = (xtl + xbr) / 2 / image_width
            y_center = (ytl + ybr) / 2 / image_height
            width = (xbr - xtl) / image_width
            height = (ybr - ytl) / image_height
            
            boxes.append(f"{class_id} {x_center} {y_center} {width} {height}")

        print(f"{file_name}: {boxes}")
        
        # Write .txt file if there are annotations
        if boxes:
            # Save to a .txt file with the same name as the image
            txt_file_path = os.path.join(output_dir, file_name.replace('.jpg', '.txt'))
            with open(txt_file_path, 'w') as f:
                f.write('\n'.join(boxes))