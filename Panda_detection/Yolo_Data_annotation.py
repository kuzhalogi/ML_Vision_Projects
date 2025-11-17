import os
import cv2 
from typing import List, Tuple, Union

# --- Utility Functions (for a complete, runnable file) ---

def get_image_dimensions(image_path: str) -> Tuple[int, int]:
    """
    Reads an image using OpenCV and returns its width and height.

    Args:
        image_path: The file path to the image.

    Returns:
        A tuple (width, height) of the image dimensions.

    Raises:
        FileNotFoundError: If the image is not found or cannot be read.
    """
    img = cv2.imread(image_path)
    
    # Check if the image was successfully loaded
    if img is None:
        raise FileNotFoundError(f"Image not found or unable to read: {image_path}")
        
    # Get the dimensions (OpenCV stores shape as (height, width, channels))
    height, width = img.shape[:2]
    
    return width, height


def parse_file_to_list(file_path: str) -> List[List[Union[str, float]]]:
    """
    Reads a file line by line, parsing the first element as a string label 
    and the rest as a list of floats (coordinates).
    
    Returns:
        A list of lists, where each inner list is [label (str), coord1 (float), ...].
    """
    data_list = []
    
    with open(file_path, 'r') as file:
        for line in file:
            # Strip leading/trailing whitespace and split the line by spaces
            parts = line.strip().split()
            
            # Only process lines that contain at least a label and coordinates
            if len(parts) > 1:
                label = parts[0]
                
                # Convert all remaining parts into a list of floats
                try:
                    coordinates = list(map(float, parts[1:]))
                except ValueError:
                    print(f"Warning: Skipping line in {file_path} due to non-float coordinate values.")
                    continue

                # Append the label (string) followed by the coordinates (floats)
                data_list.append([label] + coordinates)
                
    return data_list


def convert_bbox_to_yolo_format(x_min: float, y_min: float, x_max: float, y_max: float, image_width: int, image_height: int) -> Tuple[float, float, float, float]:
    """
    Converts absolute pixel coordinates (x_min, y_min, x_max, y_max) to 
    normalized YOLO format (x_center, y_center, width, height).
    """
    # 1. Calculate the absolute center coordinates, width, and height in pixels
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    # 2. Normalize coordinates and dimensions by the image size (scales to [0.0, 1.0])
    x_center /= image_width
    y_center /= image_height
    width /= image_width
    height /= image_height

    return (x_center, y_center, width, height)


# --- Main Logic Function ---

def process_folder(input_folder: str, output_folder: str) -> None:
    """
    Processes old-format bounding box annotations in a subfolder ('oldlabel') 
    and converts them to normalized YOLO format, saving them to the output folder.

    Args:
        input_folder: The root folder containing the image files and the 'oldlabel' subfolder.
        output_folder: The destination folder for the new YOLO annotations.
    """
    old_label_path = os.path.join(input_folder, 'oldlabel')
    
    # Check if the source label folder exists
    if not os.path.isdir(old_label_path):
        print(f"Error: Source label folder not found: {old_label_path}")
        return

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all files in the 'oldlabel' directory
    for filename in os.listdir(old_label_path):
        if filename.lower().endswith('.txt'):
            
            # image filename matches the txt filename except for extension
            image_filename = filename.replace('.txt', '.jpg') # images are JPG
            image_path = os.path.join(input_folder, image_filename)
            label_file_path = os.path.join(old_label_path, filename)

            # 1. Check for corresponding image file
            if not os.path.exists(image_path):
                print(f"Warning: Image file not found for annotation: {image_path}. Skipping.")
                continue

            # 2. Get image dimensions (for normalization)
            try:
                image_width, image_height = get_image_dimensions(image_path)
            except FileNotFoundError as e:
                print(e)
                continue
            except cv2.error as e:
                print(f"Error reading image {image_path} with OpenCV: {e}. Skipping.")
                continue

            # 3. Parse the old label file contents
            contents = parse_file_to_list(label_file_path)
            
            # 4. Prepare output file path
            txt_filename = os.path.splitext(image_filename)[0] + '.txt'
            txt_path = os.path.join(output_folder, txt_filename)

            # 5. Process and write normalized bounding boxes
            with open(txt_path, 'w') as txt_file:
                for content in contents:
                    # Unpack the content: [label, x_min, y_min, x_max, y_max]
                    if len(content) < 5:
                        print(f"Warning: Line in {label_file_path} skipped (incorrect number of elements).")
                        continue
                        
                    label, x_min, y_min, x_max, y_max = content
                    
                    # Convert to normalized YOLO format
                    normalized_bbox = convert_bbox_to_yolo_format(
                        float(x_min), float(y_min), float(x_max), float(y_max), 
                        image_width, image_height
                    )
                    
                    # Hardcoded class '0' (as requested) followed by normalized coordinates
                    panda_class = '0' 
                    
                    # Write in YOLO format: <class_id> <x_center> <y_center> <width> <height>
                    txt_file.write(f"{panda_class} {normalized_bbox[0]:.6f} {normalized_bbox[1]:.6f} {normalized_bbox[2]:.6f} {normalized_bbox[3]:.6f}\n")
    
    print(f"Successfully processed files in '{old_label_path}' and saved new annotations to '{output_folder}'.")


# --- Execution Block using user-provided paths ---

base_path = '/content/drive/MyDrive/ColabNotebooks/CV/Panda_dataset/images'
output_base_path = '/content/drive/MyDrive/ColabNotebooks/CV/Panda_dataset/yolo_labels'
folders = ['train', 'val', 'test']

if __name__ == "__main__":
    print("Starting YOLO annotation conversion...")
    
    # Loop through 'train', 'val', 'test' folders
    for folder in folders:
        input_dir = os.path.join(base_path, folder)
        output_dir = os.path.join(output_base_path, folder)
        
        print(f"\n--- Processing dataset split: {folder} ---")
        process_folder(input_dir, output_dir)

    print("\nAnnotation conversion finished for all specified folders.")