import cv2
import os
from tqdm import tqdm

def resize_and_save_images(input_folder, output_folder, target_size=(180, 180)):
    """
    Resize images in the specified input folder and save them to the output folder with a consistent size.

    Parameters:
    - input_folder: Folder containing the original images.
    - output_folder: Folder where the resized images will be saved.
    - target_size: Tuple specifying the target width and height of the images.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for img_file in tqdm(os.listdir(input_folder)):
        img_path = os.path.join(input_folder, img_file)
        
        # Check if the file is an image
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

            # Resize the image
            resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

            # Save the resized image to the output folder
            output_path = os.path.join(output_folder, img_file)
            cv2.imwrite(output_path, resized_img)

# Example usage
input_folder = r'C:\Users\FBRAC\Projects\FredSeniorExperiment\Senior-Experience-Project\ARCard\Templates'
output_folder = r'C:\Users\FBRAC\Projects\FredSeniorExperiment\Senior-Experience-Project\ARCard\ResizedTemplates'
resize_and_save_images(input_folder, output_folder)
