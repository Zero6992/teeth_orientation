import os
import pandas as pd
from PIL import Image

def makeImgLandscape(image):
    # Check if the image is in portrait orientation (height > width)
    if image.height > image.width:
        # Swap width and height to make it landscape
        image = image.transpose(Image.Transpose.ROTATE_90)

    return image

def getFaceFromCsv(imageName, imageClassificationDf):
    # Remove file extension from imageName
    imageName_base, _ = os.path.splitext(imageName)

    for index, row in imageClassificationDf.iterrows():
        currentImageName = row.iloc[0]  # Assuming the image filename is in the first column

        # Remove file extension from currentImageName
        currentImageName_base, _ = os.path.splitext(currentImageName)

        if imageName_base == currentImageName_base:
            label = row.iloc[1]  # Assuming the label is in the second column
            return label
        
    # else return none
    return None

# Path to the 'result' folder
result_folder_path = './result'
output_folder_path = './orientation'

for folder_name in os.listdir(result_folder_path):
    folder_path = os.path.join(result_folder_path, folder_name)

    if os.path.isdir(folder_path):
        sample_folder_path = os.path.join(folder_path, 'sample')
        mask_folder_path = os.path.join(folder_path, 'mask')
        csv_file_path = os.path.join(folder_path, 'imageClassification.csv')

        # Skip first line as the data is 2D or 3D
        # also add custom labels
        df = pd.read_csv(csv_file_path, skiprows=1, names=['image', 'angle'])

        if os.path.exists(sample_folder_path):
            # Iterate over image files in the 'sample' folder
            for image_filename in os.listdir(sample_folder_path):
                image_path = os.path.join(sample_folder_path, image_filename)

                # Check if it's a file and has a valid image extension
                if os.path.isfile(image_path) and image_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image = Image.open(image_path)

                    # Print image information
                    print(f"Processing image: {image_filename}, Size: {image.size}")
                    label = getFaceFromCsv(image_filename, df)
                    print(f"Image: {image_filename}, Label: {label}")

                    # if no label the skip
                    if label is None:
                        continue

                    # make image landscape
                    image = makeImgLandscape(image)

                    # process image
                    if label != 'Face':
                        # if not face then it is mirrored
                        image = image.transpose(Image.FLIP_LEFT_RIGHT)

                    # Create the output folder structure mirroring the input structure
                    output_subfolder_path = os.path.join(output_folder_path, folder_name, 'sample')
                    os.makedirs(output_subfolder_path, exist_ok=True)

                    # Save the processed image to the 'orientation' folder
                    output_image_path = os.path.join(output_subfolder_path, image_filename)
                    image.save(output_image_path)


        if os.path.exists(mask_folder_path):
            # Iterate over image files in the 'sample' folder
            for image_filename in os.listdir(mask_folder_path):
                image_path = os.path.join(mask_folder_path, image_filename)

                # Check if it's a file and has a valid image extension
                if os.path.isfile(image_path) and image_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image = Image.open(image_path)

                    image_filename_without_prefix = image_filename.replace("mask_", "")
                    print(f"Processing image: {image_filename}, Size: {image.size}")
                    label = getFaceFromCsv(image_filename_without_prefix, df)
                    print(f"Image: {image_filename}, Label: {label}")

                    # if no label the skip
                    if label is None:
                        continue

                    # make image landscape
                    image = makeImgLandscape(image)

                    # process image
                    if label != 'Face':
                        # if not face then it is mirrored
                        image = image.transpose(Image.FLIP_LEFT_RIGHT)

                    # Create the output folder structure mirroring the input structure
                    output_subfolder_path = os.path.join(output_folder_path, folder_name, 'mask')
                    os.makedirs(output_subfolder_path, exist_ok=True)

                    # Save the processed image to the 'orientation' folder
                    output_image_path = os.path.join(output_subfolder_path, image_filename)
                    image.save(output_image_path)