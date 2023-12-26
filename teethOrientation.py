import os
import pandas as pd
from PIL import Image
import csv
import numpy as np
from matplotlib.pyplot import *

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
        node_folder_path = os.path.join(folder_path, 'node')

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

                    csv_path = os.path.join(node_folder_path, f"node_{image_filename[:-3]}csv")
                    #print(csv_path)
                    readFile = open(csv_path,mode='r',newline='')
                    csvreader = csv.reader(readFile)
                    nodeArray = []
                    for row in csvreader:
                        nArray = []
                        for item in row:
                            nArray.append(float(item))
                        nodeArray.append(nArray)

                    #print(nodeArray)
                        
                    polyLine = np.polyfit(nodeArray[0],nodeArray[1],2)
                    ##print( "type poly = ",type(polyLine) ) 
                    ##print( polyLine )

                    #plt.imshow(pltImage)
                    #plt.scatter(xArray,yArray) #draw dot

                    p = np.poly1d( polyLine )
                    #print(p)
                    
                    bounds = [0, image.width]
                    crit_points = bounds + [x for x in p.deriv().r if x.imag == 0 and bounds[0] < x.real < bounds[1]]
                    print(crit_points)

                    p1 = 0
                    p2 = 0
                    if len(crit_points) == 3:
                        if crit_points[2] + 50 > image.width:
                            p2 = image.width
                        else:
                            p2 = crit_points[2] + 50
                    else:
                        p2 = crit_points[1]

                    rangeArray = np.arange(0,image.width,0.01)
                    for i in rangeArray:
                        #print(p(i), p(p2))
                        if (p(i)-p(p2)<=0.001 and p(i)-p(p2)>=0) or (p(p2)-p(i)<=0.001 and p(p2)-p(i)>=0):
                            p1 = i
                            break
                    #print(p1,p2)
                    #print(p(p1), p(p2))
                    slope = -(p(p2)-p(p1)) / (p2-p1)
                    angle = np.rad2deg(np.arctan(slope))
                    print(slope)
                    
                    image = image.rotate(slope,expand=1)

                    # Create the output folder structure mirroring the input structure
                    output_subfolder_path = os.path.join(output_folder_path, folder_name, 'orient')
                    os.makedirs(output_subfolder_path, exist_ok=True)

                    # Save the processed image to the 'orientation' folder
                    output_image_path = os.path.join(output_subfolder_path, f"orient_{image_filename}")
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

        '''
        if os.path.exists(node_folder_path):
            for csv_filename in os.listdir(node_folder_path):
                csv_path = os.path.join(node_folder_path, csv_filename)

                # Check if it's a file and has a valid image extension
                if os.path.isfile(csv_path):
                    print(csv_path)
                    
                    readFile = open(csv_path,mode='r',newline='')
                    csvreader = csv.reader(readFile)
                    nodeArray = []
                    for row in csvreader:
                        nArray = []
                        for item in row:
                            nArray.append(float(item))
                        nodeArray.append(nArray)

                    print(nodeArray)
                        
                    polyLine = np.polyfit(nodeArray[0],nodeArray[2],2)
                    ##print( "type poly = ",type(polyLine) ) 
                    ##print( polyLine )

                    #plt.imshow(pltImage)
                    #plt.scatter(xArray,yArray) #draw dot

                    p = np.poly1d( polyLine )
                    #xArray.sort()
                    #x_base = np.linspace(0,imageWidth,imageWidth)
                    #plt.plot(x_base, p(x_base),color = 'red') #draw regression
        '''