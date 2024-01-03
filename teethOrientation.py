import os
import pandas as pd
from PIL import Image, ImageDraw
import csv
import numpy as np
from matplotlib.pyplot import *
from matplotlib import pyplot as plt

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
        #sample_folder_path = os.path.join(folder_path, 'sample')
        sample_folder_path = os.path.join(folder_path, 'sample')
        mask_folder_path = os.path.join(folder_path, 'mask')
        csv_file_path = os.path.join(folder_path, 'imageClassification.csv')
        node_folder_path = os.path.join(folder_path, 'node')
        box_folder_path = os.path.join(folder_path, 'boundingBox')

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
                    Cutimage = image.copy()

                    # Print image information
                    print(f"Processing image: {image_filename}, Size: {image.size}")
                    label = getFaceFromCsv(image_filename, df)
                    print(f"Image: {image_filename}, Label: {label}")

                    # if no label the skip
                    if label is None:
                        continue

                    imgHeight = image.height
                    imgWidth = image.width
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

                    csv_path = os.path.join(box_folder_path, f"box_{image_filename[:-3]}csv")
                    readFile = open(csv_path,mode='r',newline='')
                    csvreader = csv.reader(readFile)
                    boxArray = []
                    for row in csvreader:
                        nArray = []
                        for item in row:
                            nArray.append(float(item))
                        boxArray.append(nArray)
                    
                    minX = 100000
                    minY = 100000
                    maxX = 0
                    maxY = 0

                    for box in boxArray:
                        x1, y1, x2 ,y2 = box
                        x1 = max(int(x1), 0)
                        y1 = max(int(y1), 0)
                        x2 = min(int(x2), Cutimage.width)
                        y2 = min(int(y2), Cutimage.height)
                        minX = min(x1, minX)
                        minY = min(y1, minY)
                        maxX = max(x2, maxX)
                        maxY = max(y2, maxY)

                    minX = max(0,minX-50)
                    minY =  max(0,minY-50)
                    maxX = min(Cutimage.width, maxX+50)
                    maxY = min(Cutimage.height, maxY+50)
                    
                    if Cutimage.height < Cutimage.width:
                        if (maxY - minY) * 1.5 < (maxX - minX):
                            if (maxY + ((maxX-minX)/1.5-(maxY-minY))/2) < Cutimage.height:
                                maxY += ((maxX-minX)/1.5-(maxY-minY)) / 2
                                if (minY - ((maxX-minX)/1.5-(maxY-minY))/2) > 0:
                                    minY -= ((maxX-minX)/1.5-(maxY-minY)) / 2
                                else:
                                    maxY -= (minY - ((maxX-minX)/1.5-(maxY-minY))/2)
                                    minY = 0
                            else:
                                minY -= ((maxX-minX)/1.5-(maxY-minY)) / 2
                                minY -= ((maxY + ((maxX-minX)/1.5-(maxY-minY))/2) - Cutimage.height)
                                maxY = Cutimage.height
                        elif (maxY - minY) * 1.5 > (maxX - minX):
                            if (maxX + ((maxY-minY)*1.5-(maxX-minX))/2) < Cutimage.width:
                                maxX += ((maxY-minY)*1.5-(maxX-minX)) / 2
                                if (minX - ((maxY-minY)*1.5-(maxX-minX))/2) > 0:
                                    minX -= ((maxY-minY)*1.5-(maxX-minX)) / 2
                                else:
                                    maxX -= (minX - ((maxY-minY)*1.5-(maxX-minX))/2)
                                    minX = 0
                            else:
                                minX -= ((maxY-minY)*1.5-(maxX-minX)) / 2
                                minX -= ((maxX + ((maxY-minY)*1.5-(maxX-minX))/2) - Cutimage.width)
                                maxX = Cutimage.width
                    else:
                        if (maxX - minX) * 1.5 < (maxY - minY):
                            if (maxX + ((maxY-minY)/1.5-(maxX-minX))/2) < Cutimage.width:
                                maxX += ((maxY-minY)/1.5-(maxX-minX)) / 2
                                if (minX - ((maxY-minY)/1.5-(maxX-minX))/2) > 0:
                                    minX -= ((maxY-minY)/1.5-(maxX-minX)) / 2
                                else:
                                    maxX -= (minX - ((maxY-minY)/1.5-(maxX-minX))/2)
                                    minX = 0
                            else:
                                minX -= ((maxY-minY)/1.5-(maxX-minX)) / 2
                                minX -= ((maxX + ((maxY-minY)/1.5-(maxX-minX))/2) - Cutimage.width)
                                maxX = Cutimage.width
                        elif (maxX - minX) * 1.5 > (maxY - minY):
                            if (maxY + ((maxX-minX)*1.5-(maxY-minY))/2) < Cutimage.height:
                                maxY += ((maxX-minX)*1.5-(maxY-minY)) / 2
                                if (minY - ((maxX-minX)*1.5-(maxY-minY))/2) > 0:
                                    minY -= ((maxX-minX)*1.5-(maxY-minY)) / 2
                                else:
                                    maxY -= (minY - ((maxX-minX)*1.5-(maxY-minY))/2)
                                    minY = 0
                            else:
                                minY -= ((maxX-minX)*1.5-(maxY-minY)) / 2
                                minY -= ((maxY + ((maxX-minX)*1.5-(maxY-minY))/2) - Cutimage.height)
                                maxY = Cutimage.height
                    ''''''

                    if Cutimage.height > Cutimage.width:
                        temp = minX
                        minX = minY
                        minY = Cutimage.width - maxX
                        maxX = maxY
                        maxY = Cutimage.width - temp

                    if label != 'Face':
                        if Cutimage.height > Cutimage.width:
                            temp = minX
                            minX = Cutimage.height - maxX
                            maxX = Cutimage.height - temp
                        else:
                            temp = minX
                            minX = Cutimage.width - maxX
                            maxX = Cutimage.width - temp

                    Cutimage = makeImgLandscape(Cutimage)

                    if label != 'Face':
                        Cutimage = Cutimage.transpose(Image.FLIP_LEFT_RIGHT)
                    
                    Cutimage = Cutimage.crop((minX,minY,maxX,maxY))
                    #print("Cut [", minX, ",", minY, ",", maxX, ",", maxY,"]")

                    output_subfolder_path = os.path.join(output_folder_path, folder_name, 'crop')
                    os.makedirs(output_subfolder_path, exist_ok=True)

                    output_image_path = os.path.join(output_subfolder_path, f"crop_{image_filename}")
                    Cutimage.save(output_image_path)

                    csv_path = os.path.join(node_folder_path, f"node_{image_filename[:-3]}csv")
                    readFile = open(csv_path,mode='r',newline='')
                    csvreader = csv.reader(readFile)
                    nodeArray = []
                    for row in csvreader:
                        nArray = []
                        for item in row:
                            nArray.append(float(item))
                        nodeArray.append(nArray)

                    if imgWidth < imgHeight:
                        temp = nodeArray[0]
                        nodeArray[0] = nodeArray[1]
                        nodeArray[1] = temp
                        for i in range(len(nodeArray[1])):
                            nodeArray[1][i] = imgWidth - nodeArray[1][i]
                    
                    if label != 'Face':
                        for i in range(len(nodeArray[0])):
                            nodeArray[0][i] = image.width - nodeArray[0][i]
                    ''''''
                        
                    for i in range(len(nodeArray[0])):
                        nodeArray[0][i] = nodeArray[0][i] - minX
                        nodeArray[1][i] = nodeArray[1][i] - minY
                    
                    polyLine = np.polyfit(nodeArray[0],nodeArray[1],2)

                    p = np.poly1d( polyLine )
                    #print(p)
                    
                    bounds = [0, Cutimage.width-1]
                    crit_points = bounds + [x for x in p.deriv().r if x.imag == 0 and bounds[0] < x.real < bounds[1]]
                    #print(crit_points)

                    p1 = 0
                    p2 = 0
                    if len(crit_points) == 3:
                        if crit_points[2] + 50 > Cutimage.width-1:
                            p2 = Cutimage.width-1
                        else:
                            p2 = crit_points[2] + 50
                        
                        if p(crit_points[2]) > p(p2):
                            if label == 'Up':
                                Cutimage = Cutimage.transpose(Image.FLIP_TOP_BOTTOM)
                                p = -p + Cutimage.height
                                for i in range(len(nodeArray[1])):
                                    nodeArray[1][i] = Cutimage.height - nodeArray[1][i]
                        else:
                            if label != 'Up':
                                Cutimage = Cutimage.transpose(Image.FLIP_TOP_BOTTOM)
                                p = -p + Cutimage.height
                                for i in range(len(nodeArray[1])):
                                    nodeArray[1][i] = Cutimage.height - nodeArray[1][i]
                        ''''''
                        rangeArray = np.arange(0,Cutimage.width-1,0.001)
                        for i in rangeArray:
                            if (p(i)-p(p2)<=0.001 and p(i)-p(p2)>=0) or (p(p2)-p(i)<=0.001 and p(p2)-p(i)>=0):
                                p1 = i
                                break
                    else:
                        p2 = crit_points[1]
                        p1 = crit_points[1] - 10
                        if p(crit_points[1]) > p(crit_points[0]):
                            if label == 'Up':
                                Cutimage = Cutimage.transpose(Image.FLIP_TOP_BOTTOM)
                                p = -p + Cutimage.height
                                for i in range(len(nodeArray[1])):
                                    nodeArray[1][i] = Cutimage.height - nodeArray[1][i]
                        else:
                            if label != 'Up':
                                Cutimage = Cutimage.transpose(Image.FLIP_TOP_BOTTOM)
                                p = -p + Cutimage.height
                                for i in range(len(nodeArray[1])):
                                    nodeArray[1][i] = Cutimage.height - nodeArray[1][i]
                        ''''''
                    
                    
                    #print("p1 = ", p1, " -> ", p(p1))
                    #print("p2 = ", p2, " -> ", p(p2))
                    slope = (p(p2)-p(p1)) / (p2-p1)
                    angle = np.rad2deg(slope)#np.arctan(slope)
                    #print(slope)

                    npimg = np.array(Cutimage.copy(), dtype=np.uint8)
                    pltImage = np.array(Cutimage)

                    print(angle)
                    Cutimage = Cutimage.rotate(angle,expand=1)

                    output_subfolder_path = os.path.join(output_folder_path, folder_name, 'rotate')
                    os.makedirs(output_subfolder_path, exist_ok=True)

                    output_image_path = os.path.join(output_subfolder_path, f"rotate_{image_filename}")
                    Cutimage.save(output_image_path)

                    plt.imshow(pltImage)
                    plt.scatter(nodeArray[0],nodeArray[1])
                    plt.scatter([p1,p2],[p(p1),p(p2)])

                    x_base = np.linspace(0,Cutimage.width,Cutimage.width)
                    plt.plot(x_base, p(x_base),color = 'red')

                    output_subfolder_path = os.path.join(output_folder_path, folder_name, 'line')
                    os.makedirs(output_subfolder_path, exist_ok=True)

                    output_image_path = os.path.join(output_subfolder_path, f"line_{image_filename}")
                    plt.savefig(output_image_path)
                    plt.clf()



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

        