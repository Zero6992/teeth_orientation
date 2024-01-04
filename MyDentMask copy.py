import torch
import os
from tqdm import tqdm
import time

import PIL.Image
import PIL.ImageOps

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from maskrcnn import my_get_model_instance_segmentation
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.image as mpimg
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import collections
import math
###### 標號 ######
from operator import itemgetter

import cv2
import csv

import json
def tensor2image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor.permute(1, 2, 0), dtype=np.uint8)
    return Image.fromarray(tensor)

def absoluteToScale(teethLocation,imageWidth,imageHeight):
    # x,y,w,h
    x1 = teethLocation.x1
    x2 = teethLocation.x2
    y1 = teethLocation.y1
    y2 = teethLocation.y2
    return [ ((x1+x2)/2)/imageWidth,((y1+y2)/2)/imageHeight,((x2-x1))/imageWidth,((y2-y1))/imageHeight ]

class ImageFile:
    def __init__(self,FileName):
        self.fileName = FileName
        self.is3D = True
        self.photoImageSet = []
        self.missingLabelId = []

class PhotoImage:
    def __init__(self,imageName,gradient,teethNum,grayData,teethLocationSet,width,height,image,imageTeethNodeSet, p,polyLine):   
        self.imageName = imageName
        self.gradient = gradient
        self.absGradient = abs(gradient)
        self.gradientRank = -1
        self.teethNum = teethNum
        self.teethNumRank = -1
        self.image = image
        self.grayData = grayData
        self.useFlag = False
        self.teethLocationSet = teethLocationSet
        self.width = width
        self.height = height
        self.teethScaleSet = [absoluteToScale(teethLocation,width,height) for teethLocation in teethLocationSet] #x,y,w,h
        self.view = 'Unknown'
        self.teethNodeSet = imageTeethNodeSet
        self.polyLine = polyLine
        #### mark ####
        self.regression = p
        
class TeethNode:
    def __init__(self,mask,box):   
        self.mask = mask.astype(np.uint8)
        self.box = box
        self.labelId = -1

class TeethLocation:
    def __init__(self,x1,y1,x2,y2):   
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

def exif_transpose(img):
    if not img:
        return img

    exif_orientation_tag = 274

    # Check for EXIF data (only present on some files)
    if hasattr(img, "_getexif") and isinstance(img._getexif(), dict) and exif_orientation_tag in img._getexif():
        exif_data = img._getexif()
        orientation = exif_data[exif_orientation_tag]

        # Handle EXIF Orientation
        if orientation == 1:
            # Normal image - nothing to do!
            pass
        elif orientation == 2:
            # Mirrored left to right
            img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            # Rotated 180 degrees
            img = img.rotate(180)
        elif orientation == 4:
            # Mirrored top to bottom
            img = img.rotate(180).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 5:
            # Mirrored along top-left diagonal
            img = img.rotate(-90, expand=True).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 6:
            # Rotated 90 degrees
            img = img.rotate(-90, expand=True)
        elif orientation == 7:
            # Mirrored along top-right diagonal
            img = img.rotate(90, expand=True).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 8:
            # Rotated 270 degrees
            img = img.rotate(90, expand=True)

    return img

def sortGradient(infoSet):
    infoSet = sorted(infoSet,key = lambda x : x.absGradient,reverse = True)
    for i in range( len(infoSet) ):
        infoSet[ i ].gradientRank = i+1

def sortTeeth(infoSet):
    infoSet = sorted(infoSet,key = lambda x : x.teethNum,reverse = True)
    for i in range( len(infoSet) ):
        infoSet[ i ].teethRank = i+1

def checkFlag3D(oriPltImage,resizeScale,proportion):
    pltImage = cv2.resize(oriPltImage, dsize=(int(imageWidth*resizeScale),int(imageHeight*resizeScale)), interpolation=cv2.INTER_CUBIC)
    hRGB = len(pltImage)
    wRGB = len(pltImage[0])
    totalPixel = hRGB * wRGB


    white = (255,255,255)
    mask = cv2.inRange(pltImage,white,white)
    whitePixelCnt = cv2.countNonZero(mask)

    if whitePixelCnt  < totalPixel*proportion :
        return False
    else:
        return True

def pilSave(image,path,fileLabel,prefix,imageName):
    check = imageName.split('.')
    if prefix != "" :
        prefix += "_"
    if len(check[-1]) < 4 :    
        image.save(f"{path}/{fileLabel}/{prefix}{imageName[:-3]}png")
    else:  #jpeg
        image.save(f"{path}/{fileLabel}/{prefix}{imageName[:-4]}png")

def pltSave(path,fileLabel,prefix,imageName):
    check = imageName.split('.')
    if prefix != "" :
        prefix += "_"
    if len(check[-1]) < 4 :    
        plt.savefig(f"{path}/{fileLabel}/{prefix}{imageName[:-3]}png")
    else:  #jpeg
        plt.savefig(f"{path}/{fileLabel}/{prefix}{imageName[:-4]}png")

def makeResultDirProcess(fileName):
    redir = 'result'
    sampleDir = 'sample'
    file = os.path.join(redir)
    os.makedirs(file,exist_ok=True)
    file = os.path.join(sampleDir)
    os.makedirs(file,exist_ok=True)
    file = os.path.join(redir,  fileName)
    os.makedirs(file,exist_ok=True)
    file = os.path.join(sampleDir,  fileName)
    os.makedirs(file,exist_ok=True)
    createFile = ['det','regression','sample','mask','boundingBox','node']
    for name in createFile:
        tmp = os.path.join(redir, fileName, name)
        os.makedirs(tmp,exist_ok=True)

def getBoundingBoxes(model,threshold,imageInfo):
    
    torchImage = torch.as_tensor(np.array(imageInfo.image)[...,:3]/255, dtype=torch.float32).permute(2,0,1).unsqueeze(0)
    torchImage = torchImage.to(device)
    output = model(torchImage)[0]

    scores = output["scores"]
    boxes = output["boxes"]
    masks = output["masks"]
    classes = output["labels"]
    zippedData = zip(boxes,masks, scores,classes)
    zippedData = sorted(zippedData,key=lambda x:x[2],reverse=True)
    # if len(zippedData) > 2:
    #     zippedData = zippedData[:2]
    retBoxes = []

    leftBoxes = []
    rightBoxes = []
    leftScores = []
    rightScores = []
    for box,mask, score,label in zippedData:
        #print(score.item())
        if score.item() > threshold:
            box = [b.item() for b in box]
            x1, y1, x2 ,y2 = box
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            # print(type(score.item()))
            # print(score.item())
            if (x1+x2)/2 < (imageInfo.width/2): #left
                leftBoxes.append(TeethLocation(x1,y1,x2,y2))
                leftScores.append(score.item())
            else:
                #print(x1,y1,x2,y2)
                #print('score = ',score.item())
                rightBoxes.append(TeethLocation(x1,y1,x2,y2))
                rightScores.append(score.item())
    #找左右,y軸最上or最下的牙齒
    # if imageInfo.view == 'Up':
    #     if len(leftBoxes) > 0:
    #         retBoxes.append(max(leftBoxes, key=lambda box: (box[0].y1+box[0].y2)/2  ))
    #     if len(rightBoxes) > 0:
    #         retBoxes.append(max(rightBoxes, key=lambda box: (box[0].y1+box[0].y2)/2))
    # elif imageInfo.view == 'Below':
    #     if len(leftBoxes) > 0:
    #         retBoxes.append(min(leftBoxes, key=lambda box: (box[0].y1+box[0].y2)/2  ))
    #     if len(rightBoxes) > 0:
    #         retBoxes.append(min(rightBoxes, key=lambda box: (box[0].y1+box[0].y2)/2))
    # else:
    #     print('590 line error!!!!')


    #找左右分數最大，各一顆
    if len(leftBoxes) > 0:
        retBoxes.append(  (max(zip(leftBoxes,leftScores), key=lambda box: box[1] ))[0]  )
    if len(rightBoxes) > 0:
        retBoxes.append(  (max(zip(rightBoxes,rightScores), key=lambda box: box[1]))[0] )
    
    drawImage = Image.fromarray(imageInfo.image.copy())
    draw = ImageDraw.Draw(drawImage)
    for box in retBoxes:
        x1 = box.x1
        x2 = box.x2
        y1 = box.y1
        y2 = box.y2
        detLineScale = 0.005 #det line width scale
        draw.line([(x1,y1),(x2,y1),(x2,y2),(x1,y2),(x1,y1)], fill=color, width=int(imageInfo.width*detLineScale))

    pilSave(drawImage,f"./result/{fileName}","det","det_upLowerSix",imageInfo.imageName)

    return retBoxes



if __name__ == "__main__":
    print("Cuda is available = ",torch.cuda.is_available())
    
    #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2048"
    isGPU = True
    isLabel = False
    if isGPU:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print("device -> ",device)
    model = my_get_model_instance_segmentation(2)

    # load weight from ./ckpts/
    param_dict = torch.load("./ckpts/label_two_class_best_model.pth", device)
    model.load_state_dict(param_dict)
    model.to(device)
    model.eval()

    


    # Test 1 image
    # image_path = "IMG_7627.JPG"

    # Test all image
    # path setting
    # load every file in ./dataset/Sample
    root = 'dataset'
    samdir = 'Sample'

    # load direction
    imgdir = list(sorted(os.listdir(os.path.join(root, samdir))))
    imgs = {} 

    print(imgdir)

    for fileName in imgdir:
        imgSet = list(sorted(os.listdir(os.path.join(root, samdir, fileName))))
        print(imgSet)

        imgs[ fileName ] = imgSet
    
    # bounding box threshold
    bb_thre = 0.80
    # mask threshold
    mask_thre = 0.55

    # ubuntu find font from path
    # fnt = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", size=50)

    # put every image in path and show result in corresponding folder
    # det means detection, seg means segmentation
    pbar = tqdm( total = 100 ) 
    print( "\nfileNum = ", len(imgs) )

    for fileName,imgSet in imgs.items(): 
        print( '\nProcess : ',fileName )
        makeResultDirProcess(fileName)

        imageFileInfo = ImageFile(fileName)

        imageInfoSet = []
        labelLoc = {}

        for imageName in imgSet:
            print(imageName,' => DETECT teeth ')

            #pltImage = mpimg.imread(f"./{root}/{samdir}/{fileName}/{imageName}")
            image = Image.open(f"./{root}/{samdir}/{fileName}/{imageName}")
            image = exif_transpose(image)
            
            if isGPU:
                idealSize = 1024
                image.thumbnail((idealSize,idealSize))

            pltImage = np.array(image)
            
            pilSave(image,f"./result/{fileName}","sample","",imageName)


            imageWidth = image.width
            imageHeight = image.height


            #check = imageName.split('.')

            images = torch.as_tensor(np.array(image)[...,:3]/255, dtype=torch.float32).permute(2,0,1).unsqueeze(0)
            images = images.to(device)
            output = model(images)[0]

            image = tensor2image(images[0].cpu())
            img = image.copy()

            # print("a:size",img.size)
            # print("a:type",type(img.size))
            black_img = Image.new('RGB', img.size, (0, 0, 0))
            draw = ImageDraw.Draw(img)
            scores = output["scores"]
            boxes = output["boxes"]
            masks = output["masks"]

            #print( "img type = " ,type(img) )
            #print( "draw type = ",type(draw) )
            

            # processing bounding box
            xList = []
            yList = []

            teethCnt = 0

            teethLocationSet = []
            imageTeethNodeSet = []
            
            writeFile = open(f"./result/{fileName}/boundingBox/box_{imageName[:-3]}csv",mode="w",newline="")
            csvWriter = csv.writer(writeFile)
                    
            npimg = np.array(image.copy(), dtype=np.uint8)
            npblack_img = np.array(black_img, dtype=np.uint8)
            for box, mask, score in zip(boxes, masks, scores):
                if score.item() > bb_thre:
                    #print('box score',score.item())
                    teethCnt += 1 

                    box = [b.item() for b in box]
                    x1, y1, x2 ,y2 = box
                    x1 = max(int(x1), 0)
                    y1 = max(int(y1), 0)
                    x2 = min(int(x2), imageWidth)
                    y2 = min(int(y2), imageHeight)
                    #print('detect box = ',x1,y1,x2,y2)
                    # x1 = int(x1)
                    # y1 = int(y1)
                    # x2 = int(x2)
                    # y2 = int(y2)

                    teethLocationSet.append( TeethLocation(x1,y1,x2,y2) )
                    csvWriter.writerow([x1,y1,x2,y2])
                    
                    xList.append( (x1+x2)/2 )
                    yList.append( (y1+y2)/2 )

                    mask = mask.detach().squeeze().cpu().numpy()
                    mask = np.where(mask > mask_thre, 255, 0).astype(np.uint8)
                    imageTeethNodeSet.append(TeethNode(mask,TeethLocation(x1,y1,x2,y2)))
                    color = list(np.random.choice(range(256), size=3))
                    npimg[np.where(mask>0)] = color
                    npblack_img[np.where(mask>0)] = color

                    #draw.ellipse(((x1+x2)/2 -50,(y1+y2)/2 -50,(x1+x2)/2 +50,(y1+y2)/2 +50), fill=(255,0,0,255))

                    color = tuple(np.random.choice(range(256), size=3))
                    detLineScale = 0.005 #det line width scale
                    draw.line([(x1,y1),(x2,y1),(x2,y2),(x1,y2),(x1,y1)], fill=color, width=int(imageWidth*detLineScale))
                    # draw.text((x1,y1), f"{score.item():.4f}", font=fnt) # draw confidence
            
            writeFile.close()

            writeFile = open(f"./result/{fileName}/node/node_{imageName[:-3]}csv",mode="w",newline="")
            csvWriter = csv.writer(writeFile)

            csvWriter.writerow(xList)
            csvWriter.writerow(yList)

            writeFile.close()

            xArray = np.array(xList)
            yArray = np.array(yList)

            plt.xlim( 0,imageWidth )
            plt.ylim( imageHeight,0 )
            polyLine = np.polyfit(xArray,yArray,2)
            #print( "type poly = ",type(polyLine) ) 
            #print( polyLine )

            plt.imshow(pltImage)
            plt.scatter(xArray,yArray) #draw dot

            p = np.poly1d( polyLine )
            xArray.sort()
            x_base = np.linspace(0,imageWidth,imageWidth)
            plt.plot(x_base, p(x_base),color = 'red') #draw regression

            pltSave(f"./result/{fileName}","regression","regression",imageName)
            plt.clf()

            imageGray = cv2.cvtColor(pltImage,cv2.COLOR_BGR2GRAY)

            imageInfo = PhotoImage(imageName,polyLine[0],teethCnt,imageGray,teethLocationSet,imageWidth,imageHeight,pltImage,imageTeethNodeSet, p,polyLine)
            imageInfoSet.append(imageInfo)

            resizeScale = 0.1
            whiteProportion = 1/100
            if checkFlag3D(pltImage,resizeScale,whiteProportion) == False:
                imageFileInfo.is3D = False
            
            pilSave(img,f"./result/{fileName}","det","det",imageName)

            img = Image.fromarray(npimg)
            black_img = Image.fromarray(npblack_img)
            
            pilSave(black_img,f"./result/{fileName}","mask","mask",imageName)

            # exit()

        writeFile = open(f"./result/{fileName}/imageClassification.csv",mode="w",newline="")
        csvWriter = csv.writer(writeFile)

        sortGradient(imageInfoSet)
        sortTeeth(imageInfoSet)

        #判斷面觀

        if imageFileInfo.is3D :
            csvWriter.writerow(["3D"])
        else:
            csvWriter.writerow(["2D"])

        for i in range( len(imageInfoSet) ):
            info = imageInfoSet[i]
            if info.teethRank == 1 and info.gradientRank >= 3 and info.useFlag == False:
                csvWriter.writerow([imageInfoSet[i].imageName,"Face"])
                imageInfoSet[i].view = 'Face'
                imageInfoSet[i].useFlag = True
                break
        
        for i in range( len(imageInfoSet) ):
            info = imageInfoSet[i]
            if info.teethRank != 1 and info.gradientRank <= 2 and info.useFlag == False:
                if info.gradient >= 0:
                    imageInfoSet[i].view = 'Up'
                    csvWriter.writerow([imageInfoSet[i].imageName,"Up"])
                else :
                    imageInfoSet[i].view = 'Below'
                    csvWriter.writerow([imageInfoSet[i].imageName,"Below"])
                imageInfoSet[i].useFlag = True

        leftRightCnt = 0
        for i in range( len(imageInfoSet) ):
            if imageInfoSet[i].useFlag == False:
                leftRightCnt += 1

        if leftRightCnt !=2:
            print("~~~!!! Classification Error !!!~~~")
        else:
            if imageFileInfo.is3D :
                for i in range( len(imageInfoSet) ):
                    if imageInfoSet[i].useFlag == False:
                        info = imageInfoSet[i]
                        imageInfoSet[i].useFlag = True
                        info.teethLocationSet = sorted(info.teethLocationSet,key = lambda loc : (loc.x1+loc.x2)/2)

                        teethCheckNum = 3
                        leftCnt = 0 #check width/height
                        for k in range(teethCheckNum):
                            leftCnt +=  ( (info.teethLocationSet[k].x2 - info.teethLocationSet[k].x1)/(info.teethLocationSet[k].y2 - info.teethLocationSet[k].y1) )

                        rightCnt = 0
                        for k in range(info.teethNum-teethCheckNum,info.teethNum):
                            rightCnt += ( (info.teethLocationSet[k].x2 - info.teethLocationSet[k].x1)/(info.teethLocationSet[k].y2 - info.teethLocationSet[k].y1) )
                        
                        if leftCnt > rightCnt:
                            imageInfoSet[i].view = 'Right'
                            csvWriter.writerow([info.imageName,"Right"])
                        else:
                            imageInfoSet[i].view = 'Left'
                            csvWriter.writerow([info.imageName,"Left"])
            else :
                for i in range( len(imageInfoSet) ):
                    if imageInfoSet[i].useFlag == False:
                        info = imageInfoSet[i]
                        imageInfoSet[i].useFlag = True

                        oriH = len(info.grayData)
                        oriW = len(info.grayData[0])
                        
                        resizeScale = 0.1
                        grayDataSmall = cv2.resize(info.grayData, dsize=(int(oriW*resizeScale),int(oriH*resizeScale)), interpolation=cv2.INTER_CUBIC)
                        
                        h = len(grayDataSmall)
                        w = len(grayDataSmall[0])

                        w3 = int(w/3)
                        leftCnt = 0
                        for row in range(h):
                            for col in range(w3):
                                leftCnt += grayDataSmall[row][col]
                        leftAverage = leftCnt / (h*w3)

                        rightCnt = 0
                        for row in range(h):
                            for col in range(w3+w3,w):
                                rightCnt += grayDataSmall[row][col]
                        rightAverage = rightCnt / (h*(w-w3-w3))

                        if leftAverage > rightAverage:
                            imageInfoSet[i].view = 'Left'
                            csvWriter.writerow([info.imageName,"Left"])
                        else:
                            imageInfoSet[i].view = 'Right'
                            csvWriter.writerow([info.imageName,"Right"])

        for i in range( len(imageInfoSet) ):
            print(imageInfoSet[i].imageName)
            if imageInfoSet[i].imageName == 'frontal.jpg':
                imageInfoSet[i].view = 'Face'
            if imageInfoSet[i].imageName == 'left buccal.jpg':
                imageInfoSet[i].view = 'Left'
            if imageInfoSet[i].imageName == 'lower occlusal.jpg':
                imageInfoSet[i].view = 'Below'
            if imageInfoSet[i].imageName == 'right buccal.jpg':
                imageInfoSet[i].view = 'Right'
            if imageInfoSet[i].imageName == 'upper occlusal.jpg':
                imageInfoSet[i].view = 'Up'
        
        writeFile.close()
        



   