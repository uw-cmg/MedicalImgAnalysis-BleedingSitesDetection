#!/usr/bin/env python
"""
The preparing scipts that translate all the labeling results to the input for defect detection

In this script, only images with bbox are kept!

"""

"""
Project Information modify if needed
"""
__author__ = "Mingren Shen"
__copyright__ = "Copyright 2018, The medical image analysis Project"
__credits__ = [""]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Mingren Shen"
__email__ = "mshen32@wisc.edu"
__status__ = "Development"

"""
End of Project information
"""

# import libraries
import os
import errno
from shutil import copy
from skimage import io
import csv
import numpy as np
#import matplotlib.pyplot as plt

# global path and prefix or suffix

"""
Metadata for the running of Project
 
Modify before using
"""

# Directory created

datDir = "data"
imgDir = "IMG"
csvDir = "CSV"
txtDir = "TXT"

"""
Functions
"""
def loopAllImg(imgDir,csvDir,txtDir):
    """
     For every JPG image generate TXT if CSV exist if not generate blank TXT

     Parameters
     ----------
     imgDir : the directory to store the images
     csvDir : the directory to store the csv files
     txtDir : the directory to store all the generate txt bbox information

    Returns
    -------
    None
    """
    for f in os.listdir(imgDir):
        fs = f.split('.')
        csv_file = csvDir + "/" + fs[0] + ".csv"
        txt_file = fs[0] + ".txt"
        if (os.path.exists(csv_file)):
            generateTXT(csv_file,txt_file)
        else:
            with open(txt_file, 'a'):  # Create file if does not exist
                pass
        copy(txt_file, txtDir)



def generateTXT(csvFile,txtFile):
    """
    generate TXT from CSV and

    Parameters
    ----------
    csvFile : the CSV File that needs to be processed
    txtFile : the TXT File that stores the bounding box information

    Returns
    -------
    None
    """
    with open(txtFile,'w') as txtfile:
        with open(csvFile) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                bbox = calculateBoundingBoxes(0,float(row['X']), float(row['Y']),float(row['Width']),float(row['Height']))
                for i in range(0,len(bbox)):
                    if i == len(bbox) - 1:
                        txtfile.write("%s \n" % bbox[i])
                    else:
                        txtfile.write("%s " % bbox[i])

def calculateBoundingBoxes(label, x, y, w, h):
    """
    calculate bounding box information form the center and length, width

    Parameters
    ----------
    label : the label of current bbox
    x : the x coordinate of center of bbox
    y : the y coordinate of center of bbox
    w : width of bbox
    h : hight of bbox

    Returns
    -------
    list contains the [label,Y1,X1,Y2,X2]
    Where (X1,Y1) is the top left point of bbox
          (X2,Y2) is the bottom right point of bbox
    """
    X1 = x - (w / 2)
    Y1 = y - (h / 2)
    X2 = x + (w / 2)
    Y2 = y + (h / 2)

    return [label, round(Y1, 2), round(X1, 2), round(Y2, 2), round(X2, 2)]

def splitFiles_withBBox(datDir,imgDir,csvDir):
    """
    pre-processing the files and prepare all needed files

    Parameters
    ----------
    datDir : the directory that you store all the data
    imgDir : the directory to store the images
    csvDir : the directory to store the csv files

    Returns
    -------
    None
    """
    for f in os.listdir(datDir):
        fs = f.split('.')
        #if fs[1] == "tif":
        if fs[1] == "csv":
            copy(datDir+'/'+f, csvDir)
            covertTIF2JPG(datDir + '/' + f, fs[0])
            copy(fs[0] + '.jpg', imgDir)

def covertTIF2JPG(imgsource,imgName):
    """
    Convert source TIF image to JPG image

    Parameters
    ----------
    imgsource : the TIF source image
    imgName : the target image name

    Returns
    -------
    None
    """
    img = io.imread(imgsource)
    io.imsave(imgName + ".jpg", img, quality=100)

def createFolder(folderName):
    """
    Safely create folder when needed


    Parameters
    ----------
    folderName : the directory that you  want to safely create

    Returns
    -------
    None
    """
    if not os.path.exists(folderName):
        try:
            os.makedirs(folderName)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

if __name__ == '__main__':
    print("start the pre-processing scripts")
    print("Initialization")
    createFolder(imgDir)
    createFolder(csvDir)
    createFolder(txtDir)
    print("move files to separated files")
    splitFiles_withBBox(datDir, imgDir, csvDir)
    print("generate bbox from CSV \n and pair every JPG with TXT")
    loopAllImg(imgDir, csvDir, txtDir)


