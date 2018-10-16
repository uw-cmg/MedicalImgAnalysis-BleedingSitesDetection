#!/usr/bin/env python
"""
The preparing scipts that translate all the labeling results to the input for defect detection

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
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

# global path and prefix or suffix

"""
Metadata for the running of Project
 
Modify before using
"""

dataDir = "./data"

"""
Functions
"""

def getFilesList(dataDir):
    """


    Parameters
    ----------
    dataDir : the directory that you store all the data

    Returns
    -------
    NRMSESummaryList : list that contains NRMSE of each images
    """
    numImages = len(os.listdir(convImgDir))
    NRMSESummaryList = list()
    # check starting ID for the edge case starting from 2
    if "Pt-Mo50_input_images" in convImgDir.split("/"):
        for id in range(2,numImages+1):
            NRMSESummaryList.append(NRMSE_two_images(convPrefix,convSuffix,convImgDir,multiPrefix,multiSuffix,multiImgDir,id))
    else:
        for id in range(1,numImages):
            NRMSESummaryList.append(NRMSE_two_images(convPrefix,convSuffix,convImgDir,multiPrefix,multiSuffix,multiImgDir,id))
    return NRMSESummaryList

def NRMSE_two_images(convPrefix,convSuffix,convImgDir,multiPrefix,multiSuffix,multiImgDir,id):
    """
    Calculate NRMSE of two images

    Parameters
    ----------
    convPrefix : the prefix of convolution image name
    convSuffix : the Suffix of convolution image name
    convImgDir : the directory stores image data of convolution images
    multiPrefix : the prefix of multislice image name
    multiSuffix : the Suffix of multislice image name
    multiImgDir : the directory stores image data of multislice images
    id : the current image id to compare

    Returns
    -------
    nrMSE : the NRMSE of current two images
    """
    convArr = np.loadtxt(convImgDir + "/" + convPrefix + str(id) + convSuffix)
    multiArr = io.imread(multiImgDir + "/" + multiPrefix + str(id) + multiSuffix)
    # check if the two array has the same shape
    # if not, something wrong with the data
    assert convArr.shape == multiArr.shape
    mseArr = ((normalize(convArr) - normalize(multiArr))**2) / (normalize(multiArr)**2)
    return np.sqrt(np.sum(mseArr) / (convArr.shape[0] * convArr.shape[1]))

def normalize(v):
    """
    Normalize the image 2D array of the image

    Parameters
    ----------
    v : input image 2D array that needs to be normalized

    Returns
    -------
    normailzed 2D array
    """
    return (v / np.sqrt((np.sum(v ** 2))))

if __name__ == '__main__':
    numImages = len(os.listdir(dataDir))
    print(numImages)


