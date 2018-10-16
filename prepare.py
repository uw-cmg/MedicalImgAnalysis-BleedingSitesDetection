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
#from skimage import io
#import matplotlib.pyplot as plt

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

    fileList = os.listdir(dataDir)
    numImages = len(fileList)

    if
    print(numImages)
    for f in fileList:
        print(f)
    # NRMSESummaryList = list()
    # # check starting ID for the edge case starting from 2
    # if "Pt-Mo50_input_images" in convImgDir.split("/"):
    #     for id in range(2,numImages+1):
    #         NRMSESummaryList.append(NRMSE_two_images(convPrefix,convSuffix,convImgDir,multiPrefix,multiSuffix,multiImgDir,id))
    # else:
    #     for id in range(1,numImages):
    #         NRMSESummaryList.append(NRMSE_two_images(convPrefix,convSuffix,convImgDir,multiPrefix,multiSuffix,multiImgDir,id))
    # return NRMSESummaryList


if __name__ == '__main__':
    print("start the pre-processing scripts")
    getFilesList(dataDir)


