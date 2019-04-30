# Import Packages

from shutil import copy
import os
import errno
# Define Image Augumentation Operations
print(os.getcwd())

augOperations = ["GaussianNoise",
                 "GaussianBlur",
                 "Brightness",
                 "ContrastNormalization",
                 "Fliplr",
                 "Flipud",
                 "Rot90or270Degree"
                 ]

# Define Helper Functions
def createFolder(folderName):
    """
    Safely create folder when needed

    :param folderName : the directory that you  want to safely create
    :return: None
    """
    if not os.path.exists(folderName):
        try:
            os.makedirs(folderName)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

# Define the data folder

dataPath = "tmp/" #"./Data3TypesYminXminYmaxXmax6/"
createFolder(dataPath + "images")
createFolder(dataPath + "bounding_boxes")

# Loop through images and bboxes

for oper in augOperations:
    for f in os.listdir('Aug/'+ oper + "Images/"):
        print(f)
        # Copy Images
        copy('Aug/'+ oper + "Images/" + f, dataPath + "images")
        # Copy TxT
        copy('Aug/'+ oper + "TXT/" + f.rstrip(".jpg") + ".txt",
             dataPath + "bounding_boxes")

print("Done!")

# for f in os.listdir(datDir):
#     fs = f.split('.')
#     if fs[1] == "tif":
#         covertTIF2JPG(datDir + '/' + f, fs[0])
#         copy(fs[0] + '.jpg', imgDir)
#     if fs[1] == "csv":
#         copy(datDir + '/' + f, csvDir)