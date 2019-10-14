"""
This file will rename the files under a certain folde to a new name

For our project this is to rename the starting ID of the images for PyTorch

"""
import os

startingBleedingID = 6177
startingNonBleedingID = 3121

# Change Bleeding Images
datPath = "../data/DataSet9/fastAITrain/valid/bleeding/"
print(len(os.listdir(datPath)))
for img in os.listdir(datPath):
    new_img_name = "bleeding."+str(startingBleedingID)+".jpg"
    os.rename(datPath+img,datPath+new_img_name)
    startingBleedingID += 1

# # Change Non-Bleeding Images
datPath = "../data/DataSet9/fastAITrain/valid/nonbleeding/"
print(len(os.listdir(datPath)))
for img in os.listdir(datPath):
    new_img_name = "nonbleeding."+str(startingNonBleedingID)+".jpg"
    os.rename(datPath+img,datPath+new_img_name)
    startingNonBleedingID += 1