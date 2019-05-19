'''
#  Remove all the unmasked images in the named folder

'''

# Import packages
import os

# Constant Settings
imgFolder = '5_5_19 Upload'

for img in os.listdir(imgFolder):
    imgs = img.split('.')
    if imgs[1] == 'jpg' and imgs[0][-1] == 'U':
        os.remove(imgFolder + '/' + img)