dataPath = "../Data_postiveOnly/"
import os
print(os.getcwd())
import imgaug as ia
import skimage.io
import errno
import numpy as np
import skimage.color as color
#import matplotlib.pyplot as plt
import os
#from skimage.color import gray2rgb


def bboxSetupInImage(datapath, txtFile, img):
    """
    This is the function that reads in the bounding box files and then using imgaug to set up the bounding box on images

    :param txtFile: the txt file that store bounding box information
    :param img: the image file variable to represent the img to be plotted bounding box on it
    :return bbs: the image with bounding box in it
    """
    with open(datapath + 'bounding_boxes/' + txtFile, 'r') as f:
        content = [line.rstrip('\n') for line in f]
        iaBBoxList = []
        for bbline in content:
            bbox = bbline.strip().split()
            # print(bbox[1])
            if len(bbox) == 4:
                iaBBoxList.append(ia.BoundingBox(
                    x1=float(bbox[1]),
                    y1=float(bbox[0]),
                    x2=float(bbox[3]),
                    y2=float(bbox[2])))
        bbs = ia.BoundingBoxesOnImage(iaBBoxList, shape=img.shape)
        return bbs


def saveAugbbox2TXT(txtFile, bbs):
    """
    This is the function that save the augmented bounding box files into ChainerCV bbox format

    :param txtFile: the txt file that want to save
    :param bbs: bounding box lists
    """
    with open('' + txtFile, 'w') as f:
        for i in range(len(bbs.bounding_boxes)):
            bb = bbs_aug.bounding_boxes[i]
            # print("%s %.2f %.2f %.2f %.2f"%(bb.label,bb.y1,bb.x1,bb.y2,bb.x2))
            f.write("%.2f %.2f %.2f %.2f\n" % ( bb.y1, bb.x1, bb.y2, bb.x2))

def getImageList(imageTXT):
    """
    Function to loop the testing images for test
    :param imageTXT: the txt that stores the
    :return: imageFileList: the list contains all the original test image list
    """
    imageFileList = list()
    with open(imageTXT,'r') as f:
        lines = f.readlines()
        for line in lines:
            imageFileList.append(line.strip())
    return imageFileList


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

##################################################
# 1. Define data augmentation operations
##################################################

trainImageTxtFile = dataPath + "trainimages.txt"
imageList = getImageList(trainImageTxtFile)

current_operation = "GaussianNoise"

# Add gaussian noise.
# For 50% of all images, we sample the noise once per pixel.
# For the other 50% of all images, we sample the noise per pixel AND
# channel. This can change the color (not only brightness) of the
# pixels.

from imgaug import augmenters as iaa

ia.seed(1)
seq = iaa.Sequential([
    iaa.AdditiveGaussianNoise(loc=0,
                              scale=(0.0, 0.01 * 255),
                              per_channel=0.5)
])

# seq = iaa.Sequential([
#     # Adjust contrast by scaling each pixel value to (I_ij/255.0)**gamma.
#     # Values in the range gamma=(0.5, 2.0) seem to be sensible.
#     iaa.GammaContrast((0.5, 1.5))
# ])

# Make our sequence deterministic.
# We can now apply it to the image and then to the BBs and it will
# lead to the same augmentations.
# IMPORTANT: Call this once PER BATCH, otherwise you will always get the exactly same augmentations for every batch!
seq_det = seq.to_deterministic()

##################################################
# 2. loop through images
##################################################

for img in imageList:
    print(img)
    # Grayscale images must have shape (height, width, 1) each.
    # print(os.listdir(dataPath+'images/'))
    currentimage = skimage.io.imread(dataPath + 'images/' + img).astype(np.uint8)
    # gray2rgb() simply duplicates the gray values over the three color channels.
    currentimage = color.gray2rgb(currentimage)
    bbs = bboxSetupInImage(dataPath, img.rstrip('.jpg') + '.txt', currentimage)
    # Augment BBs and images.
    # As we only have one image and list of BBs, we use
    # [image] and [bbs] to turn both into lists (batches) for the# functions and then [0] to reverse that. In a real experiment, your
    # variables would likely already be lists.
    image_aug = seq_det.augment_images([currentimage])[0]
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
    print(bbs_aug)
    augImgFolder = current_operation + "Images"
    augTxTFolder = current_operation + "TXT"
    createFolder(augImgFolder)
    createFolder(augTxTFolder)
    # Save aug images and bboxes
    skimage.io.imsave(augImgFolder + '/' +
                      img.rstrip('.jpg') +
                      '_' + current_operation +
                      '.jpg'
                      , image_aug)
    saveAugbbox2TXT(augTxTFolder + '/' +
                    img.rstrip('.jpg') +
                    '_' + current_operation +
                    '.txt', bbs_aug)
    # image with BBs before/after augmentation (shown below)
    # image_before = bbs.draw_on_image(currentimage, thickness=2)
    # image_after = bbs_aug.draw_on_image(image_aug,
    # thickness=2, color=[0, 0, 255])
    # image with BBs before/after augmentation (shown below)
    # plot and save figures before and after data augmentations
    # skimage.io.imshow(image_before)
    # skimage.io.imshow(image_after)
    # for i in range(len(bbs.bounding_boxes)):
    #     before = bbs.bounding_boxes[i]
    #     after = bbs_aug.bounding_boxes[i]
    #     print("BB %d: (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
    #         i,
    #         before.x1, before.y1, before.x2, before.y2,
    #         after.x1, after.y1, after.x2, after.y2)
    #     )

##################################################
# 1. Define data augmentation operations
##################################################

trainImageTxtFile = dataPath + "trainimages.txt"
imageList = getImageList(trainImageTxtFile)

current_operation = "GaussianBlur"

# blur images with a sigma of 0 to 3.0
from imgaug import augmenters as iaa
ia.seed(1)
seq = iaa.Sequential([
  iaa.GaussianBlur(sigma=(0, 3))
])

# seq = iaa.Sequential([
#     # Adjust contrast by scaling each pixel value to (I_ij/255.0)**gamma.
#     # Values in the range gamma=(0.5, 2.0) seem to be sensible.
#     iaa.GammaContrast((0.5, 1.5))
# ])

# Make our sequence deterministic.
# We can now apply it to the image and then to the BBs and it will
# lead to the same augmentations.
# IMPORTANT: Call this once PER BATCH, otherwise you will always get the exactly same augmentations for every batch!
seq_det = seq.to_deterministic()

##################################################
# 2. loop through images
##################################################

for img in imageList:
    print(img)
    # Grayscale images must have shape (height, width, 1) each.
    #print(os.listdir(dataPath+'images/'))
    currentimage = skimage.io.imread(dataPath+'images/'+img).astype(np.uint8)
    # gray2rgb() simply duplicates the gray values over the three color channels.
    currentimage = color.gray2rgb(currentimage)
    bbs = bboxSetupInImage(dataPath , img.rstrip('.jpg') + '.txt',currentimage)
    # Augment BBs and images.
    # As we only have one image and list of BBs, we use
    # [image] and [bbs] to turn both into lists (batches) for the# functions and then [0] to reverse that. In a real experiment, your
    # variables would likely already be lists.
    image_aug = seq_det.augment_images([currentimage])[0]
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
    augImgFolder = current_operation + "Images"
    augTxTFolder = current_operation + "TXT"
    createFolder(augImgFolder)
    createFolder(augTxTFolder)
    # Save aug images and bboxes
    skimage.io.imsave(augImgFolder + '/'+
                      img.rstrip('.jpg') +
                      '_' + current_operation +
                      '.jpg'
                      ,image_aug)
    saveAugbbox2TXT(augTxTFolder+ '/'+
                    img.rstrip('.jpg') +
                    '_'+ current_operation +
                    '.txt',bbs_aug)
    # image with BBs before/after augmentation (shown below)
    # image_before = bbs.draw_on_image(currentimage, thickness=2)
    # image_after = bbs_aug.draw_on_image(image_aug,
    # thickness=2, color=[0, 0, 255])
    # image with BBs before/after augmentation (shown below)
    # plot and save figures before and after data augmentations
    #skimage.io.imshow(image_before)
    #skimage.io.imshow(image_after)
    # for i in range(len(bbs.bounding_boxes)):
    #     before = bbs.bounding_boxes[i]
    #     after = bbs_aug.bounding_boxes[i]
    #     print("BB %d: (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
    #         i,
    #         before.x1, before.y1, before.x2, before.y2,
    #         after.x1, after.y1, after.x2, after.y2)
    #     )

##################################################
# 1. Define data augmentation operations
##################################################

trainImageTxtFile = dataPath + "trainimages.txt"
imageList = getImageList(trainImageTxtFile)

current_operation = "Brightness"

# Strengthen or weaken the contrast in each image.
from imgaug import augmenters as iaa
ia.seed(1)
seq = iaa.Sequential([
    iaa.Multiply((1.2, 1.5))
])

# seq = iaa.Sequential([
#     # Adjust contrast by scaling each pixel value to (I_ij/255.0)**gamma.
#     # Values in the range gamma=(0.5, 2.0) seem to be sensible.
#     iaa.GammaContrast((0.5, 1.5))
# ])

# Make our sequence deterministic.
# We can now apply it to the image and then to the BBs and it will
# lead to the same augmentations.
# IMPORTANT: Call this once PER BATCH, otherwise you will always get the exactly same augmentations for every batch!
seq_det = seq.to_deterministic()

##################################################
# 2. loop through images
##################################################

for img in imageList:
    print(img)
    # Grayscale images must have shape (height, width, 1) each.
    #print(os.listdir(dataPath+'images/'))
    currentimage = skimage.io.imread(dataPath+'images/'+img).astype(np.uint8)
    # gray2rgb() simply duplicates the gray values over the three color channels.
    currentimage = color.gray2rgb(currentimage)
    bbs = bboxSetupInImage(dataPath , img.rstrip('.jpg') + '.txt',currentimage)
    # Augment BBs and images.
    # As we only have one image and list of BBs, we use
    # [image] and [bbs] to turn both into lists (batches) for the# functions and then [0] to reverse that. In a real experiment, your
    # variables would likely already be lists.
    image_aug = seq_det.augment_images([currentimage])[0]
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
    augImgFolder = current_operation + "Images"
    augTxTFolder = current_operation + "TXT"
    createFolder(augImgFolder)
    createFolder(augTxTFolder)
    # Save aug images and bboxes
    skimage.io.imsave(augImgFolder + '/'+
                      img.rstrip('.jpg') +
                      '_' + current_operation +
                      '.jpg'
                      ,image_aug)
    saveAugbbox2TXT(augTxTFolder+ '/'+
                    img.rstrip('.jpg') +
                    '_'+ current_operation +
                    '.txt',bbs_aug)
    # image with BBs before/after augmentation (shown below)
    # image_before = bbs.draw_on_image(currentimage, thickness=2)
    # image_after = bbs_aug.draw_on_image(image_aug,
    # thickness=2, color=[0, 0, 255])
    # image with BBs before/after augmentation (shown below)
    # plot and save figures before and after data augmentations
    #skimage.io.imshow(image_before)
    #skimage.io.imshow(image_after)
    # for i in range(len(bbs.bounding_boxes)):
    #     before = bbs.bounding_boxes[i]
    #     after = bbs_aug.bounding_boxes[i]
    #     print("BB %d: (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
    #         i,
    #         before.x1, before.y1, before.x2, before.y2,
    #         after.x1, after.y1, after.x2, after.y2)
    #     )

##################################################
# 1. Define data augmentation operations
##################################################

trainImageTxtFile = dataPath + "trainimages.txt"
imageList = getImageList(trainImageTxtFile)

current_operation = "Fliplr"

# Flip/mirror input images horizontally.

from imgaug import augmenters as iaa

ia.seed(1)
seq = iaa.Sequential([
    iaa.Fliplr(1.0)
])

# seq = iaa.Sequential([
#     # Adjust contrast by scaling each pixel value to (I_ij/255.0)**gamma.
#     # Values in the range gamma=(0.5, 2.0) seem to be sensible.
#     iaa.GammaContrast((0.5, 1.5))
# ])

# Make our sequence deterministic.
# We can now apply it to the image and then to the BBs and it will
# lead to the same augmentations.
# IMPORTANT: Call this once PER BATCH, otherwise you will always get the exactly same augmentations for every batch!
seq_det = seq.to_deterministic()

##################################################
# 2. loop through images
##################################################

for img in imageList:
    print(img)
    # Grayscale images must have shape (height, width, 1) each.
    # print(os.listdir(dataPath+'images/'))
    currentimage = skimage.io.imread(dataPath + 'images/' + img).astype(np.uint8)
    # gray2rgb() simply duplicates the gray values over the three color channels.
    currentimage = color.gray2rgb(currentimage)
    bbs = bboxSetupInImage(dataPath, img.rstrip('.jpg') + '.txt', currentimage)
    # Augment BBs and images.
    # As we only have one image and list of BBs, we use
    # [image] and [bbs] to turn both into lists (batches) for the# functions and then [0] to reverse that. In a real experiment, your
    # variables would likely already be lists.
    image_aug = seq_det.augment_images([currentimage])[0]
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
    augImgFolder = current_operation + "Images"
    augTxTFolder = current_operation + "TXT"
    createFolder(augImgFolder)
    createFolder(augTxTFolder)
    # Save aug images and bboxes
    skimage.io.imsave(augImgFolder + '/' +
                      img.rstrip('.jpg') +
                      '_' + current_operation +
                      '.jpg'
                      , image_aug)
    saveAugbbox2TXT(augTxTFolder + '/' +
                    img.rstrip('.jpg') +
                    '_' + current_operation +
                    '.txt', bbs_aug)
    # image with BBs before/after augmentation (shown below)
    # image_before = bbs.draw_on_image(currentimage, thickness=2)
    # image_after = bbs_aug.draw_on_image(image_aug,
    # thickness=2, color=[0, 0, 255])
    # image with BBs before/after augmentation (shown below)
    # plot and save figures before and after data augmentations
    # skimage.io.imshow(image_before)
    # skimage.io.imshow(image_after)
    # for i in range(len(bbs.bounding_boxes)):
    #     before = bbs.bounding_boxes[i]
    #     after = bbs_aug.bounding_boxes[i]
    #     print("BB %d: (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
    #         i,
    #         before.x1, before.y1, before.x2, before.y2,
    #         after.x1, after.y1, after.x2, after.y2)
    #     )

##################################################
# 1. Define data augmentation operations
##################################################

trainImageTxtFile = dataPath + "trainimages.txt"
imageList = getImageList(trainImageTxtFile)

current_operation = "Flipud"

# Flip/mirror input images vertically.

from imgaug import augmenters as iaa

ia.seed(1)
seq = iaa.Sequential([
    iaa.Flipud(1.0)
])

# seq = iaa.Sequential([
#     # Adjust contrast by scaling each pixel value to (I_ij/255.0)**gamma.
#     # Values in the range gamma=(0.5, 2.0) seem to be sensible.
#     iaa.GammaContrast((0.5, 1.5))
# ])

# Make our sequence deterministic.
# We can now apply it to the image and then to the BBs and it will
# lead to the same augmentations.
# IMPORTANT: Call this once PER BATCH, otherwise you will always get the exactly same augmentations for every batch!
seq_det = seq.to_deterministic()

##################################################
# 2. loop through images
##################################################

for img in imageList:
    print(img)
    # Grayscale images must have shape (height, width, 1) each.
    # print(os.listdir(dataPath+'images/'))
    currentimage = skimage.io.imread(dataPath + 'images/' + img).astype(np.uint8)
    # gray2rgb() simply duplicates the gray values over the three color channels.
    currentimage = color.gray2rgb(currentimage)
    bbs = bboxSetupInImage(dataPath, img.rstrip('.jpg') + '.txt', currentimage)
    # Augment BBs and images.
    # As we only have one image and list of BBs, we use
    # [image] and [bbs] to turn both into lists (batches) for the# functions and then [0] to reverse that. In a real experiment, your
    # variables would likely already be lists.
    image_aug = seq_det.augment_images([currentimage])[0]
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
    augImgFolder = current_operation + "Images"
    augTxTFolder = current_operation + "TXT"
    createFolder(augImgFolder)
    createFolder(augTxTFolder)
    # Save aug images and bboxes
    skimage.io.imsave(augImgFolder + '/' +
                      img.rstrip('.jpg') +
                      '_' + current_operation +
                      '.jpg'
                      , image_aug)
    saveAugbbox2TXT(augTxTFolder + '/' +
                    img.rstrip('.jpg') +
                    '_' + current_operation +
                    '.txt', bbs_aug)
    # image with BBs before/after augmentation (shown below)
    # image_before = bbs.draw_on_image(currentimage, thickness=2)
    # image_after = bbs_aug.draw_on_image(image_aug,
    # thickness=2, color=[0, 0, 255])
    # image with BBs before/after augmentation (shown below)
    # plot and save figures before and after data augmentations
    # skimage.io.imshow(image_before)
    # skimage.io.imshow(image_after)
    # for i in range(len(bbs.bounding_boxes)):
    #     before = bbs.bounding_boxes[i]
    #     after = bbs_aug.bounding_boxes[i]
    #     print("BB %d: (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
    #         i,
    #         before.x1, before.y1, before.x2, before.y2,
    #         after.x1, after.y1, after.x2, after.y2)
    #     )

##################################################
# 1. Define data augmentation operations
##################################################

trainImageTxtFile = dataPath + "trainimages.txt"
imageList = getImageList(trainImageTxtFile)

current_operation = "Rot90or270Degree"

# Rotates all images by 90 or 270 degrees.

from imgaug import augmenters as iaa

ia.seed(1)
seq = iaa.Sequential([
    iaa.Rot90([1, 3])
])

# seq = iaa.Sequential([
#     # Adjust contrast by scaling each pixel value to (I_ij/255.0)**gamma.
#     # Values in the range gamma=(0.5, 2.0) seem to be sensible.
#     iaa.GammaContrast((0.5, 1.5))
# ])

# Make our sequence deterministic.
# We can now apply it to the image and then to the BBs and it will
# lead to the same augmentations.
# IMPORTANT: Call this once PER BATCH, otherwise you will always get the exactly same augmentations for every batch!
seq_det = seq.to_deterministic()

##################################################
# 2. loop through images
##################################################

for img in imageList:
    print(img)
    # Grayscale images must have shape (height, width, 1) each.
    # print(os.listdir(dataPath+'images/'))
    currentimage = skimage.io.imread(dataPath + 'images/' + img).astype(np.uint8)
    # gray2rgb() simply duplicates the gray values over the three color channels.
    currentimage = color.gray2rgb(currentimage)
    bbs = bboxSetupInImage(dataPath, img.rstrip('.jpg') + '.txt', currentimage)
    # Augment BBs and images.
    # As we only have one image and list of BBs, we use
    # [image] and [bbs] to turn both into lists (batches) for the# functions and then [0] to reverse that. In a real experiment, your
    # variables would likely already be lists.
    image_aug = seq_det.augment_images([currentimage])[0]
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
    augImgFolder = current_operation + "Images"
    augTxTFolder = current_operation + "TXT"
    createFolder(augImgFolder)
    createFolder(augTxTFolder)
    # Save aug images and bboxes
    skimage.io.imsave(augImgFolder + '/' +
                      img.rstrip('.jpg') +
                      '_' + current_operation +
                      '.jpg'
                      , image_aug)
    saveAugbbox2TXT(augTxTFolder + '/' +
                    img.rstrip('.jpg') +
                    '_' + current_operation +
                    '.txt', bbs_aug)
    # image with BBs before/after augmentation (shown below)
    # image_before = bbs.draw_on_image(currentimage, thickness=2)
    # image_after = bbs_aug.draw_on_image(image_aug,
    # thickness=2, color=[0, 0, 255])
    # image with BBs before/after augmentation (shown below)
    # plot and save figures before and after data augmentations
    # skimage.io.imshow(image_before)
    # skimage.io.imshow(image_after)
    # for i in range(len(bbs.bounding_boxes)):
    #     before = bbs.bounding_boxes[i]
    #     after = bbs_aug.bounding_boxes[i]
    #     print("BB %d: (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
    #         i,
    #         before.x1, before.y1, before.x2, before.y2,
    #         after.x1, after.y1, after.x2, after.y2)
    #     )

