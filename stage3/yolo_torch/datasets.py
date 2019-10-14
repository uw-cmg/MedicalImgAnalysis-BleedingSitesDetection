import os
import numpy as np
import torch
# import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import Image


# class ImageFolder(Dataset):
#     def __init__(self, folder_path, img_size=416):
#         self.files = sorted(glob.glob('%s/*.*' % folder_path))
#         self.img_shape = (img_size, img_size)

#     def __getitem__(self, index):
#         img_path = self.files[index % len(self.files)]
#         # Extract image
#         img = np.array(Image.open(img_path))
        
#         ## Adding a a channel for the grayscale image, as the input it takes is NCHW
#         input_img = img[None,: , :]
#         # As pytorch tensor
#         input_img = torch.FloatTensor(input_img)

#         return img_path, input_img

#     def __len__(self):
#         return len(self.files)


class ListDataset(Dataset):

    def __init__(self, list_path, img_size=800):
        with open(list_path, 'r') as file:
            self.img_files = file.readlines()
        self.img_files = [i.strip() for i in self.img_files]
        ### bbox label files have the same name and .txt extension
        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt')[:-4] +"_yolo.txt" for path in self.img_files]
        self.img_shape = (img_size, img_size) #dbt are images supposed to be square?!! 
        #### self.max_objects = 50  ### max objects that could be there in an image
        self.max_objects = 5  ### max objects that could be there in an image


    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()
        patient = img_path.split("/")[2].split("_")[0]
        imgtyps = ["_hf", "_vf", "_orig"]
        for typ in imgtyps:
            if typ in img_path:
                aggimg_path = "/".join(img_path.split("/")[:-1])+"/"+patient+typ+"_aggimg.jpg"
                aggdiff_path = "/".join(img_path.split("/")[:-1])+"/"+patient+typ+"_aggdiff.jpg"
                break
        # diffimg_path = "/".join(img_path.split("/")[:-1])+"/"+patient+"_"+imgtyp+"_diff.jpg"

        img = np.array(Image.open(img_path))
        aggimg = np.array(Image.open(aggimg_path))
        aggdiff = np.array(Image.open(aggdiff_path))

        ### normalize
        img = (img - img.mean())/img.std()
        aggimg = (aggimg - aggimg.mean())/aggimg.std()
        aggdiff = (aggdiff - aggdiff.mean())/aggdiff.std()


        # if len(img.shape)>2: ## if color then 3rd channel will be there
        #     h, w, _ = img.shape ### the earlier code ignored the 3rd channel even for color why? #dbt #todo
        # else:
        #     h, w = img.shape
        # dim_diff = np.abs(h - w)
        # # Upper (left) and lower (right) padding
        # pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2 #dbt0
        # # Determine padding
        # pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # # Add padding
        # input_img = np.pad(img, pad, 'constant', constant_values=128) / 255.
        # padded_h, padded_w, _ = input_img.shape
        # # Resize and normalize
        # input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        # # Channels-first - by default np.transpose would do 2,1,0, we can give the axes argument for any other order
        # input_img = np.transpose(input_img, axes=(2, 0, 1)) 
        
        ### Adding a channel for the grayscale image, as the input it takes is BCHW(batchsize, channels, width, height)
        ### Below we only add the color channel, the B in BCHW is added by the data loader based on batch size
        ### as we are stacking 2 imgs, we already have 2 channels
        input_img = np.stack((img, aggimg, aggdiff))
        # input_img = np.stack((img, aggimg, aggdiff))
        # input_img = img[None,:,:]
        # As pytorch tensor
        input_img = torch.FloatTensor(input_img)

        #---------
        #  Label
        #---------

        bbx_path = self.label_files[index % len(self.img_files)].rstrip()

        labels = None
        if os.path.exists(bbx_path):
            labels = np.loadtxt(bbx_path).reshape(-1, 4)
        ### again none of this is required for now as we have the labels in the required format already

        #     # Extract coordinates for unpadded + unscaled image
        #     x1 = w * (labels[:, 1] - labels[:, 3]/2)
        #     y1 = h * (labels[:, 2] - labels[:, 4]/2)
        #     x2 = w * (labels[:, 1] + labels[:, 3]/2)
        #     y2 = h * (labels[:, 2] + labels[:, 4]/2)
        #     # Adjust for added padding
        #     x1 += pad[1][0]
        #     y1 += pad[0][0]
        #     x2 += pad[1][0]
        #     y2 += pad[0][0]
        #     # Calculate ratios from coordinates
        #     labels[:, 1] = ((x1 + x2) / 2) / padded_w
        #     labels[:, 2] = ((y1 + y2) / 2) / padded_h
        #     labels[:, 3] *= w / padded_w
        #     labels[:, 4] *= h / padded_h
        # Fill matrix

        filled_labels = np.zeros((self.max_objects, 4))
        if labels is not None:
            filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        filled_labels = torch.from_numpy(filled_labels)

        return img_path, input_img, filled_labels

    def __len__(self):
        return len(self.img_files)
