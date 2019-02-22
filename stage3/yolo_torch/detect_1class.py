from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from matplotlib.font_manager import FontProperties
# bbtop_font = FontProperties()
# bbtop_font.set_size("small")
# bbtop_font.set_weight("bold")


parser = argparse.ArgumentParser()
parser.add_argument('--image_folder', type=str, default='data/bleeds_test_samples', help='path to dataset')
parser.add_argument('--config_path', type=str, default='config/yolov3_1class.cfg', help='path to model config file')
parser.add_argument('--weights_path', type=str, default='checkpoints/09_Feb_18_31/140_weights.pt', help='path to weights file')
# parser.add_argument('--class_path', type=str, default='data/bleeds.names', help='path to class label file')
parser.add_argument("--data_config_path", type=str, default="config/bleeds.data", help="path to data config file")
parser.add_argument('--conf_thres', type=float, default=0.6, help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.5, help='iou thresshold for non-maximum suppression')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--n_cpu', type=int, default=1, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_size', type=int, default=800, help='size of each image dimension')
parser.add_argument('--use_cuda', type=bool, default=False, help='whether to use cuda if available')
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available() and opt.use_cuda

os.makedirs('output', exist_ok=True)

# Set up model
model = Darknet(opt.config_path, img_size=opt.img_size)
model.load_state_dict(torch.load(opt.weights_path))

if cuda:
    model.cuda()

model.eval() # Set in evaluation mode

# Get data configuration
data_config = parse_data_config(opt.data_config_path)
test_path = data_config["test"]

dataloader = torch.utils.data.DataLoader(ListDataset(test_path), batch_size=opt.batch_size, shuffle=True)

# classes = load_classes(opt.class_path) # Extracts class labels from file

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

odir = "output/" + "__".join(opt.weights_path.split("/")[1:])
if not os.path.exists(odir):
    os.makedirs(odir)

# img_paths = []           # Stores image paths
# img_detections = [] # Stores detections for each image index

print ('Performing object detection:')
prev_time = time.time()

for batch_i, (img_paths_i, input_imgs,_) in enumerate(dataloader):
    # Configure input
    input_imgs = Variable(input_imgs.type(Tensor))

    # Get detections
    with torch.no_grad():
        detections = model(input_imgs)
        detections = non_max_suppression(detections, 1, opt.conf_thres, opt.nms_thres)

    # Log progress
    current_time = time.time()
    inference_time = datetime.timedelta(seconds=current_time - prev_time)
    prev_time = current_time
    print ('\t+ Batch %d, Inference Time: %s' % (batch_i, inference_time))

    # # Save image and detections
    # img_paths.extend(img_paths_i)
    # img_detections.extend(detections)
    # if batch_i>=2: break
    print(batch_i, "detected", "Saving Images")

    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(img_paths_i, detections)):

        print ("(%d) Image: '%s'" % (img_i, path))

        # Create plot
        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img, cmap='gray')

        # The amount of padding that was added
        pad_x = max(img.shape[0] - img.shape[1], 0) * (opt.img_size / max(img.shape))
        pad_y = max(img.shape[1] - img.shape[0], 0) * (opt.img_size / max(img.shape))
        # Image height and width after padding is removed
        unpad_h = opt.img_size - pad_y
        unpad_w = opt.img_size - pad_x

        ### getting original box coords
        tru_bb_path = path.replace('.jpg', '.txt')
        with open(tru_bb_path,"r") as f: 
            tru_bb = f.read()
            tru_bb = [int(float(i)*opt.img_size) for i in tru_bb.split()[1:]]
        ### getting bottom left for matplotlib Rectangle api
        bottom_left = (tru_bb[0]-tru_bb[2]/2.,tru_bb[1]-tru_bb[3]/2.)
        


        # Draw bounding boxes and labels of detections
        if detections is not None:
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            # bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                print ('cls_conf: %.5f, obj_conf: %.5f' % (cls_conf.item(), conf.item()))
                # Rescale coordinates to original dimensions
                box_h = ((y2 - y1) / unpad_h) * img.shape[0]
                box_w = ((x2 - x1) / unpad_w) * img.shape[1]
                y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
                x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]

                # color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                ax.add_patch(patches.Rectangle((x1, y1),box_w,box_h,linewidth=1,edgecolor="b",facecolor='none'))
                # Add score
                plt.text(x1, y1, s=round(conf.item(),2), color='w', fontsize=6)
        ax.add_patch(patches.Rectangle(xy=bottom_left, width=tru_bb[2], height=tru_bb[3],linewidth=1,edgecolor='g',facecolor='none'))
        # Save generated image with detections
        plt.axis('off')
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        plt.savefig(odir+'/'+path.split("/")[-1], bbox_inches='tight', pad_inches=0.0)
        plt.close()