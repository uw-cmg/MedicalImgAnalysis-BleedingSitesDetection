import matplotlib
matplotlib.use('Agg')
import glob

from models import *
from utils import *
from datasets import *
import shutil

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



import logging
import joblib as jlb
logging.basicConfig(filename='detect.log', level=logging.INFO)
logging.info(" ")
logging.info("$$"*30)
logging.info(" ")
logging.info("available cpus - "+str(jlb.cpu_count()))

parser = argparse.ArgumentParser()
parser.add_argument('--model_config_path', type=str, default='config/yolov3_1class.cfg', help='path to model config file')
parser.add_argument('--weights_folder', type=str, default='checkpoints/last7_16_May_02_44*', help='path to weights file')
parser.add_argument('--outbase', type=str, default='output2/', help='path to out base')
# parser.add_argument('--class_path', type=str, default='data/bleeds.names', help='path to class label file')
parser.add_argument("--data_config_path", type=str, default="config/bleeds.data", help="path to data config file")
parser.add_argument('--conf_thres', type=float, default=0.6, help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.3, help='iou thresshold for non-maximum suppression')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--ncpu', type=int, default=1, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_size', type=int, default=800, help='size of each image dimension')
parser.add_argument('--use_cuda', type=bool, default=False, help='whether to use cuda if available')
parser.add_argument('--topn', type=int, default=3, help='top n detections to keep')
parser.add_argument('-l', dest="linux", action="store_true")
opt = parser.parse_args()
logging.info(opt)


def dump_plots(idx, weights_path):
	import matplotlib
	matplotlib.use('Agg')
	logging.basicConfig(filename='detect.log', level=logging.INFO)
	logging.info(idx +" "+weights_path)
	cuda = torch.cuda.is_available() and opt.use_cuda
	# Get data configuration
	data_config = parse_data_config(opt.data_config_path)
	test_path = data_config["test"]
	dataloader = torch.utils.data.DataLoader(ListDataset(test_path), batch_size=opt.batch_size, shuffle=True)
	Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

	odir = opt.outbase + "__".join(weights_path.split("/")[1:])+"_top"+str(opt.topn)
	os.makedirs(odir, exist_ok=True)

	# Set up model
	model = Darknet(opt.model_config_path, img_size=opt.img_size)
	model.load_state_dict(torch.load(weights_path,map_location='cpu'))

	if cuda:
		model.cuda()
	model.eval() # Set in evaluation mode

	logging.info(idx +" "+'Performing object detection:')
	for batch_i, (img_paths_i, input_imgs,_) in enumerate(dataloader):
		# Configure input
		input_imgs = Variable(input_imgs.type(Tensor))

		# Get detections
		with torch.no_grad():
			detections = model(input_imgs)
			detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

		logging.info(idx +" batch "+ str(batch_i) +" "+ "detected, Saving Images")
		# Iterate through images and save plot of detections
		image_start = time.time() 
		for img_i, (path, detections) in enumerate(zip(img_paths_i, detections)):
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
			tru_bb_path = path.replace('.jpg', '.txt')[:-4]+"_yolo.txt"
			with open(tru_bb_path,"r") as f: 
				tru_bb = f.read()
				tru_bb = [int(float(i)*opt.img_size) for i in tru_bb.split()]
			### getting bottom left for matplotlib Rectangle api
			bottom_left = (tru_bb[0]-tru_bb[2]/2.,tru_bb[1]-tru_bb[3]/2.)
			

			# Draw bounding boxes and labels of detections
			ax.add_patch(patches.Rectangle(xy=bottom_left, width=tru_bb[2], height=tru_bb[3],linewidth=1,edgecolor='g',facecolor='none'))
			# if detections is not None:
			count = 0
			if detections is not None:
				detections = detections[:opt.topn]
				unique_labels = detections[:, -1].cpu().unique()
				n_cls_preds = len(unique_labels)
				logging.info(idx +" "+str(img_i) +" "+ path)
				for x1, y1, x2, y2, conf in detections:
					count += 1
					logging.info(idx + " obj_conf: " +str(round(float(conf.item()),3)) )
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
			# Save generated image with detections
			plt.axis('off')
			plt.gca().xaxis.set_major_locator(NullLocator())
			plt.gca().yaxis.set_major_locator(NullLocator())
			# path_tmp = path.replace(".jpg","_t"+str(count)+".jpg")
			# plt.savefig(odir+'/'+path_tmp.split("/")[-1], bbox_inches='tight', pad_inches=0.0)
			# else:
			plt.savefig(odir+'/'+path.split("/")[-1], bbox_inches='tight', pad_inches=0.0)
			plt.cla()
			plt.clf()
			plt.close()
			print(weights_path, path, (time.time()-image_start))
			image_start = time.time()


# weights_paths = sum([[i.replace("\\","/")+"/"+j for j in os.listdir(i)] for i in glob.glob(opt.weights_folder)], [])
### the sum just flattens the list!
# print(weights_paths)
weights_paths = ["checkpoints/last7_16_May_02_44/240_weights.pt"]
# ,"checkpoints/last5_15_May_02_27/160_weights.pt",
# weights_paths = ["checkpoints/last5_16_May_00_51/100_weights.pt","checkpoints/last5_16_May_00_51/120_weights.pt"]

if opt.linux:
	jlb.Parallel(n_jobs=opt.ncpu)(jlb.delayed(dump_plots)(str(idx)+"/"+str(len(weights_paths)), weights_path) for idx, weights_path in enumerate(weights_paths))
	shutil.make_archive(base_name = opt.outbase, format ='zip', root_dir = opt.outbase, base_dir = None)
else:
	for idx, weights_path in enumerate(weights_paths):
		dump_plots(str(idx)+"/"+str(len(weights_paths)), weights_path)
