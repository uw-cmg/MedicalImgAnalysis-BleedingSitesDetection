import os
from skimage import io, img_as_float
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15,5)

from matplotlib.patches import Rectangle
import numpy as np
import utils as utl

p03 = lambda x: np.percentile(x,3)
p97 = lambda x: np.percentile(x,97)
clip397 = lambda x: np.clip(x,p03(x),p97(x))

def get_img_list(patient, typ, datapath):
	images = []
	bboxes = []
	for file in os.listdir(datapath):
		if (".jpg" in file) and (typ in file) and (file.startswith(patient) and ("avg" not in file)):
			img = io.imread(datapath+file)
			images.append(img)
		if (".txt" in file) and (typ in file) and (file.startswith(patient)):
			with open(datapath+file,"r") as f: bbx = f.read()
			bbx = [int(float(i)*800) for i in bbx.split()]
			bboxes.append(bbx)
	return images, bboxes

def bbx_viz(image, bbx, img_name, axis):
	"""
	expects that the bbox is in xymiwh format, size of image is 800x800
	"""
#	 fig1, ax1 = plt.subplots(figsize = figs)
	# Rectangle takes bottom-left coords (bottom in terms of data coordinates, as the y axis is flipped, its top left visually)
	bottom_left = (bbx[0]-bbx[2]/2.,bbx[1]-bbx[3]/2.)
	axis.add_patch(Rectangle(xy=bottom_left, width=bbx[2], height=bbx[3],linewidth=1,edgecolor='r',facecolor='none'))
	axis.imshow(image, cmap = "gray")
	axis.set_title(img_name,fontsize=18)

train = [15, 47, 58, 61, 68, 71, 12, 17, 24, 35, 40, 63, 67, 6, 20, 42, 50, 74, 75, 44, 64, 5, 49, 51, 54, 56, 66, 73]
test = [1, 18, 48, 60, 72]
new_test = [62, 76, 77, 78, 79, 81, 82, 83]#80
for j in new_test:
	data = "tr " if j in train else "te " 
	j = ("0"+str(j))[-2:]
	imglist, bbxlist = get_img_list(j+"_", "orig", "proc_versions/clip_img/")
	aggdiff = np.zeros((800,800),dtype='float64')
	agg_wgh = np.zeros((800,800),dtype="float64")
	for i in range(len(imglist)):
		if i==0:
			continue
		img_curr = np.array(imglist[i], dtype="float64")
		img_prev = np.array(imglist[i-1], dtype="float64")	
		img_curr = clip397(img_curr)
		img_prev = clip397(img_prev)
		agg_wgh = agg_wgh + (i)**6*img_curr
		agg_wgh = clip397(agg_wgh)

		diff = img_curr - img_prev
		diff[diff>0] *= 1
		diff[diff<=0] *= 5
		aggdiff += clip397(diff)

		fig1,axs = plt.subplots(1, 5, sharey=True, figsize=(15,5))
		bbx_viz(img_prev, bbxlist[-1], data+j+"_"+str(i)+" prev", axs[0])
		bbx_viz(img_curr, bbxlist[-1], data+j+"_"+str(i)+" curr", axs[1])
		bbx_viz(diff, bbxlist[-1], data+j+"_"+str(i)+" diff", axs[2])
		bbx_viz(agg_wgh, bbxlist[-1], data+j+"_"+str(i)+" agg_wgh", axs[3])
		bbx_viz(aggdiff, bbxlist[-1], data+j+"_"+str(i)+" aggdiff", axs[4])
		print(j)
		plt.savefig("tmp/"+str(j)+"_"+str(i)+".jpg")
