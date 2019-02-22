import csv
import os
from skimage import io
from collections import Counter
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt


def read_bbox(csvfilepath, coord_type="mid"):
    """
    returns x, y, width, height from the csv dumps we have
    """
    with open(csvfilepath) as csvfile:
        reader = csv.DictReader(csvfile)
        if coord_type == "mid":
            for row in reader:
                 return [int(float(row['X'])), int(float(row['Y'])), int(float(row['Width']))
                     , int(float(row['Height']))]

        # if coord_type == "left_top":
        #     for row in reader:
        #          return [int(float(row['BX'])), int(float(row['BY'])), int(float(row['Width']))
        #              , int(float(row['Height']))]


def shape_counts(bleeding_path):
    shape_counts = Counter()
    for pfolder in os.listdir(bleeding_path):
        for fname in os.listdir(bleeding_path+pfolder):
            if ".csv" in fname:
                try:
                    img = io.imread(bleeding_path+pfolder+"/"+fname[:-4]+".tif")
                    shape_counts.update([img.shape])
                except:
                    print(fname[:-4], "X"*20)
    return shape_counts.most_common()


def draw_rect(axis, bbox):
    axis.add_patch(Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none'))


def imgbboxdump(fname, nparr, bbox):
    plt.figure()
    plt.imshow(nparr, cmap="gray")
    draw_rect(plt.gca(),bbox)
    plt.savefig(fname, dpi=300)
    plt.close()

# def x1y1x2y2_to_xywh(bbox):
# 	"""
# 	assuming x1y1 is the bottom-left corner, x2y2 is top-right corner
# 	"""
# 	x1,y1,x2,y2 = bbox
# 	x = (x1+x2)/2.
# 	y = (y1+y2)/2.
# 	w = x2-x1
# 	h = y2-y1
# 	return x,y,w,h

# def xywh_to_x1y1x2y2(bbox):
# 	x,y,w,h = bbox
# 	x1 = x - w/2.
# 	y1 = y + h/2.
# 	x2 = x+w/2.
# 	y2 = y - h/2.
# 	return x1,y1,x2,y2

def proper_im_num(num_string):
    """
    for now expecting both the patient number and image number be <=99
    converts "15_6" to 1506
    """
    return int("".join(["{:02d}".format(int(i)) for i in  num_string.split("_")]))