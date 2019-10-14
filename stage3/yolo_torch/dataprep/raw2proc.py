import os
import shutil
import numpy as np
from skimage import io, img_as_float, img_as_ubyte,exposure

import matplotlib.pyplot as plt
import utils as ut
from matplotlib.patches import Rectangle
import imgaug.augmenters as iaa
import imgaug as ia
from skimage import img_as_uint
p03 = lambda x: np.percentile(x,3)
p97 = lambda x: np.percentile(x,97)
clip397 = lambda x: np.clip(x,p03(x),p97(x))

raw_path = "raw_data/Dataset_5/"

### version1 is just the raw image with bbx's drawn on top, dunped in jpg fromat

ver1_path = "proc_versions/ver6/"

### dump the 800x800 train/test data with accum and diff features

currver_path = ver1_path
# currver_dump = "proc_versions/tmp1/"

os.makedirs(currver_path, exist_ok=True)
# os.makedirs(currver_dump, exist_ok=True)


train_patients = [1, 4, 5, 6, 11, 12, 17, 18, 19, 20, 24, 25, 33, 35, 40, 42, 44, 45, 48, 49, 50, 51, 52
				, 54, 55, 56, 57, 58, 61, 63, 64, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 7934, 7936, 7948
				, 80, 81, 82, 83]

# train_patients1 = [1002, 1003, 1004, 1005, 1006, 1009, 1010, 1011, 1012, 1014, 1016, 1017, 1019, 1021, 1024, 1025, 1026,
# 					1027, 1028]
# 22 (2480, 1920)
# 46 (2480, 1920)
# 47 (1920, 1920)
# 59 (1920, 2480)
# 62 (720, 720)

# test_patients = [1,18,48,60, 72]
# new_test = [76, 77, 78, 79, 81, 82, 83, 84, 85]#62, 80
# easy_test = [60, ]
# hard_and_ok_test = [72, 77, 78, 60]

#remove

test_patients = [15, 60, 72, 77, 78] + [37,38,39]
test_patients1 = [1002, 1003, 1004, 1005, 1006, 1009, 1010, 1011, 1012, 1014, 1016, 1017, 1019, 1021, 1024, 1025, 1026,
					1027, 1028]
test_patients = test_patients + test_patients1
# 28, 29

hflip = lambda : iaa.Fliplr(1)
vflip = lambda : iaa.Flipud(1)
aug_dict = {
"_hf" : iaa.Sequential([hflip()]),
"_vf" : iaa.Sequential([vflip()]),
}

for pfolder in os.listdir(raw_path):
	images_p = os.listdir(raw_path+pfolder)
	for fname in images_p:
		# if ("37_11" not in fname):
		# 	continue 
		print(fname)
		patient = int(fname.split("_")[0])

		if patient not in train_patients+test_patients:
			continue

		pathtopatientfile = raw_path+pfolder+"/"+fname
		if (".csv" in fname) & ((fname[:-4]+".tif" in images_p)or(fname[:-4]+".jpg" in images_p)):
			### read image abd bbx
			if int(fname.split("_")[0]) in [37,38,39]:
				img = io.imread(pathtopatientfile[:-4]+".jpg",as_gray=True)
				### above is read in between 0 and 1 so we convert to 0-255 using below
				img = img_as_ubyte(img)
			else:
			#remove
				img = io.imread(pathtopatientfile[:-4]+".tif")
			print(img.max(), img.min(), img.dtype)
			mwhlist = ut.read_bbox(pathtopatientfile[:-4]+".csv")
			bxmin,bymin,bxmax,bymax = [max(i) for i in zip(*[ut.mid2ltrb(mwh) for mwh in mwhlist])]

			# print("X"*30)
			# print(patient, img.shape)
			# print("X"*30)

			# #######################################################################
			# ### dump with bbox drawn on top the images
			# #######################################################################
			# lbwhlist = [ut.mid2lb(mwh) for mwh in mwhlist] ### convert to lbwh format required by matplolib Rectangle
			# plt.imshow(img, cmap="gray")
			# for lbwh in lbwhlist:
			# 	ut.draw_rect(plt.gca(), lbwh)
			# plt.savefig(ver1_path+fname[:-4]+".jpg")
			# plt.close()

			#######################################################################
			### dump the 800x800 train/test data with augmentations
			#######################################################################
			# sizeproblem = [47, 49, 50]

			right_crop800 = bxmax <= 800
			bot_crop800 = bymax <= 800

			top_crop = bymin >= 200
			left_crop = bxmin >= 200

			right_crop1k = bxmax <= 1024
			bot_crop1k = bymax <= 1024

			if right_crop800 and bot_crop800:
				img800 = img[:800,:800]
				mwh800 = mwhlist

				mwh800y = []
				for b800y in mwh800:
					b800y = [j/800 for j in b800y]
					mwh800y.append(b800y)

			### for the cases where we need to remove the crop from the starting, we need to change bbox too.
			elif top_crop and left_crop and right_crop1k and bot_crop1k:
				img800 = img[200:1000, 200:1000]
				
				mwh800 = []
				for b in mwhlist:
					b800 = [i-200 for i in b[:2]] + b[2:]
					mwh800.append(b800)

				mwh800y = []
				for b800y in mwh800:
					b800y = [j/800 for j in b800y]
					mwh800y.append(b800y)

			assert img800.shape == (800,800), fname + " has shape of "+str(img.shape)+" so not converted to 800x800"
			

			io.imsave(currver_path+fname[:-4]+"_orig.jpg", clip397(img800).astype(int))
			np.savetxt(currver_path+fname[:-4]+"_orig_yolo.txt", mwh800y, fmt=("%f "*4)[:-1])

			### augmenting image and the bbox
			for augname, augseq in aug_dict.items():
				aug_img = augseq.augment_image(img800)
				io.imsave(currver_path+fname[:-4]+augname+".jpg", clip397(aug_img).astype(int))

				mwhaug = []
				for bb800 in mwh800:
					top_left = [bb800[0]-bb800[2]/2.,bb800[1]+bb800[3]/2.]
					bottom_right = [bb800[0]+bb800[2]/2.,bb800[1]-bb800[3]/2.]
					bbx_tlbr = ia.BoundingBoxesOnImage.from_xyxy_array(xyxy=np.array([top_left+bottom_right]),shape=img800.shape)
					bb_aug = augseq.augment_bounding_boxes([bbx_tlbr])
					bb_aug = list(bb_aug[0].to_xyxy_array()[0])
					bb_aug_xymiwh = [(bb_aug[0]+bb_aug[2])/2., (bb_aug[1]+bb_aug[3])/2., bb_aug[2]-bb_aug[0], bb_aug[3] - bb_aug[1]]

					mwhaug.append([j/800. for j in bb_aug_xymiwh])
				np.savetxt(currver_path+fname[:-4]+augname+"_yolo.txt", mwhaug, fmt=("%f "*4)[:-1])


imgtypes = ["orig", "vf", "hf"]
def get_img_list(patient, typ, datapath):
    images = []
    fnames = []
    for file in os.listdir(datapath):
        if (".jpg" in file) and (typ in file) and (file.startswith(patient)) and ("agg" not in file):
            img = io.imread(datapath+file)
            images.append(np.array(img, dtype="float64"))
            fnames.append(file)
    return images,fnames

for ptnt in train_patients+test_patients:
	ptnt = str(ptnt)+"_"
	for typ in imgtypes:
		imglist,fnames = get_img_list(ptnt, typ, currver_path)
		aggdiff = np.zeros((800,800),dtype='float64')
		aggimg_wgh = np.zeros((800,800),dtype="float64")
		for i in range(len(imglist)):
			if i==0:
				continue
			img_curr = clip397(np.array(imglist[i], dtype="float64"))
			img_prev = clip397(np.array(imglist[i-1], dtype="float64"))
			aggimg_wgh = clip397(aggimg_wgh + (i)**3*img_curr)
			diff = img_curr - img_prev
			diff[diff>0] *= 1
			diff[diff<=0] *= 3
			aggdiff += clip397(diff)
		# aggimg_wgh = aggimg_wgh - aggimg_wgh.mean()
		# aggimg_wgh = aggimg_wgh / aggimg_wgh.std()
		# plt.imshow(aggdiff, cmap="gray")
		# plt.savefig(currver_path+ptnt+"_"+typ+"_avg.jpg")
		# new_arr = ((aggimg_wgh - aggimg_wgh.min()) * (1/(aggimg_wgh.max() - aggimg_wgh.min()) * 255)).astype('uint8')
		# print("here")
		aggimg_wgh = exposure.rescale_intensity(aggimg_wgh.astype("int32"), out_range=(0, 2**31 - 1))
		aggdiff = exposure.rescale_intensity(aggdiff.astype("int32"), out_range=(0, 2**31 - 1))

		io.imsave(currver_path+ptnt+typ+"_aggimg.jpg", aggimg_wgh)
		io.imsave(currver_path+ptnt+typ+"_aggdiff.jpg", aggdiff)


files = [i for i in os.listdir(currver_path) if (".jpg" in i) and ("agg" not in i)]
f_tr = open(currver_path+'train.txt','w')
f_te = open(currver_path+'test.txt','w')
for ptnt in train_patients:
	ptnt_files = [i for i in files if i.startswith(str(ptnt)+"_")]
	fnums = set([int(i.split("_")[1]) for i in ptnt_files])
	fnums = sorted(fnums)
	last5 = fnums[-7:]
	ptntfnames5 = [str(ptnt)+"_"+str(i) for i in last5]
	for fname in files:
		fname_gen = "_".join(fname.split(".")[0].split("_")[:2])
		if (fname_gen in ptntfnames5) and ("orig" in fname):
			f_tr.write("data/bleeds5/"+fname+"\n")

for ptnt in test_patients:
	ptnt_files = [i for i in files if i.startswith(str(ptnt)+"_")]
	fnums = set([int(i.split("_")[1]) for i in ptnt_files])
	fnums = sorted(fnums)
	last5 = fnums[-5:]
	ptntfnames5 = [str(ptnt)+"_"+str(i) for i in last5]
	for fname in files:
		fname_gen = "_".join(fname.split(".")[0].split("_")[:2])
		if (fname_gen in ptntfnames5) and ("orig" in fname):
			f_te.write("data/bleeds5/"+fname+"\n")

f_tr.close()
f_te.close()

shutil.make_archive(base_name = currver_path, format ='zip', root_dir = currver_path, base_dir = None)
