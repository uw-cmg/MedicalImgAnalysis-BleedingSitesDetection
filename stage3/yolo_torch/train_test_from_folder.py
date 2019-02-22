import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--image_folder", type=str, default="data/bleeds/", help="path to dataset")
parser.add_argument("--tmp_arg", type=str, default="bleeds1", help="path to dataset")

opt = parser.parse_args()
print(opt)

def patient_train_test(image_folder,patients_for_test):
	"""
	image_folder - image_folder to the bleeds data folder containing images and labels
	patients_for_test - int list of patient numbers for testing

	"""
	files = [i for i in os.listdir(image_folder) if ".jpg" in i]
	f_tr = open(image_folder+'train.txt','w')
	f_te = open(image_folder+'test.txt','w')
	for i in files:
		patient = int(i.split("_")[0])
		if patient in patients_for_test:
			f_te.write(image_folder+i+"\n")
		else:
			f_tr.write(image_folder+i+"\n")
	f_tr.close()
	f_te.close()

if opt.tmp_arg=="bleeds1":
	patient_train_test(opt.image_folder, [6,11,15])