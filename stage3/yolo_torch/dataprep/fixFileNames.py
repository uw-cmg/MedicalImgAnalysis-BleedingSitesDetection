"""
some fines have name 45_6.tif, rename then to 45_06.tif, basically make everything 2 digit, this will be useful later
"""
import os

data_dir = "raw_data/Dataset_5_fixedNames/"
patients = os.listdir(data_dir)
for pfolder in patients:
    if pfolder==".DS_Store":
        continue
    fnames = os.listdir(data_dir+pfolder)

    for name in fnames:
        pname, extn = name.split(".") # .tif or .csv or .jpg
        splits = pname.split("_")
        if len(splits[1])==1: 
            splits[1] = '0'+splits[1]
            newname = "_".join(splits) + "."+extn
            print(name)
            print(newname)
            print("---"*10)
            os.rename(data_dir+pfolder+"/"+name, data_dir+pfolder+"/"+newname)