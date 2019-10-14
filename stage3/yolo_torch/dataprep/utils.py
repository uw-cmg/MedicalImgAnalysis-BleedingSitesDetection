import csv
import os
from skimage import io
from collections import Counter
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt


def read_bbox(csvfilepath):
    """
    returns x, y, width, height from the csv dumps we have
    """
    with open(csvfilepath) as csvfile:
        out_coords = []
        reader = csv.DictReader(csvfile)
        for row in reader:
            ### reading the one/first bbx
            out_coords.append([int(float(row['X'])), int(float(row['Y'])), int(float(row['Width'])), int(float(row['Height']))])
        return out_coords


def mid2lb(xy_m_wh):
	"""
	matplotlib rectangle needs coords in this format
	"""
	lb = [0]*4
	lb[0] = int(xy_m_wh[0]-xy_m_wh[2]/2)
	lb[1] = int(xy_m_wh[1]-xy_m_wh[3]/2)
	lb[2:] = xy_m_wh[2:]
	return lb


def mid2ltrb(xy_m_wh):
    """
    takes xy mid, wh as input and return xy left top, xy right bottom
    """
    xy_ltrb = [0]*4
    xy_ltrb[0] = int(xy_m_wh[0] - xy_m_wh[2]/2)
    xy_ltrb[1] = int(xy_m_wh[1] - xy_m_wh[3]/2)
    xy_ltrb[2] = int(xy_m_wh[0] + xy_m_wh[2]/2)
    xy_ltrb[3] = int(xy_m_wh[1] + xy_m_wh[3]/2)
    return xy_ltrb

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
    axis.add_patch(Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='g', facecolor='none'))


def imgbboxdump(fname, nparr, bbox):
    plt.figure()
    plt.imshow(nparr, cmap="gray")
    draw_rect(plt.gca(),bbox)
    plt.savefig(fname, dpi=300)
    plt.close()


def proper_im_num(num_string):
    """
    for now expecting both the patient number and image number be <=99
    converts "15_6" to 1506
    """
    return int("".join(["{:02d}".format(int(i)) for i in  num_string.split("_")]))


def dump_data_info(raw_path):
    # raw_path = "raw_data/bleeding_cases/"
    dataarr = np.empty((0,8), dtype='O')
    for pfolder in os.listdir(raw_path):
        images_p = os.listdir(raw_path+pfolder)
        for fname in images_p:
            pathtopatientfile = raw_path+pfolder+"/"+fname
            if (".csv" in fname) & (fname[:-4]+".tif" in images_p):
                ### read image abd bbx
                img = io.imread(pathtopatientfile[:-4]+".tif")
                mwhlist = ut.read_bbox(pathtopatientfile[:-4]+".csv")
                ltrb_arr = np.array([ut.mid2ltrb(mwh) for mwh in mwhlist])
                minx = ltrb_arr[:,[0,2]].min()
                maxx = ltrb_arr[:,[0,2]].max()
                miny = ltrb_arr[:,[1,3]].min()
                maxy = ltrb_arr[:,[1,3]].max()
                
                nbbx = len(mwhlist)
                shape = "x".join([str(i) for i in img.shape])
                
                dataarr = np.vstack((dataarr, [fname.split("_")[0],fname[:-4], nbbx, shape, minx, maxx, miny, maxy]))

    np.savetxt("datainfo.csv",dataarr, delimiter=',', fmt='%s')