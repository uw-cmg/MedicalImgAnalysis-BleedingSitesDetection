import numpy as np

"""
This is the code for crowd voting.

@all_bbox_file: the txt file contains all the bbox for all the patients.
                Format: jpg name,Ymin,Xmin,Ymax,Xmax,score
                jpg name format:patientId_frameId.jpg
                eg:48_20.jpg,532.9326,388.33987,576.0951,432.8795,0.10115564

@patient_ids: an array of patient id
              eg:["18","48","60"]

Output: The information of voted bounding box; one patient, one row, one bounding box.
        The output is written to crowd_vote_result.txt. Format: jpg name,Ymin,Xmin,Ymax,Xmax,score
                                                        jpg name format:patientId_frameId.jpg
        eg:18_10.jpg,491.96744,113.14628,533.2568,153.23444,0.9348341,0.7855689961758875
           48_24.jpg,357.22125,299.84146,395.42416,330.64755,0.9572147,0.6061841445094583
           60_33.jpg,692.35065,340.7826,740.7708,380.7977,0.9680265,0.9008500634163339

"""




patient_ids = ['1005', '1006', '1017', '1004', '1019', '1010', '1016', '1026',
               '1011', '1002', '1024', '1028', '1009', '1021', '1027', '1025', '1003']


# this is the function for calculating IoU
def bbox_iou(a, b):
    epsilon = 1e-5
    # COORDINATES OF THE INTERSECTION BOX
    y1 = max(a[0], b[0])
    x1 = max(a[1], b[1])
    y2 = min(a[2], b[2])
    x2 = min(a[3], b[3])

    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width < 0) or (height < 0):
        return 0.0
    area_overlap = width * height

    # COMBINED AREA
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined + epsilon)
    return iou


# this is the function for crowd voting
def crowd_vote(all_bbox_file, patient_id):
    # read corresponding patient bbox info
    with open(all_bbox_file, 'r') as f:
        lines = f.readlines()
        info = list()
        box_sum = list()
        for line in lines:
            get_id = line.split("/")
            if get_id[2][-4:] != patient_id:
                continue
            box = line.strip().split(".tif,")[1].split(",")
            cur_box = [-1, -1, -1, -1]
            for i in range(4):
                cur_box[i] = float(box[i])
            box_sum.append(cur_box)
            info.append(line)

            # make the matrix
    n_frame = len(box_sum)
    fusion = np.zeros(shape=(n_frame, n_frame))
    for i in range(n_frame):
        for j in range(n_frame):
            if (i == j):
                fusion[i, j] = 0
            else:
                fusion[i, j] = bbox_iou(box_sum[i], box_sum[j])

    # calculate avg and vote
    max_mean = -1
    vote = -1
    for i in range(n_frame):
        cur_mean = np.mean(fusion[i])
        if (cur_mean > max_mean):
            vote = i
            max_mean = cur_mean

    return info[vote]


def main(all_bbox_file, patient_ids):
    vote_result = list()
    for patient_id in patient_ids:
        vote_result.append(crowd_vote(all_bbox_file, patient_id))
    return vote_result


if __name__ == '__main__':
    all_bbox_file = "crowd_vote_1/all_bbox_n=3.txt"
    vote_result = main(all_bbox_file, patient_ids)
    with open('crowd_vote_1/crowd_vote_result_all_bbox.txt', 'w') as f:
        for item in vote_result:
            f.write("%s" % item)
