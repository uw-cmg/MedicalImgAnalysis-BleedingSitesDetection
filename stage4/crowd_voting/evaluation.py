"""
This is the code for analyzing detection results, calculating patient score. Generate .txt for all the bbox labels respectively and name them by patientID_frameId.txt
                                                                             Format: Ymin Xmin Ymax Xmax
                                                                             eg., 361.0 293.0 397.0 325.0 
                                                                             file name example: 48_12.txt
                                                                             
@detect_res_file: the txt file contains the detection results. Format: jpg name,Ymin,Xmin,Ymax,Xmax,score
                                                               jpg name format:patientId_frameId.jpg
                                                               eg: 18_10.jpg,491.96744,113.14628,533.2568,153.23444,0.9348341,0.7855689961758875
@test_id: the tested patient or you wanna test all patients, input[]
          eg:[18,48,60]

The output is written into eva_result.txt: format: patient id(missing if all patients are tested), patient score
                                            eg.18,1.0
"""   

import numpy as np

detect_res_file = "crowd_vote_result.txt"
patient_ids = ['1012', '1005', '1006', '1017', '1004', '1019', '1010', '1016', '1026', '1014', '1011', '1002', '1024', '1028', '1009', '1021', '1027', '1025', '1003']

#this is the function for calculating iou
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
    iou = area_overlap / (area_combined+epsilon)
    return iou

#this is the function for calculating patient score
def patient_score_eva(file_name,patient_id):
    with open(file_name,'r') as f:
        lines = f.readlines()
        info=list()
        img_name=list()
        label_boxes = dict()
        for line in lines:
            cur_patient = int(line.split("_")[0])
            if cur_patient==patient_id or patient_id==-1:
                sp=line.split("jpg")
                cur_img_name = sp[0]            
                
                cur_frame = int(cur_img_name.split("_")[0].replace(".",""))
                cur_box_info = sp[1].split(",")

                #find the label for cur img
                if cur_img_name in label_boxes:
                    label_box=label_boxes[cur_img_name]
                else:
                    cur_img_label = cur_img_name.replace(".",".txt")
                    label_box=[-1,-1,-1,-1]
                    with open(cur_img_label) as f:
                            cur_content = f.read().strip('\n')
                            ls_label_box = cur_content.split(" ")                
                            for i in range(4):  
                                label_box[i]=float(ls_label_box[i])
                    label_boxes[cur_img_name]=label_box            
                
                #read loc for cur img and calculate iou
                cur_box=[-1,-1,-1,-1]
                for i in range(4):
                    cur_box[i]=float(cur_box_info[i+1])        
                cur_iou = bbox_iou(cur_box,label_box)
                #read score for cur img
                cur_score = float(cur_box_info[5].replace("\n",""))

                cur_info=[cur_patient,cur_frame,cur_iou,cur_score]
                info.append(cur_info)
                img_name.append(cur_img_name)

    m=np.array(info)
    select_res=list()

    #if one frame has several bounding boxes, then selecet the one with max score
    for cur_img_name in set(img_name):
        cur_patient = int(cur_img_name.split("_")[0])
        cur_frame = int(cur_img_name.split("_")[0].replace(".",""))
        max_score=-1
        select_img = -1
        for i in range(len(m)):
            if m[i,0]==cur_patient and m[i,1]==cur_frame and m[i,3]>max_score:
                max_score=m[i,1]
                select_img=i
        select_res.append(m[select_img,])

    #find the number of res with iou>0
    r=np.array(select_res)
    count=0
    for i in range(len(r)):
        if r[i,2]>0:
            count+=1

    patient_score = count/len(r)
    return patient_score


def main(detect_res_file, patient_ids):
    if len(patient_ids) == 0:
        return str(patient_score_eva(detect_res_file, -1))
    else:
        score=list()
        for patient_id in patient_ids:
            cur_score = str(patient_score_eva(detect_res_file,patient_id))
            score.append(str(patient_id)+","+cur_score)
        return score


if __name__ == '__main__':
    eva_result = main(detect_res_file, patient_ids)
    with open('eva_result.txt', 'w') as f:
        for item in eva_result:
            f.write("%s" % item)