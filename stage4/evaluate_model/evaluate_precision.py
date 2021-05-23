"""Evaluate the precision of trained weight of an mxnet model.

  Typical usage example:

  model_name = 'yolo3_darknet53_custom'
  first_n_prediction = 3
  iou_threshold = [0.0001, 0.1, 0.2]
  param_name = 'freeze-half-yolo3_darknet53_custom_best.params'
"""

from gluoncv import model_zoo, data, utils
import mxnet as mx
import numpy as np
import cv2
import csv
from os import listdir
from os.path import isfile, join


def getBbox(csv_fname):
    """Get the bounding box of a image

    Retrieve the bounding box(es) in the csv file by using the csv reader. 
    Bounding box format: [xmin, ymin, xmax, ymax]

    Args:
        csv_fname: file path to label file ending with .csv

    Returns:
        an array of bounding boxes
    """
    bboxes = []
    with open(csv_fname, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            bboxes.append([int(row["BX"]),
                           int(row["BY"]),
                           int(row["BX"]) + int(row["Width"]),
                           int(row["BY"]) + int(row["Height"])])
    return bboxes


def evaluate_single(net, im_fname, first_n_prediction, iou_threshold):
    """Evaluate a single image

    Compute iou of first n prediction and ground truth box. 

    Args:
        net: model with trained weights
        im_fname: file path to image file ending with .tif
        first_n_prediction: number of prediction bounding box 
        iou_threshold: an array of iou threshold, e.x. [0.001, 0.1, 0.2]

    Returns:
        an array of 0 and 1, where 0 means the iou between predicted bbox is 
        less than the corresponding threshold
    """
    im = cv2.imread(im_fname)
    x, img = data.transforms.presets.yolo.load_test(im_fname, short=im.shape[0])

    class_IDs, scores, predicted_bounding_boxs = net(x)
    ground_truth_boxes = np.array(getBbox(im_fname.replace('.tif', '.csv')))
    result = [0] * len(iou_threshold)
    max_iou = 0
    for ground_truth_box in ground_truth_boxes:
        for box in predicted_bounding_boxs[0][:first_n_prediction]:
            box = box.asnumpy()
            xmin = max(box[0], ground_truth_box[0])
            ymin = max(box[1], ground_truth_box[1])
            xmax = min(box[2], ground_truth_box[2])
            ymax = min(box[3], ground_truth_box[3])
            inter_area = max(0, xmax - xmin + 1) * max(0, ymax - ymin + 1)
            ground_truth_box_area = (ground_truth_box[2] - ground_truth_box[0] + 1) * (
                    ground_truth_box[3] - ground_truth_box[1] + 1)
            box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
            iou = inter_area / float(ground_truth_box_area + box_area - inter_area)
            max_iou = max(iou, max_iou)
        for i in range(len(iou_threshold)):
            if max_iou > iou_threshold[i]:
                result[i] = 1
    return result, max_iou


def evaluate_all(net, files_path, first_n_prediction, iou_threshold, param_name, performance_dic, fw1, patient_iou, fw2):
    """Evaluate testing images

    Compute and print the precision of trained weight

    Args:
        net: model with trained weights
        files_path: file containing all testing file pathes
        first_n_prediction: number of prediction bounding box 
        iou_threshold: an array of iou threshold, e.x. [0.001, 0.1, 0.2]
        performance_dic: a dictionary stores performance of all params under ious
        param_name: name of parameter

    Returns:
        Print the precision of each iou threshold
    """
    files = np.loadtxt(files_path, dtype=np.str, delimiter='/n')
    num_of_files = 0
    num_correct = [0] * len(iou_threshold)

    # patients = {}

    for file in files:
        patient_id = file.split("/")[2]
        # if patient_id not in patients:
        #     patients[patient_id] = [[0] * len(patient_iou), 0.0]
        single_result, iou = evaluate_single(net, file, first_n_prediction, iou_threshold)
        print(file, iou)
        fw2.write("{}: {}\n".format(file, str(iou)))
        for i in range(len(iou_threshold)):
            # patients[patient_id][0][i] += single_result[i]
            num_correct[i] += single_result[i]
        # patients[patient_id][1] += 1
        num_of_files += 1

    precision_under_ious = []
    for i in range(len(iou_threshold)):
        precision = num_correct[i] / num_of_files
        precision_under_ious.append(precision)

    # num_patient_correct_under_iou = [0] * len(patient_iou)
    # for k in patients:
    #     print("{}: {}".format(k, str(patients[k][0][0] / patients[k][1])))
    #     for i in range(len(patient_iou)):
    #         if (patients[k][0][0] / patients[k][1]) >= patient_iou[i]:
    #             print(k, str(patient_iou[i]))
    #             num_patient_correct_under_iou[i] += 1

    # print(num_patient_correct_under_iou)
    # for i in range(len(patient_iou)):
    #     fw.write("Patient_level_iou = {}: {}\n".format(patient_iou[i], num_patient_correct_under_iou[i] / len(patients)))

    fw1.write("image_level_iou = {}".format(" ".join(map(str, precision_under_ious)) + "\n"))
    performance_dic[param_name] = precision_under_ious
    return num_correct, num_of_files


def main():
    eval_param_name_list = ["yolo3_darknet53_custom_" + format(x, '04') for x in range(100, 1001, 100)]
    # eval_param_name_list = ["yolo3_darknet53_custom_" + format(x, '04') for x in range(10, 301, 10)]
    # eval_param_name_list.append("yolo3_darknet53_custom_best.params")
    model_name = 'yolo3_darknet53_custom'
    c = np.array(['bleeding_site'])  # numpy array of class name.
    net = model_zoo.get_model(model_name, pretrained=False, classes=c)
    first_n_prediction = 5
    iou_threshold = [0.1, 0.3, 0.5]
    patient_iou = [0.1, 0.3, 0.5, 0.7]
    test_filename = 'dataset/validation_set.txt'

    y = [30, 31, 32, 33, 34]
    param_path_list = ['yolo_' + str(x) + '/' for x in [25, 26]]

    for param_path in param_path_list:
        all_filename = [f for f in listdir(param_path) if isfile(join(param_path, f))]
        performance_dic = {}
        ##########################################
        # write yolo_i/evaluation.txt
        ##########################################
        with open(param_path + 'evaluation_validation_set_100.txt', 'w') as fw1, open(param_path + 'evaluation_validation_set_100.log', 'w') as fw2:
            fw1.write("model name: {}\nfirst_n_prediction: {}\niou threshold: {}\ntest_filename: {}\n".format(
                model_name, first_n_prediction, iou_threshold, test_filename))
            for param in sorted(list(filter(lambda x: x.startswith(tuple(eval_param_name_list)), all_filename))):
                param_name = param_path + param
                print("Evaluating {}".format(param_name))
                fw1.write(param_name + "\n")
                net.load_parameters(param_name)
                num_correct, num_of_files = evaluate_all(net, test_filename, first_n_prediction, iou_threshold, param_name, performance_dic, fw1, patient_iou, fw2)
            performance_dic = dict(sorted(performance_dic.items(), key=lambda item: item[1][0], reverse=True))
            fw1.write("Top 3 parameter are\n")
            fw1.write(str({k: performance_dic[k] for k in list(performance_dic)[:3]}) + "\n")


if __name__ == "__main__":
    main()
