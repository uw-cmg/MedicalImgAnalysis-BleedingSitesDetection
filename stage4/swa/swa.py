"""
 Use Stochastic Weight Averaging to improve result
"""

from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt
import numpy as np
import random
import cv2
import csv


def swa(net, param_names_list):
    params = []
    for param_name in param_names_list:
        net.load_parameters(param_name)
        params.append(net.collect_params())

    for layer_name in params[0]:
        sum = 0
        for i in range(len(params)):
            sum += params[i][layer_name].data()
        params[0][layer_name].set_data(sum / len(params))

    params[0].save("swa.params")
