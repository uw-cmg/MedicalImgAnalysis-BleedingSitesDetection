import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import sys

from PIL import Image

from parse_config import *
from utils import build_targets
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import logging
logging.getLogger('logfile.log')


def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0) 
    #dbt why is this pop here? understand the module_defs object!
    output_filters = [int(hyperparams["channels"])]
    #dbt why do we need to keep track of the output filters??
    module_list = nn.ModuleList()
    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2 if int(module_def["pad"]) else 0
            modules.add_module(
                "conv_%d" % i,
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias= not bn,
                ),
            )
            if bn:
                modules.add_module("batch_norm_%d" % i, nn.BatchNorm2d(filters))
            if module_def["activation"] == "leaky":
                modules.add_module("leaky_%d" % i, nn.LeakyReLU(0.1))

        ### #dbt why is this here! no maxpool is used! remove #todo
        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                padding = nn.ZeroPad2d((0, 1, 0, 1))
                modules.add_module("_debug_padding_%d" % i, padding)
            maxpool = nn.MaxPool2d(
                kernel_size=int(module_def["size"]),
                stride=int(module_def["stride"]),
                padding=int((kernel_size - 1) // 2),
            )
            modules.add_module("maxpool_%d" % i, maxpool)

        elif module_def["type"] == "upsample":
            upsample = Interpolate(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module("upsample_%d" % i, upsample)

        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")] #dbt
            filters = sum([output_filters[layer_i] for layer_i in layers]) #dbt
            modules.add_module("route_%d" % i, EmptyLayer()) #dbt read about the empty layer again

        elif module_def["type"] == "shortcut":
            filters = output_filters[int(module_def["from"])]
            modules.add_module("shortcut_%d" % i, EmptyLayer())

        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs] #filter only for the anchors in the mask
            ##### num_classes = int(module_def["classes"])
            img_height = int(hyperparams["height"])
            # Define detection layer
            ##### yolo_layer = YOLOLayer(anchors, num_classes, img_height)
            yolo_layer = YOLOLayer(anchors, img_height)
            modules.add_module("yolo_%d" % i, yolo_layer)
        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list

### using interpolate layer as upsample is depricated
class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = F.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode)
        return x

class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()

class YOLOLayer(nn.Module):
    """Detection layer"""

    ##### def __init__(self, anchors, num_classes, img_dim):
    def __init__(self, anchors, img_dim):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        ##### self.num_classes = num_classes
        ##### self.bbox_attrs = 5 + num_classes
        self.bbox_attrs = 5
        self.image_dim = img_dim
        self.ignore_thres = 0.5  ### this ignore thresh is for the threshold for anchor box intersection with the ground truth
        self.lambda_coord = 1 #dbt what is lambda coord?

        self.mse_loss = nn.MSELoss(reduction='mean')  # Coordinate loss
        self.bce_loss = nn.BCELoss(reduction='mean')  # Confidence loss
        ##### self.ce_loss = nn.CrossEntropyLoss()  # Class loss

    def forward(self, x, targets=None):
        
        ### input coming in is the batch size, channels, then gxg the number of grids
        num_anchs = self.num_anchors
        bsize = x.size(0) 
        nG = x.size(2) ### number of grids, formed from the previous layer 
        stride_wrt_orig = self.image_dim / nG ### we calculate this stride value wrt. the original image to scale the anchor boxes accordingly
        ### basically this has the equivalent effect of doing sliding window at different **scales**

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        prediction = x.view(bsize, num_anchs, self.bbox_attrs, nG, nG).permute(0, 1, 3, 4, 2).contiguous()
        ### view just gives a reshaped view, .contiguous() puts all data points in memory together contiguously 

        ### #dbt how does this data get transformed by the above line such that we have below?!!?

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        ##### pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred. 
        # print(w)
        # sys.exit()

        ### Calculate offsets for each grid. The code below basically creates cells with their index number.
        ### as the xy coords are between 0-1 for each cell, we are adding the offset from the start
        grid_x = torch.arange(nG).repeat(nG, 1).view([1, 1, nG, nG]).type(FloatTensor)
        grid_y = torch.arange(nG).repeat(nG, 1).t().view([1, 1, nG, nG]).type(FloatTensor)

        ### we scale the anchor boxes wrt. the current scale of detection
        ### #dbt but why do we divide the anchors with stride_wrt_orig = 800/100 = 8!??, they are not 8 times smaller?! even when we have dim==800, the anchors are only abt 1-200 big right? so why would be divide by 8?
        ### #major 
        scaled_anchors = FloatTensor([(a_w / stride_wrt_orig, a_h / stride_wrt_orig) for a_w, a_h in self.anchors])

        ### change dimension so later when we multiply, we multiply to the right dimension
        anchor_w = scaled_anchors[:, 0:1].view((1, num_anchs, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, num_anchs, 1, 1))

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape) ### :4 is the x,y,w,h
        ### adding the x,y to the actual grid numbers, so now they have actual centers!
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        ### #dbt learn abt by do we do exp of w, h before multiplying to anchor box dims?
        ### also, we are basically scaling the w,h based on the anchor box here
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        ### print(targets)
        ### targets here is set to 50x5 allowing a maximum of 50 objects to be detected.
        ### #todo reduce this to max 5 allowed objects(in case of texting we might need more.) but if training and testing is independent of the number of targets here, make it just 1
        # Training
        if targets is not None:

            if x.is_cuda:
                self.mse_loss = self.mse_loss.cuda()
                self.bce_loss = self.bce_loss.cuda()
                ##### self.ce_loss = self.ce_loss.cuda()

            ##### n_real_pos, n_true_pos, mask, conf_mask, tx, ty, tw, th, tconf, tcls = build_targets(
            n_real_pos, n_true_pos, mask, conf_mask, tx, ty, tw, th, tconf = build_targets(
                pred_boxes=pred_boxes.cpu().data,
                pred_conf=pred_conf.cpu().data,
                ##### pred_cls=pred_cls.cpu().data,
                target=targets.cpu().data,
                anchors=scaled_anchors.cpu().data,#dbt what?
                num_anchors=num_anchs,
                ##### num_classes=self.num_classes,
                grid_size=nG,
                ignore_thres=self.ignore_thres,
                img_dim=self.image_dim,
            )

            n_pred_pos = int((pred_conf > 0.7).sum().item())
           
            rcll = float(n_true_pos / n_real_pos) if n_real_pos else 1
            if n_pred_pos:
               prcn = float(n_true_pos / n_pred_pos)
            elif n_true_pos:
               prcn = 0
            else:
               prcn = 1

            # Handle masks
            mask = Variable(mask.type(ByteTensor))
            conf_mask = Variable(conf_mask.type(ByteTensor))

            # Handle target variables
            tx = Variable(tx.type(FloatTensor), requires_grad=False)
            ty = Variable(ty.type(FloatTensor), requires_grad=False)
            tw = Variable(tw.type(FloatTensor), requires_grad=False)
            th = Variable(th.type(FloatTensor), requires_grad=False)
            tconf = Variable(tconf.type(FloatTensor), requires_grad=False)
            ##### tcls = Variable(tcls.type(LongTensor), requires_grad=False)

            # Get conf mask where gt and where there is no gt
            conf_mask_true = mask
            conf_mask_false = conf_mask - mask

            # Mask outputs to ignore non-existing objects
            loss_x = self.mse_loss(x[mask], tx[mask])
            loss_y = self.mse_loss(y[mask], ty[mask])
            loss_w = self.mse_loss(w[mask], tw[mask])
            loss_h = self.mse_loss(h[mask], th[mask])
            ### #dbt why adding 2 losses here?
            loss_conf = self.bce_loss(pred_conf[conf_mask_false], tconf[conf_mask_false]) + self.bce_loss(
                pred_conf[conf_mask_true], tconf[conf_mask_true]
            )
            ##### loss_cls = (1 / bsize) * self.ce_loss(pred_cls[mask], torch.argmax(tcls[mask], 1))
            ##### loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf            
            return (
                loss,
                loss_x.item(),
                loss_y.item(),
                loss_w.item(),
                loss_h.item(),
                loss_conf.item(),
                ##### loss_cls.item(),
                n_true_pos,
                n_real_pos,
                n_pred_pos,
                rcll,
                prcn,
            )

        else:
            ### If not in training phase return predictions
            output = torch.cat(
                (
                    pred_boxes.view(bsize, -1, 4) * stride_wrt_orig,
                    pred_conf.view(bsize, -1, 1),
                    ##### pred_cls.view(bsis, -1, self.num_classes),
                ),
                -1,
            )
            return output

class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.img_size = img_size
        ##### self.loss_names = ["x", "y", "w", "h", "conf", "cls", "rcll", "prcn"]
        self.notes_names = ["x", "y", "w", "h", "conf", "ntp", "nrp", "npp", "rcll", "prcn"]
        

    def forward(self, x, targets=None, itrdet=None):
        is_training = targets is not None
        self.notes = defaultdict(float)
        self.notes['ntp'] = []
        self.notes['nrp'] = []
        self.notes['npp'] = []
        output = [] ### this is for yolo layers
        layer_outputs = [] ### this is book keeping for other layers when we have dp dp routing
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            #print("inp",module_def["type"], x.shape,module)
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "route":
                layer_i = [int(x) for x in module_def["layers"].split(",")]
                x = torch.cat([layer_outputs[i] for i in layer_i], 1)
                ### the 1 above is coz we are concatenating across the channels(depth)
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                # Train phase: get loss
                if is_training:
                    x, *notevals = module[0](x, targets)
                    for name, noteval in zip(self.notes_names, notevals):
                        if name in ["nrp","ntp","npp"]:
                            self.notes[name] += [noteval]
                        else:
                            self.notes[name] += noteval
                # Test phase: Get detections
                else:
                    #print(module)
                    x = module(x) ### #dbt check why the call signature is different for training and testing? model[0](x,targ) vs model(x). targets is optional so that is ok. but why model[0] and model?
                output.append(x)
            layer_outputs.append(x)
        self.notes["rcll"] /= 3 ## divide by 3 coz we have these from 3 yolo layers and they get added
        self.notes["prcn"] /= 3

        out = sum(output) if is_training else torch.cat(output, 1)

        if is_training and (itrdet['curr_batch'] % 50==0):

            logging.info("[epch {}/{}, btch {}/{}] [lss: x {:.3f}, y {:.3f}, w {:.3f}, h {:.3f}, conf {:.3f}, tot {:.3f}]".format( itrdet["curr_epoch"], itrdet["tot_epochs"], itrdet["curr_batch"], itrdet["tot_batches"], self.notes["x"], self.notes["y"], self.notes["w"], self.notes["h"], self.notes["conf"], out))

            logging.info("ntp:{}, nrp:{}, npp:{}, rcll: {:.3f}, prcn: {:.3f}".format(self.notes["ntp"] , self.notes["nrp"], self.notes["npp"], self.notes["rcll"] , self.notes["prcn"]))

        return out
