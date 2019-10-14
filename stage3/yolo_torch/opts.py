import argparse

def parse_opts():
    parser = argparse.ArgumentParser()

	### Logistics
	############################################
    parser.add_argument("-t", dest="temp", action="store_true")

    parser.add_argument("--epochs", type=int, default=300, help="number of epochs")
    parser.add_argument("--image_folder", type=str, default="data/withjun2data/", help="path to dataset")
    parser.add_argument("--batch_size", type=int, default=5, help="size of each image batch")
    parser.add_argument("--model_config_path", type=str, default="config/yolov3_1class.cfg"
                        , help="path to model config file")
    parser.add_argument("--data_config_path", type=str, default="config/bleeds.data"
                        , help="path to data config file")
    parser.add_argument("--weights_path", type=str, default=None
                        , help="path to weights file")
    # parser.add_argument("--class_path", type=str, default="data/bleeds.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4
                        , help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=1
                        , help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=800, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1
                        , help="interval between saving model weights")
    parser.add_argument("--use_cuda", type=bool, default=True
                        , help="whether to use cuda if available")
    parser.add_argument("--multigpu", type=bool, default=False, help="whether to use multiple gpus")
    parser.add_argument("--expname", type=str, default="exp", help="name of experiment")
	# parser.add_argument("--cudnn_benchmark",type=bool,default=False)  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
    args = parser.parse_args()

    return args
