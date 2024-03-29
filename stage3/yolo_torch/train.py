import os
import logging
import datetime
import torch
torch.manual_seed(321)
import opts

from models import *
from utils import *
from datasets import *
from parse_config import *

import torch.backends.cudnn as cudnn
torch.backends.cudnn.deterministic = True
cudnn.benchmark = True

cfg = opts.parse_opts()

experiment_ts = datetime.datetime.now().strftime("%d_%b_%H_%M")

logging.basicConfig(filename='trlog.log', level=logging.INFO)
logging.info("***"*30)
logging.info(checkpoint_dir)
logging.info(cfg)
logging.info("***"*30)

cuda = torch.cuda.is_available() and cfg.use_cuda

# amke checkpoint dirs
if cfg.temp:
    checkpoint_dir = "checkpoints/temp/"
else:
    checkpoint_dir = "checkpoints/"+cfg.expname+"_"+experiment_ts+"/"
os.makedirs(checkpoint_dir, exist_ok=True)

# Get data configuration
data_path = cfg.data_path
train_fold_paths = sorted([data_path+i for i in os.listdir(data_path) if "train" in i])

for train_path in train_fold_paths:
    fold = train_path.split("/")[2].split("_")[0]
    fold_checkpoint_dir = checkpoint_dir+fold+'/'
    os.makedirs(fold_checkpoint_dir, exist_ok=True)

    # # Get hyper parameters
    # hyperparams = parse_model_config(cfg.model_config_path)[0]
    # learning_rate = float(hyperparams["learning_rate"])
    # momentum = float(hyperparams["momentum"])
    # decay = float(hyperparams["decay"])
    # burn_in = int(hyperparams["burn_in"])

    # Initiate model
    model = Darknet(cfg.model_config_path)
    if cfg.weights_path != None:
        model.load_state_dict(torch.load(cfg.weights_path))
    else:
        model.apply(weights_init_normal)

    device = torch.device("cuda:0" if cuda else "cpu")

    if (torch.cuda.device_count() > 1) and (cfg.multigpu):
        logging.info("$$$"*40)
        logging.info("using "+str(torch.cuda.device_count())+" GPUs in data parallel mode" )
        for i in range(torch.cuda.device_count()):
            logging.info("device "+ str(i) +" - "+ torch.cuda.get_device_name(i))
        logging.info("$$$"*40)
        model = torch.nn.DataParallel(model)

    if cuda:
        model = model.cuda()

    model.to(device) 

    model.train() ### .train is a nn.Module class

    # Get dataloader
    dataloader = torch.utils.data.DataLoader(ListDataset(train_path), batch_size=cfg.batch_size, shuffle=False)

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters())) 

    for epoch in range(cfg.epochs):
        for batch_i, (img_path, imgs, targets) in enumerate(dataloader):
            if batch_i%20==0:
                print(f"{fold}/{5}, {epoch}/{cfg.epochs}, {batch_i}")
            imgs = torch.autograd.Variable(imgs.type(Tensor)) #dbt is requires grad true by default for Variables?
            targets = torch.autograd.Variable(targets.type(Tensor), requires_grad=False) ### constant, doesnot require gradients!

            optimizer.zero_grad()

            iter_details = {"curr_epoch":epoch, "tot_epochs":cfg.epochs, "curr_batch":batch_i, "tot_batches":len(dataloader)}
            
            loss = model(imgs, targets, iter_details) ### this basically calls the forward function. and since that function returns the loss, we receive the loss in loss variable

            if cfg.multigpu: 
                loss.sum().backward()
            else:
                loss.backward() 
            
            optimizer.step()
        if epoch>200:
            if (epoch % cfg.checkpoint_interval == 0) and (cfg.multigpu):
                torch.save(model.module.state_dict(), "%s/%d_weights.pt" % (fold_checkpoint_dir, epoch))
            if (epoch % cfg.checkpoint_interval == 0) and (not cfg.multigpu):
                torch.save(model.state_dict(), "%s/%d_weights.pt" % (fold_checkpoint_dir, epoch))
    del model

