import argparse
import sys
import os

import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import tqdm
import pprint
from src import utils as ut
import torchvision
from haven import haven_utils as hu
from haven import haven_chk as hc


from src import datasets, models
from torch.utils.data import DataLoader
import exp_configs
from torch.utils.data.sampler import RandomSampler
from src import wrappers







def trainval(exp_dict, savedir_base, reset, metrics_flag=True, datadir=None, cuda=False):
    # bookkeeping
    # ---------------

    # get experiment directory
    #exp_id = hu.hash_dict(exp_dict)
    #savedir = os.path.join(savedir_base, exp_id)
    savedir = os.path.join(savedir_base, "checkpoints")

    if reset:
        # delete and backup experiment
        hc.delete_experiment(savedir, backup_flag=True)
    
    # create folder and save the experiment dictionary
    os.makedirs(savedir, exist_ok=True)
    hu.save_json(os.path.join(savedir, 'exp_dict.json'), exp_dict)
    print(pprint.pprint(exp_dict))
    print('Experiment saved in %s' % savedir)


    # set seed
    # ==================
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        device = 'cuda'
        torch.cuda.manual_seed_all(seed)
        assert torch.cuda.is_available(), 'cuda is not, available please run with "-c 0"'
    else:
        device = 'cpu'

    print('Running on device: %s' % device)
    
    # Dataset
    # Load val set and train set
    val_set = datasets.get_dataset(dataset_name=exp_dict["dataset"], split="val",
                                   transform=exp_dict.get("transform"),
                                   datadir=datadir)

    train_set = datasets.get_dataset(dataset_name=exp_dict["dataset"],
                                     split="train", 
                                     transform=exp_dict.get("transform"),
                                     datadir=datadir)
    
    # Load train loader, val loader, and vis loader
    train_loader = DataLoader(train_set, 
                            sampler=RandomSampler(train_set,
                            replacement=True, num_samples=max(min(500, 
                                                            len(train_set)), 
                                                            len(val_set))),
                            batch_size=exp_dict["batch_size"],
                            num_workers = 32,
                            
                            )

    val_loader = DataLoader(val_set, shuffle=False, batch_size=exp_dict["batch_size"], 
                            num_workers = 32,
                            )
    vis_loader = DataLoader(val_set, sampler=ut.SubsetSampler(train_set,
                                                     indices=[0, 1, 2]),
                            batch_size=1, num_workers = 32,)


    # Create model, opt, wrapper
    model_original = models.get_model(exp_dict["model"], exp_dict=exp_dict).cuda()

    lr_rate = 1e-4
    opt = torch.optim.Adam(model_original.parameters(), 
                        lr=lr_rate , weight_decay=0.0005)

    lr_sched = CosineAnnealingLR(opt, len(train_loader), eta_min=lr_rate )  # https://discuss.pytorch.org/t/how-to-implement-torch-optim-lr-scheduler-cosineannealinglr/28797/3                      


    model = wrappers.get_wrapper(exp_dict["wrapper"], model=model_original, opt=opt, lr_sched=lr_sched ).cuda()



    # Checkpointing
    # =============
    score_list_path = os.path.join(savedir, "score_list.pkl")
    model_path = os.path.join(savedir, "model_state_dict.pth")
    opt_path = os.path.join(savedir, "opt_state_dict.pth")

    if os.path.exists(score_list_path):
        # resume experiment
        score_list = ut.load_pkl(score_list_path)
        model.load_state_dict(torch.load(model_path))
        
        s_epoch = score_list[-1]["epoch"] + 1
    else:
        # restart experiment
        score_list = []
        s_epoch = 0



    model.vis_on_loader(vis_loader, savedir=os.path.join(savedir, "images"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', nargs='+')
    parser.add_argument('-sb', '--savedir_base', required=True)
    parser.add_argument('-d', '--datadir', required=True)
    parser.add_argument('-r', '--reset',  default=0, type=int)
    parser.add_argument('-ei', '--exp_id', default=None)
    parser.add_argument('-c', '--cuda', type=int, default=1)

    args = parser.parse_args()


    # Collect experiments
    # -------------------
    if args.exp_id is not None:
        # select one experiment
        savedir = os.path.join(args.savedir_base, args.exp_id)
        exp_dict = hu.load_json(os.path.join(savedir, 'exp_dict.json'))        
        
        exp_list = [exp_dict]
        
    else:
        # select exp group
        exp_list = []
        for exp_group_name in args.exp_group_list:
            exp_list += exp_configs.EXP_GROUPS[exp_group_name]

    ####
    # Run experiments or View them
    # ----------------------------
    
    # run experiments
    for exp_dict in exp_list:
        # do trainval
        trainval(exp_dict=exp_dict,
                savedir_base=args.savedir_base,
                reset=args.reset,
                datadir=args.datadir,
                cuda=args.cuda)
    