import argparse
import sys
import os
import os.path as osp 

import torch
import cv2
import numpy as np
import pandas as pd
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import tqdm
import pprint
from src import utils as ut
import torchvision
from haven import haven_utils as hu
from haven import haven_chk as hc
from haven import haven_utils as hu
from skimage import morphology as morph
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.segmentation import find_boundaries
from skimage.segmentation import mark_boundaries
from skimage import data, io, segmentation, color
from skimage.measure import label


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
    savedir = os.path.join(savedir_base, "sample_1")

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


    val_loader = DataLoader(val_set, shuffle=False, batch_size=exp_dict["batch_size"], 
                            num_workers = 32,
                            )

    vis_loader = DataLoader(val_set, sampler=ut.SubsetSampler(train_set,
                                                     indices=[0, 1, 2]),
                            batch_size=1, num_workers = 32,)


    # Create model, opt, wrapper
    model = models.get_model(exp_dict["model"], exp_dict=exp_dict).cuda()


    model_path = os.path.join(savedir, "loc_80.pth")


    name_list = list(model.state_dict().keys())


    if os.path.exists(model_path):
        print("Model_loaded...")
        param_dict = torch.load(model_path)
        
        for idx, i in enumerate(param_dict):

            if i in model.state_dict().keys():
                print(i)
                model.state_dict()[i].copy_(param_dict[i])


    else: 
        print("No pretrained model")
        exit()
        

#    for n_iter, batch in enumerate(tqdm.tqdm(val_loader)):
#        with torch.no_grad():
#            images = batch["images"].cuda()
#            counts = batch["counts"].cuda()
#            
#            pred_counts = predict_on_batch(model, batch, method="counts") 
#            pred_blobs = predict_on_batch(model, batch, method="blobs")
#    

    for n_iter, batch in enumerate(tqdm.tqdm(val_loader)):
        with torch.no_grad():
            pred_counts = predict_on_batch(model, batch, method="counts") 
            pred_blobs = predict_on_batch(model, batch, method="blobs")


            pred_count = pred_counts.ravel()[0]
            pred_blobs = pred_blobs.squeeze()

            img = hu.get_image(batch["images"],denorm="rgb")  # Tensor to numpy 
            
#            points = ndimage.zoom(batch["points"],11).squeeze()
            img_np = hu.f2l(img).squeeze().clip(0,1)  # Tensor to numpy 



#            img = np.squeeze(img)
#            img = img.transpose(1,2,0)
#            cv2.imshow("test", img_np)
#            cv2.waitKey(0)

            out = color.label2rgb(label(pred_blobs), image=(img_np), image_alpha=1.0, bg_label=0)
            img_mask = mark_boundaries(out.squeeze(),  label(pred_blobs).squeeze())

            out = color.label2rgb(label(points), image=(img_np), image_alpha=1.0, bg_label=0)
            img_points = mark_boundaries(out.squeeze(),  label(points).squeeze())

#            cv2.imshow("test", out)
#            cv2.waitKey(0)

#            img = hu.get_image(0.7*batch["images"] + 0.3*torch.FloatTensor(pred_blobs).squeeze(), denorm="rgb")
#            img_mask = 0.7*img[0] + 0.3*hu.l2f(hu.gray2cmap(pred_blobs)[0])

#            hu.save_image("/home/kist-ubuntu/workspace_playground/DeepFish/outputs/sample_1/pred_80/%d.jpg" % batch["meta"]["index"], img_mask)
            hu.save_image("/home/kist-ubuntu/workspace_playground/DeepFish/outputs/sample_1/pred_point/%d.jpg" % batch["meta"]["index"], img_points)





def predict_on_batch(model, batch, **options):
    model.eval()
    # feat_8s, feat_16s, feat_32s = self.model.extract_features(batch["images"].cuda())
    if options["method"] == "counts":
        images = batch["images"].cuda()
        pred_mask = model.forward(images).data.max(1)[1].squeeze().cpu().numpy()
        counts = np.zeros(model.n_classes - 1)
        for category_id in np.unique(pred_mask):
            if category_id == 0:
                continue
            blobs_category = morph.label(pred_mask == category_id)
            n_blobs = (np.unique(blobs_category) != 0).sum()
            counts[category_id - 1] = n_blobs
        return counts[None]
    elif options["method"] == "blobs":
        images = batch["images"].cuda()
        pred_mask = model.forward(images).data.max(1)[1].squeeze().cpu().numpy()
        h, w = pred_mask.shape
        blobs = np.zeros((model.n_classes - 1, h, w), int)
        for category_id in np.unique(pred_mask):
            if category_id == 0:
                continue
            blobs[category_id - 1] = morph.label(pred_mask == category_id)
        return blobs[None]
    elif options["method"] == "points":
        images = batch["images"].cuda()
        pred_mask = model.forward(images).data.max(1)[1].squeeze().cpu().numpy()
        h, w = pred_mask.shape
        blobs = np.zeros((model.n_classes - 1, h, w), int)
        for category_id in np.unique(pred_mask):
            if category_id == 0:
                continue
            blobs[category_id - 1] = morph.label(pred_mask == category_id)
        return blobs[None]

def get_blob_dict_base(model, blobs, points, training=False):
    if blobs.ndim == 2:
        blobs = blobs[None]

    blobList = []

    n_multi = 0
    n_single = 0
    n_fp = 0
    total_size = 0

    for l in range(blobs.shape[0]):
        class_blobs = blobs[l]
        points_mask = points == (l + 1)
        # Intersecting
        blob_uniques, blob_counts = np.unique(class_blobs * (points_mask), return_counts=True)
        uniques = np.delete(np.unique(class_blobs), blob_uniques)

        for u in uniques:
            blobList += [{"class": l, "label": u, "n_points": 0, "size": 0,
                          "pointsList": []}]
            n_fp += 1

        for i, u in enumerate(blob_uniques):
            if u == 0:
                continue

            pointsList = []
            blob_ind = class_blobs == u

            locs = np.where(blob_ind * (points_mask))

            for j in range(locs[0].shape[0]):
                pointsList += [{"y": locs[0][j], "x": locs[1][j]}]

            assert len(pointsList) == blob_counts[i]

            if blob_counts[i] == 1:
                n_single += 1

            else:
                n_multi += 1
            size = blob_ind.sum()
            total_size += size
            blobList += [{"class": l, "size": size,
                          "label": u, "n_points": blob_counts[i],
                          "pointsList": pointsList}]

    blob_dict = {"blobs": blobs, "blobList": blobList,
                 "n_fp": n_fp,
                 "n_single": n_single,
                 "n_multi": n_multi,
                 "total_size": total_size}

    return blob_dict




def lc_loss_base(logits, images, points, counts, blob_dict):
    N = images.size(0)
    assert N == 1

    S = F.softmax(logits, 1)
    S_log = F.log_softmax(logits, 1)

    # IMAGE LOSS
    loss = compute_image_loss(S, counts)

    # POINT LOSS
    loss += F.nll_loss(S_log, points,
                       ignore_index=0,
                       reduction='sum')
    # FP loss
    if blob_dict["n_fp"] > 0:
        loss += compute_fp_loss(S_log, blob_dict)

    # split_mode loss
    if blob_dict["n_multi"] > 0:
        loss += compute_split_loss(S_log, S, points, blob_dict)

    # Global loss
    S_npy = hu.t2n(S.squeeze())
    points_npy = hu.t2n(points).squeeze()
    for l in range(1, S.shape[1]):
        points_class = (points_npy == l).astype(int)

        if points_class.sum() == 0:
            continue

        T = watersplit(S_npy[l], points_class)
        T = 1 - T
        scale = float(counts.sum())
        loss += float(scale) * F.nll_loss(S_log, torch.LongTensor(T).cuda()[None],
                                          ignore_index=1, reduction='mean')

    return loss / N


# Loss Utils
def compute_image_loss(S, Counts):
    n, k, h, w = S.size()

    # GET TARGET
    ones = torch.ones(Counts.size(0), 1).long().cuda()
    BgFgCounts = torch.cat([ones.float(), Counts.float()], 1)
    Target = (BgFgCounts.view(n * k).view(-1) > 0).view(-1).float()

    # GET INPUT
    Smax = S.view(n, k, h * w).max(2)[0].view(-1)

    loss = F.binary_cross_entropy(Smax, Target, reduction='sum')

    return loss

def compute_split_loss(S_log, S, points, blob_dict):
    blobs = blob_dict["blobs"]
    S_numpy = hu.t2n(S[0])
    points_numpy = hu.t2n(points).squeeze()

    loss = 0.

    for b in blob_dict["blobList"]:
        if b["n_points"] < 2:
            continue

        l = b["class"] + 1
        probs = S_numpy[b["class"] + 1]

        points_class = (points_numpy == l).astype("int")
        blob_ind = blobs[b["class"]] == b["label"]

        T = watersplit(probs, points_class * blob_ind) * blob_ind
        T = 1 - T

        scale = b["n_points"] + 1
        loss += float(scale) * F.nll_loss(S_log, torch.LongTensor(T).cuda()[None],
                                          ignore_index=1, reduction='mean')

    return loss


def watersplit(_probs, _points):
    points = _points.copy()

    points[points != 0] = np.arange(1, points.sum() + 1)
    points = points.astype(float)

    probs = ndimage.black_tophat(_probs.copy(), 7)
    seg = watershed(probs, points)

    return find_boundaries(seg)


def compute_fp_loss(S_log, blob_dict):
    blobs = blob_dict["blobs"]

    scale = 1.
    loss = 0.
    n_terms = 0
    for b in blob_dict["blobList"]:
        if n_terms > 25:
            break

        if b["n_points"] != 0:
            continue

        T = np.ones(blobs.shape[-2:])
        T[blobs[b["class"]] == b["label"]] = 0

        loss += scale * F.nll_loss(S_log, torch.LongTensor(T).cuda()[None],
                                   ignore_index=1, reduction='mean')

        n_terms += 1
    return loss


def compute_game(pred_points, gt_points, L=1):
    n_rows = 2**L
    n_cols = 2**L

    pred_points = pred_points.astype(float)
    gt_points = gt_points.astype(float)
    h, w = pred_points.shape
    se = 0.

    hs, ws = h//n_rows, w//n_cols
    for i in range(n_rows):
        for j in range(n_cols):

            sr, er = hs*i, hs*(i+1)
            sc, ec = ws*j, ws*(j+1)

            pred_count = pred_points[sr:er, sc:ec]
            gt_count = gt_points[sr:er, sc:ec]
            
            se += float(abs(gt_count.sum() - pred_count.sum()))
    return se

def blobs2points(blobs):
    points = np.zeros(blobs.shape).astype("uint8")
    rps = skimage.measure.regionprops(blobs)

    assert points.ndim == 2

    for r in rps:
        y, x = r.centroid

        points[int(y), int(x)] = 1


    # assert points.sum() == (np.unique(blobs) != 0).sum()
       
    return points




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', default="loc" ,  nargs='+')
    parser.add_argument('-sb', '--savedir_base',default="/home/kist-ubuntu/workspace_playground/DeepFish/outputs" )
    parser.add_argument('-d', '--datadir', default="/home/kist-ubuntu/workspace_playground/DeepFish/DeepFish")
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
#        for exp_group_name in args.exp_group_list:
        exp_list += exp_configs.EXP_GROUPS["loc"]

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
