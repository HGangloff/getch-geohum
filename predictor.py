import os
import matplotlib.pyplot as plt
from skimage.io import imread
import numpy as np
from glob import glob
import copy
import pandas as pd
import os
# import pathlib
import matplotlib.pyplot as plt
# import imageio
import cv2
import skimage
from scipy import ndimage
import torch
from sklearn.metrics import f1_score, jaccard_score,  mean_squared_error, mean_absolute_error
from scipy.stats import linregress
import math


def computeMask(file, binary='max', plot=True):
    arr = imread(file)
    if binary=='max':
        gray = np.max(arr, axis=-1).astype(np.uint8) #   
    elif binary=='normal':
        gray = cv2.cvtColor(arr,cv2.COLOR_BGR2GRAY)
        
    kernel = np.ones((3,3),np.uint8)
    im_blur=cv2.GaussianBlur(gray,(5,5),0)
    ret,th = cv2.threshold(im_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    opening = cv2.morphologyEx(th,cv2.MORPH_OPEN,kernel, iterations = 2)              # clean noise
    sure_bg = cv2.dilate(opening,kernel,iterations=3)                                 # correct eroded ones
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)                     # # Finding sure foreground area
    ret, sure_fg = cv2.threshold(dist_transform,0.000012*dist_transform.max(),255,0)  # ver subjective number
    sure_fg = np.uint8(sure_fg)                                                       # the mask we need
    if not plot:
        return  np.where(sure_bg==255, 1, sure_bg)
    else:
        unknown = cv2.subtract(sure_bg,sure_fg) 
        ret, markers = cv2.connectedComponents(sure_fg)  
        markers = markers+1  
        markers[unknown==255] = 0
        arr[markers == -1] = [255,0,0]
        return np.where(sure_bg==255, 1, sure_bg), markers, arr   # mask, marker, arra
    
    
def countObjects(mask, remove_small=True, min_size=4):
    ret, markers = cv2.connectedComponents(mask)
    if remove_small:
        count=0
        for i in range(ret):
            obj_mask = np.where(markers==i,1,0)
            size = np.sum(obj_mask)
            if size>=min_size:
                count+=1
        return count-1
    else:
        return ret-1 # subtracting 1 is mainly to remove bakground object

    
def computeMetricsAll(refs, preds, binary='max', save=True, save_name=None, save_mask=True, im_path=None, rep_path=None):
    assert len(preds) == len(refs)
    IOU = []
    P = []
    R = []
    F1 = []
    for i in range(len(preds)):
        r_mask = imread(refs[i])
        p_mask, mkr, arr_mkr = computeMask(preds[i], binary=binary)   # 

        r_obj = countObjects(r_mask)  # count reference objects 
        p_obj = countObjects(p_mask)  # count predicted objects

        iou = jaccard_score(y_true=r_mask.ravel(), y_pred=p_mask.ravel(), pos_label=1, average='weighted')
        f1 = f1_score(y_true=r_mask.ravel(), y_pred=p_mask.ravel(), average='weighted')

        IOU.append(iou)
        P.append(p_obj)
        R.append(r_obj)
        F1.append(f1)
        
        print(f'{save_name}: iou --> {iou}  f1-->{f1}', end='\r', flush=True)
        if save_mask:
            name = os.path.split(preds[i])[1]
            np.save(f'{im_path}/mask_{name}', p_mask)
            np.save(f'{im_path}/marker_{name}', mkr)
            np.save(f'{im_path}/bound_{name}', arr_mkr)
    
    
    metric = {'iou':IOU, 'ref_n':R, 'pred_n':P, 'f1':F1}  # chips level metrics will be saved as dataframe
    
    # scene label metrics
    
    det = linregress(np.array(P), np.array(R))
    mse = mean_squared_error(R,P)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(R,P)
    
    mets = {'R':det.rvalue, 'mse':mse, 'rmse':rmse, 'mae':mae}
    mets['tot_ref'] = np.nansum(R)
    mets['tot_pred'] = np.nansum(P)
    mets['tot_abs_dev'] = abs(np.nansum(R)-np.nansum(P))
    
    if save:
        df = pd.DataFrame.from_dict(metric)
        df.to_csv(f'{rep_path}/{binary}_{save_name}.csv')
    return mets

def parseArgs():
    parser = argparse.ArgumentParser(description='A process to create mask from anomaly scores and compute related metrics')
    parser.add_argument('--data_fold', help='root folder that contained all predicted data folders', type=str)
    parser.add_argument('--mask_fold', help='path to save predicted binary masks', type=str)
    parser.add_argument('--rep_fold', help='path to save metrics and reportage', type=str)
    parser.add_argument('--binary', help='mechanism to create greyscale from jet RGB, could be either "normal" or "max"" ', type=str, default='normal') 
    args= parser.parse_args()
    return args

if __name__ == "__main__":
    args = parseArgs()
    
    if not os.path.exists(args.rep_fold):
        os.makedirs(rep_fold, exist_ok=True)
    
    met_dict = {}
    for fold in os.listdir(args.data_fold):
        print(f'Processing for {fold} ...')
        data_fold = f'{args.data_fold}/{fold}'
        image_fold = f'{args.mask_fold}/{fold}'
        if not os.path.exists(image_fold):
            os.makedirs(image_fold, exist_ok=True)

        pngs = glob(f'{data_fold}/*.png')
        ori = sorted([a for a in pngs if 'ori' in a])
        gts = sorted([a for a in pngs if 'gt' in a])
        scor = sorted([a for a in pngs if 'amap.png' in a])
        assert len(ori) == len(gts) == len(scor)
        im_mtertic = computeMetricsAll(refs=gts, preds=scor, binary=binary, save=True, save_name=fold, save_mask=True, im_path=image_fold, rep_path=rep_fold)
        met_dict[fold] = im_mtertic
    df = pd.DataFrame.from_dict(met_dict)
    df.to_csv(f'{rep_fold}/all_metric_{binary}.csv')