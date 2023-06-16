import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

def cal_dice(masks,label):
    dices = []
    for i in range(1,14):
        if np.sum((label==i)) == 0:
            continue
        dice = 0
        gt = np.where(label==i,1,0)
        ###思路：找到masks中与label第i个器官重合度最大的mask来计算
        max_dice = 0
        for mask in masks:
            seg = mask["segmentation"]
            area = mask["area"]
            intersection = np.sum(gt*seg)
            union = gt.sum()+area
            dice = 2*intersection/union
            max_dice = max(max_dice,dice)
        dices.append(max_dice)
    mdice = np.mean(dices)
    print(dices)
    return mdice

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

def generate_box(label):
    box = []
    ind_list = []
    for i in range(label.shape[0]):
        mask = label[i]
        while True:
            ind = random.randint(1,13)
            if torch.sum(mask==ind)!=0:
                ind_list.append(ind)
                break
        y_list,x_list = torch.where(mask==ind)
        xmax,xmin = torch.max(x_list),torch.min(x_list)
        ymax,ymin = torch.max(y_list),torch.min(y_list)
        box.append(np.array([xmin,ymin,xmax,ymax]))
    return box,ind_list

class DiceLoss(nn.Module):
    def __init__(self,weight=None,size_average=True):
        super(DiceLoss,self).__init__()
        
    def forward(self,inputs,targets,smooth=1e-8):
        # inputs = F.sigmoid(inputs)       
        # inputs = inputs.view(-1)
        # targets = targets.view(-1)
        intersection = torch.sum((inputs * targets),(1,2))
        #print(torch.sum((inputs * targets),(1,2)).shape)               
        dice = (2.*intersection + smooth)/(torch.sum(inputs,(1,2)) + torch.sum(targets,(1,2)) + smooth)  
        # print(intersection)
        # print(torch.sum(inputs,(1,2)))
        # print(torch.sum(targets,(1,2)))
        # print(dice)
        return 1 - torch.mean(dice)

