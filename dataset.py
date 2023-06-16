import torch.nn as nn
import torch
import os
import nibabel as nib
from PIL import Image
from segment_anything.utils.transforms import ResizeLongestSide
import cv2
import numpy as np

def slice_img(data_path,save_path):
    for i in range(1,11):
        img_name = data_path + "/img/img" + str(i+10000)[1:5] + ".nii.gz"
        label_name = data_path + "/label/label" + str(i+10000)[1:5] + ".nii.gz"
        img = nib.load(img_name).get_fdata().astype(np.uint8)
        label = nib.load(label_name).get_fdata().astype(np.uint8)
        queue = img.shape[2]
        for idx in range(queue):
            sliced_img = img[:,:,idx]
            sliced_label = label[:,:,idx]
            cv2.imwrite(save_path+"/img"+str(i)+"-"+str(idx)+".png",sliced_img)
            # sliced_img = Image.fromarray(sliced_img, 'L')
            # sliced_img.save(save_path+"/img"+str(i)+"-"+str(idx)+".png")
            np.savetxt(save_path+"/label"+str(i)+"-"+str(idx)+".txt",sliced_label)

info = [1,147, 139, 198, 140, 117, 131, 163, 148, 149, 148,1,1,1,1,1,1,1,1,1,1, 143, 89, 96, 124, 85, 131, 88, 89, 100, 153, 93, 144, 104, 98, 94, 184, 99, 100, 90, 195]

class BTCVdataset(torch.utils.data.Dataset):
    def __init__(self,path,sam_model,device):
        self.img = []
        self.label = []
        self.sum = 0
        for i in range(1,41):
            #运行时改到41
            if i>10 and i<21:
                continue
            for j in range(info[i]):
                img_name = path + "/img"+str(i)+"-"+str(j)+".png"
                label_name = path + "/label"+str(i)+"-"+str(j)+".txt"
                
                
                img_label = np.loadtxt(label_name)
                
                if np.sum(img_label) == 0:
                    self.sum+=1
                    continue
                self.img.append(img_name)
                self.label.append(img_label)

    def __getitem__(self,idx):
        return (self.img[idx],self.label[idx])

    def __len__(self):
        return len(self.img)

if __name__ =='__main__':
    # data_path = "../RawData/Training"
    # save_path = "./data"
    # slice_img(data_path,save_path)
    print(sum(info))