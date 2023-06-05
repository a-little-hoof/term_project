import matplotlib
import matplotlib.pyplot as plt
import random

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything import sam_model_registry, SamPredictor

from utils import cal_dice,show_anns,show_box,show_mask,show_points

import nibabel as nib
from PIL import Image
import cv2
import numpy as np

###读文件
example_img = './img0001.nii.gz'
example_label = './label0001.nii.gz'
img = nib.load(example_img).get_fdata().astype(np.uint8)
label = nib.load(example_label).get_fdata().astype(np.uint8)

queue = img.shape[2]
idx = 100
img = img[:,:,idx]
label = label[:,:,idx]
np.savetxt("test.txt",label)
#print(label.shape)
img = Image.fromarray(img, 'L')
#label = Image.fromarray(label*255)

img.save("./result/test.png")
#label.save("./result/label_test.png")
image = cv2.imread("./result/test.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

model_type = "vit_h"
sam_checkpoint = "./sam_vit_h_4b8939.pth"

###mask_generator
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device="cuda")
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)
print(len(masks))
plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.savefig("./result/sam_result.png")
mdice = cal_dice(masks,label)
print(mdice)

###
###
###predictor
print("Predic:")
predictor = SamPredictor(sam)
predictor.set_image(image)
#single point
dices = []
for label_ind in range(1,14):
    if np.sum((label==label_ind)) == 0:
            continue
    y_list,x_list = np.where(label==label_ind)
    x = random.choice(list(x_list))
    y = random.choice(list(y_list))
    input_point = np.array([[x , y]])
    input_label = np.array([label_ind])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    max_dice = 0
    gt = np.where(label==label_ind,1,0)
    for mask in masks:
        intersection = np.sum(gt*mask)
        union = gt.sum() + mask.sum() 
        dice = 2*intersection/union
        max_dice = max(max_dice,dice)
    dices.append(max_dice)
mdice = np.mean(dices)
print(dices)
print(mdice)

#multipoint
print("MultiPoint_Predic:")
dices = []
for label_ind in range(1,14):
    input_point_=[]
    input_label_=[]
    if np.sum((label==label_ind)) == 0:
            continue
    y_list,x_list = np.where(label==label_ind)
    sample_num = 15
    # randomly select "sample_num" points
    for j in range(sample_num) :
        x = random.choice(list(x_list))
        y = random.choice(list(y_list))
        input_point_.append([x,y])
        input_label_.append(1)
    # convert list to np
    input_point = np.array(input_point_, dtype=float)
    input_label = np.array(input_label_, dtype=float)
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    max_dice = 0
    gt = np.where(label==label_ind,1,0)
    for mask in masks:
        intersection = np.sum(gt*mask)
        union = gt.sum() + mask.sum() 
        dice = 2*intersection/union
        max_dice = max(max_dice,dice)
    dices.append(max_dice)
mdice = np.mean(dices)
print(dices)
print(mdice)
    