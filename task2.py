import torch
import torch.nn as nn
from dataset import BTCVdataset
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything import sam_model_registry, SamPredictor
from torch.utils.data import DataLoader
from segment_anything.utils.transforms import ResizeLongestSide
import torch.nn.functional as F
from utils import generate_box,show_box,show_mask,DiceLoss
import cv2
import monai
from tqdm import tqdm
from numpy import *
import matplotlib.pyplot as plt
from torch.utils.tensorboard  import SummaryWriter

data_path = "./data"
train_batch_size = 4
test_batch_size = 4
model_type = 'vit_h'
checkpoint = 'sam_vit_h_4b8939.pth'
device = "cuda:0"
lr = 1e-6
wd = 0
num_epochs = 100
input_size = (1024,1024)
original_image_size = (512,512)


def main():
    sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
    sam_model.to(device)

    dataset = BTCVdataset(data_path,sam_model,device)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size,shuffle=False)
       
    
    sam_model.train()

    optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(),lr=lr, weight_decay=wd) 
    seg_loss = DiceLoss()
    
    losses = []
    writer = SummaryWriter(log_dir="./vis/")
    best_per = 1
    train_num = 0
    eva_num = 0
    for epoch in range(num_epochs):
        epoch_losses = []
        for i,(image,label) in enumerate(tqdm(train_loader)):
            
            image_list = []
            for img_name in image:   
                image1 = cv2.imread(img_name)
                image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
                transform = ResizeLongestSide(sam_model.image_encoder.img_size)
                input_image = transform.apply_image(image1)
                input_image_torch = torch.as_tensor(input_image, device=device)
                transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
                input_image = sam_model.preprocess(transformed_image)
                image_list.append(input_image)
            input_image = torch.cat(tuple(image_list),0)
            #print(input_image.shape)
            
            boxes,ind_list = generate_box(label)
            transform = ResizeLongestSide(sam_model.image_encoder.img_size)
            boxes_tensor = []
            for box in boxes:
                box = transform.apply_boxes(box, original_image_size)
                boxes_tensor.append(box[0])
            boxes_torch = torch.tensor(boxes_tensor, dtype=torch.float, device=device)
            

            with torch.no_grad():
                image_embedding = sam_model.image_encoder(input_image)
                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=None,
                    boxes=boxes_torch,
                    masks=None,
                )
            low_res_masks, iou_predictions = sam_model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            upscaled_masks = sam_model.postprocess_masks(low_res_masks, input_size, original_image_size).to(device)
            binary_mask = F.normalize(F.threshold(upscaled_masks, 0.0, 0))
            
            gt_masks = []
            for num,gt_mask in enumerate(label):
                gt_mask = torch.where(gt_mask==ind_list[num],1,0)
                # print(gt_mask.shape)
                gt_masks.append(gt_mask.tolist())
            
            gt_masks = torch.tensor(gt_masks, dtype=torch.float32,device=device)

            # mask = binary_mask[1]
            # img_name = image[1]
            # image1 = cv2.imread(img_name)
            # image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            # box = boxes[1]
            # plt.figure(figsize=(10, 10))
            # plt.imshow(image1)
            # show_mask(gt_masks[1].cpu().detach().numpy(), plt.gca(), random_color=True)
            # show_mask(mask.cpu().detach().numpy(), plt.gca(), random_color=True)
            # show_box(box, plt.gca())
            # print(box)
            # plt.axis('off')
            # plt.savefig("1.png")
            
            #print(gt_masks.shape)
            #print(binary_mask.shape)
            loss = seg_loss(binary_mask.squeeze(), gt_masks)
            #print(loss)
            # exit(0)
            if i % 10==0:
                print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            writer.add_scalar('train',loss.item(),train_num)
            train_num+=1
        losses.append(epoch_losses)
        print(f'EPOCH: {epoch}')
        print(f'Mean loss: {mean(epoch_losses)}')

        with torch.no_grad():
            for i,(eva_image,eva_label) in enumerate(tqdm(test_loader)):
                eva_losses = []
                eva_image_list = []
                for eva_img_name in eva_image:   
                    eva_image1 = cv2.imread(eva_img_name)
                    eva_image1 = cv2.cvtColor(eva_image1, cv2.COLOR_BGR2RGB)
                    transform = ResizeLongestSide(sam_model.image_encoder.img_size)
                    eva_input_image = transform.apply_image(eva_image1)
                    eva_input_image_torch = torch.as_tensor(eva_input_image, device=device)
                    eva_transformed_image = eva_input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
                    eva_input_image = sam_model.preprocess(eva_transformed_image)
                    eva_image_list.append(eva_input_image)
                eva_input_image = torch.cat(tuple(eva_image_list),0)
                #print(input_image.shape)
                
                eva_boxes,eva_ind_list = generate_box(eva_label)
                transform = ResizeLongestSide(sam_model.image_encoder.img_size)
                eva_boxes_tensor = []
                for eva_box in eva_boxes:
                    eva_box = transform.apply_boxes(eva_box, original_image_size)
                    eva_boxes_tensor.append(eva_box[0])
                eva_boxes_torch = torch.tensor(eva_boxes_tensor, dtype=torch.float, device=device)
                

                with torch.no_grad():
                    eva_image_embedding = sam_model.image_encoder(eva_input_image)
                    eva_sparse_embeddings, eva_dense_embeddings = sam_model.prompt_encoder(
                        points=None,
                        boxes=eva_boxes_torch,
                        masks=None,
                    )
                eva_low_res_masks, eva_iou_predictions = sam_model.mask_decoder(
                    image_embeddings=eva_image_embedding,
                    image_pe=sam_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=eva_sparse_embeddings,
                    dense_prompt_embeddings=eva_dense_embeddings,
                    multimask_output=False,
                )

                eva_upscaled_masks = sam_model.postprocess_masks(eva_low_res_masks, input_size, original_image_size).to(device)
                eva_binary_mask = F.normalize(F.threshold(eva_upscaled_masks, 0.0, 0))
                
                eva_gt_masks = []
                for num,eva_gt_mask in enumerate(eva_label):
                    eva_gt_mask = torch.where(eva_gt_mask==eva_ind_list[num],1,0)
                    # print(gt_mask.shape)
                    eva_gt_masks.append(eva_gt_mask.tolist())
                
                eva_gt_masks = torch.tensor(eva_gt_masks, dtype=torch.float32,device=device)
                eva_loss = seg_loss(eva_binary_mask.squeeze(), eva_gt_masks)
                eva_losses.append(eva_loss.item())
                writer.add_scalar('test',eva_loss.item(),eva_num)
                eva_num+=1
        print(mean(eva_losses))
        if mean(eva_losses)<best_per:
            best_per = mean(eva_losses)
            torch.save(sam_model.state_dict(), f"best_weights.pth")
            print(f"Model Was Saved! Current Best val loss {best_per}")


        


if __name__=="__main__":
    main()