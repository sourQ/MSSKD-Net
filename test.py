import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


from IMD_dataloader import eval_dataset_loader_init
from utils.utils import save_image

from models.seg_hrnet import get_seg_model
from models.seg_hrnet_config import get_hrnet_cfg
from utils.config import get_pscc_args
from models.NLCDetection import NLCDetection
from models.detection_head import DetectionHead
from utils.load_vdata import TestData
from utils.utils1 import viz_log
import os

device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device('cuda:0')


def load_network_weight(net, checkpoint_dir, name):
    weight_path = '{}/{}.pth'.format(checkpoint_dir, name)
    net_state_dict = torch.load(weight_path, map_location='cuda:0')
    net.load_state_dict(net_state_dict)
    print('{} weight-loading succeeds'.format(name))


def test(args):
    # define backbone
    FENet_name = 'HRNet'
    FENet_cfg = get_hrnet_cfg()
    FENet = get_seg_model(FENet_cfg)

    # define localization head
    SegNet_name = 'NLCDetection'
    SegNet = NLCDetection(args)

    # define detection head
    ClsNet_name = 'DetectionHead'
    ClsNet = DetectionHead(args)

    FENet_checkpoint_dir = './checkpoint/{}_checkpoint'.format(FENet_name)
    SegNet_checkpoint_dir = './checkpoint/{}_checkpoint'.format(SegNet_name)
    ClsNet_checkpoint_dir = './checkpoint/{}_checkpoint'.format(ClsNet_name)

    # load FENet weight
    FENet = FENet.to(device)
    FENet = nn.DataParallel(FENet, device_ids=device_ids)
    load_network_weight(FENet, FENet_checkpoint_dir, FENet_name)

    # load SegNet weight
    SegNet = SegNet.to(device)
    SegNet = nn.DataParallel(SegNet, device_ids=device_ids)
    load_network_weight(SegNet, SegNet_checkpoint_dir, SegNet_name)

    # load ClsNet weight
    ClsNet = ClsNet.to(device)
    ClsNet = nn.DataParallel(ClsNet, device_ids=device_ids)
    load_network_weight(ClsNet, ClsNet_checkpoint_dir, ClsNet_name)

    test_data_loader = DataLoader(TestData(args), batch_size=1, shuffle=False,
                                  num_workers=8)

    for batch_id, test_data in enumerate(test_data_loader):

        image, cls, name = test_data
        image = image.to(device)

        with torch.no_grad():

            # backbone network
            FENet.eval()
            feat = FENet(image)

            # localization head
            SegNet.eval()
            pred_mask = SegNet(feat)[0]

            pred_mask = F.interpolate(pred_mask, size=(image.size(2), image.size(3)), mode='bilinear',
                                      align_corners=True)

            # classification head
            ClsNet.eval()
            pred_logit = ClsNet(feat)

        # ce
        sm = nn.Softmax(dim=1)
        pred_logit = sm(pred_logit)
        _, binary_cls = torch.max(pred_logit, 1)

        pred_tag = 'forged' if binary_cls.item() == 1 else 'authentic'

        if args.save_tag:
            save_image(pred_mask, name, 'mask_curve')

        print_name = name[0].split('/')[-1].split('.')[0]

        print(f'The image {print_name} is {pred_tag}')

def Inference_loc(
                args,LOSS_MAP=None,
                iter_num=None, 
                save_tag=False, 
                localization=True
                ):
    '''
        the inference pipeline for the pre-trained model.
        the image-level detection will dump to the csv file.
        the pixel-level localization will be saved as in the npy file.
    '''
        # define backbone
    FENet_name = 'HRNet'
    FENet_cfg = get_hrnet_cfg()
    FENet = get_seg_model(FENet_cfg)
    # define localization head
    SegNet_name = 'NLCDetection'
    SegNet = NLCDetection(args)
    # define detection head
    ClsNet_name = 'DetectionHead'
    ClsNet = DetectionHead(args)
    FENet_checkpoint_dir = './checkpoint/{}_checkpoint'.format(FENet_name)
    SegNet_checkpoint_dir = './checkpoint/{}_checkpoint'.format(SegNet_name)
    ClsNet_checkpoint_dir = './checkpoint/{}_checkpoint'.format(ClsNet_name)
    # load FENet weight
    FENet = FENet.to(device)
    FENet = nn.DataParallel(FENet, device_ids=device_ids)
    load_network_weight(FENet, FENet_checkpoint_dir, FENet_name)
    # load SegNet weight
    SegNet = SegNet.to(device)
    SegNet = nn.DataParallel(SegNet, device_ids=device_ids)
    load_network_weight(SegNet, SegNet_checkpoint_dir, SegNet_name)
    # load ClsNet weight
    ClsNet = ClsNet.to(device)
    ClsNet = nn.DataParallel(ClsNet, device_ids=device_ids)
    load_network_weight(ClsNet, ClsNet_checkpoint_dir, ClsNet_name)

    seg_correct, seg_total, cls_correct, cls_total = [0] * 4
    #val_list = [1]#[0,1,2,3]
    val_list = [2]
    for val_tag in val_list:

        val_data_loader, data_label = eval_dataset_loader_init(args, val_tag)
        print(f"working on the dataset: {data_label}.")
        F1_lst, auc_lst = [], []
   
        pred_soft_ncls = []
        ncls = []
        with torch.no_grad():
            FENet.eval()
            SegNet.eval()
            ClsNet.eval()

            class_gt = []
            class_pre = []

            for step, val_data in enumerate(tqdm(val_data_loader)):
                image, mask, cls, image_names = val_data
                image, mask = image.to(device), mask.to(device)
                mask = torch.squeeze(mask, axis=1)
                cls[cls != 0] = 1
                cls = cls.to(device)

                # model
                try:
                    feat = FENet(image)
                    pred_mask = SegNet(feat)[0]
                    pred_logit = ClsNet(feat)
                except:
                    print(f"does not work on the ", image_names)
                    continue
                if pred_mask.shape != mask.shape:
                    pred_mask = F.interpolate(pred_mask, size=(mask.size(1), mask.size(2)), mode='bilinear', align_corners=True)

                binary_mask1 = torch.zeros_like(pred_mask)
                binary_mask1[pred_mask > 0.5] = 1
                binary_mask1[pred_mask <= 0.5] = 0
                

                seg_correct += (binary_mask1 == mask).sum().item()
                seg_total += int(torch.ones_like(mask).sum().item())

                # ce
                sm = nn.Softmax(dim=1)
                pred_logit = sm(pred_logit)
                pred_soft_ncls.extend(pred_logit[:, 1])
                ncls.extend(cls)
                _, binary_cls = torch.max(pred_logit, 1)
                cls_correct += (binary_cls == cls).sum().item()
                cls_total += int(torch.ones_like(cls).sum().item())
                binary_cls = binary_cls.cpu()
                cls = cls.cpu()

                class_gt.append(np.array(cls)[0])
                class_pre.append(np.array(binary_cls)[0])
    
                  
                
                if args.loss_type == 'dm':
                    loss_map, loss_manip, loss_nat = LOSS_MAP(mask1_fea, mask)
                    pred_mask = LOSS_MAP.dis_curBatch.squeeze(dim=1)
                    pred_mask_score = LOSS_MAP.dist.squeeze(dim=1)
                elif args.loss_type == 'ce':
                    pred_mask_score = binary_mask1
                    pred_mask = torch.zeros_like(binary_mask1)
                    pred_mask[binary_mask1 > 0.5] = 1
                    pred_mask[binary_mask1 <= 0.5] = 0
                image_name = os.path.basename(image_names[0])
                image_name = image_name.split('.')[0]
                viz_log(args, mask, pred_mask, image, iter_num, f"{step}_{image_name}", mode='eval/'+data_label)

                mask = torch.unsqueeze(mask, axis=1)
            
                if cls !=0:
                    for img_idx, cur_img_name in enumerate(image_names):

                        mask_ = torch.unsqueeze(mask[img_idx,0], 0)
                        pred_mask_ = torch.unsqueeze(pred_mask[img_idx], 0)
                        pred_mask_score_ = torch.unsqueeze(pred_mask_score[img_idx], 0)

                        mask_ = mask_.cpu().clone().cpu().numpy().reshape(-1)
                        pred_mask_ = pred_mask_.cpu().clone().cpu().numpy().reshape(-1)
                        pred_mask_score_ = pred_mask_score_.cpu().clone().cpu().numpy().reshape(-1)

                        F1_a  = metrics.f1_score(mask_, pred_mask_, average='macro')
                        auc_a = metrics.roc_auc_score(mask_, pred_mask_score_)

                        pred_mask_[np.where(pred_mask_ == 0)] = 1
                        pred_mask_[np.where(pred_mask_ == 1)] = 0

                        F1_b  = metrics.f1_score(mask_, pred_mask_, average='macro')
                        if F1_a > F1_b:
                            F1 = F1_a
                        else:
                            F1 = F1_b
                        F1_lst.append(F1)
                        AUC_score = auc_a if auc_a > 0.5 else 1-auc_a
                        auc_lst.append(AUC_score)


        print(np.array(class_gt))
        print(np.array(class_pre))
        F1_cls = metrics.f1_score(class_gt, class_pre, average='binary')
        auc_cls = metrics.roc_auc_score(np.array(class_gt), np.array(class_pre))

        
        print("F1_cls",np.mean(F1_cls)) 
        print("auc_cls",np.mean(auc_cls)) 
        print("acc",cls_correct / cls_total)           
        #print("segacc: ", seg_correct / seg_total)
        print("F1: ", np.mean(F1_lst))
        print("AUC: ", np.mean(auc_lst))
        
if __name__ == '__main__':
    args = get_pscc_args()
    Inference_loc(args)
    # test(args)
