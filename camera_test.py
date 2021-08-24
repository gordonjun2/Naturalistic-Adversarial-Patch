import cv2
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image
from PIL import Image, ImageDraw, ImageFont
import matplotlib.image as mpimg 
import time
import pylab
import imageio
from tqdm import tqdm
from torch import autograd
from ensemble_tool.utils import *
from ensemble_tool.model import train_rowPtach, eval_rowPtach

from pytorchYOLOv4.tool.utils import load_class_names
from PyTorchYOLOv3.detect import DetectorYolov3
from pytorchYOLOv4.demo import DetectorYolov4
from adversarialYolo.demo import DetectorYolov2
from adversarialYolo.load_data import InriaDataset, PatchTransformer, PatchApplier

### -----------------------------------------------------------    Setting     ---------------------------------------------------------------------- ###
model_name            = "yolov4"    # yolov4, yolov3, yolov2
yolo_tiny             = True        # only yolov4, yolov3

##############################################
# Only detect patches captured by the camera #
############### Don't change #################
# patch shape
by_rectangle          = False
# transformation options
enable_rotation       = False
enable_randomLocation = False
enable_crease         = False
enable_projection     = False
enable_rectOccluding  = False
enable_blurred        = False
# output images with bbox
enable_with_bbox      = True
# other setting
enable_show_plt       = False
enable_clear_output   = True
enable_no_random      = True
# patch
cls_id_attacked       = 0
patch_scale           = 0.2
alpha_latent          = 0.99
max_labels_per_img    = 14
enable_check_patch    = False
##############################################

# detection threshold
cls_conf_threshold    = 0.5
# input size
img_size              = 416
# output size
img_size_output       = 1000


### ----------------------------------------------------------- Initialization ---------------------------------------------------------------------- ###
# init
plt2tensor = transforms.Compose([
        transforms.ToTensor()]) 
device = get_default_device()

cap = cv2.VideoCapture(0)
xres = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
yres = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
tensor = transforms.ToTensor()

# init patch_transformer and patch_applier
if torch.cuda.is_available():
    patch_transformer = PatchTransformer().cuda()
    patch_applier = PatchApplier().cuda()
else:
    patch_transformer = PatchTransformer()
    patch_applier = PatchApplier()


### -----------------------------------------------------------    Detector    ---------------------------------------------------------------------- ###
# select detector
if(model_name == "yolov2"):
    detectorYolov2 = DetectorYolov2(show_detail=False)
    detector = detectorYolov2
if(model_name == "yolov3"):
    detectorYolov3 = DetectorYolov3(show_detail=False, tiny=yolo_tiny)
    detector = detectorYolov3
if(model_name == "yolov4"):
    detectorYolov4   = DetectorYolov4(show_detail=False, tiny=yolo_tiny)
    detector = detectorYolov4


### -----------------------------------------------------------     Camera     ---------------------------------------------------------------------- ###
batch_size   = 1 # one by one
while(True):

    ret, frame = cap.read()
    if not ret: break
    
    # cv2 to numpy
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = tensor(img)
    frame = img.permute(1,2,0).contiguous().numpy()
    
    # to tensor
    img        = img.unsqueeze(0)
    img        = F.interpolate(img, size=img_size)
    imm_tensor = img
    imm_tensor = imm_tensor.to(device, torch.float)
    img_side   = imm_tensor.size()[-1]
    img_output = imm_tensor
    # print("imm_tensor size : "+str(imm_tensor.size()))
    # detect image
    max_prob_obj, max_prob_cls, bboxes = detector.detect(input_imgs=imm_tensor, cls_id_attacked=cls_id_attacked, with_bbox=True) # Be always with bbox
    # add patch
    # get bbox label.
    labels = []          # format:  (label, x_center, y_center, w, h)  ex:(0 0.5 0.6 0.07 0.22)
    labels_rescale = []  # format:  (label, confendence, left, top, right, bottom)  ex:(person 0.76 0.6 183.1 113.5 240.3 184.7)
    if(len(bboxes) == batch_size):
        ## ONLY batch_size = 1
        bbox = bboxes[0]
    if(model_name == "yolov3" or model_name == "yolov4"):
        for b in bbox:
            if (int(b[-1]) == int(cls_id_attacked)):
                label          = np.array([b[-1], (b[0]+b[2])/2.0, (b[1]+b[3])/2.0, (b[2]-b[0]), (b[3]-b[1]), b[4]], dtype=np.float32)
                labels.append(label)
                b[:-3] = b[:-3] * img_side
                label_rescale  = np.array([b[-1], b[-2], b[0], b[1], b[2], b[3]], dtype=np.float32)
                labels_rescale.append(label_rescale)
        labels = np.array(labels)
        labels_rescale = np.array(labels_rescale)
    elif(model_name == "yolov2"):
        for b in bbox:
            if (int(b[-1]) == int(cls_id_attacked)):
                label          = np.array([b[-1], b[0], b[1], b[2], b[3], b[4]], dtype=np.float32)
                labels.append(label)
                b[:-3] = b[:-3] * img_side
                label_rescale  = np.array([b[-1], b[-2], (b[0]-(b[2]/2.0)), (b[1]-(b[3]/2.0)), (b[0]+(b[2]/2.0)), (b[1]+(b[3]/2.0))], dtype=np.float32)
                labels_rescale.append(label_rescale)
        labels = np.array(labels)
        labels_rescale = np.array(labels_rescale)
    # Take only the top 14 largest of objectness_conf (max_labels_per_img)
    if(labels.shape[0]>0):
        num_bbox, _ = labels.shape
        if(num_bbox>max_labels_per_img):
            # sort
            labels_sorted  = labels[np.argsort(-labels[:,5])]
            labels_rescale_sorted = labels_rescale[np.argsort(-labels[:,5])]
            # clamp
            labels         = labels_sorted[:max_labels_per_img, 0:5]
            labels_rescale = labels_rescale_sorted[:max_labels_per_img]
        else:
            labels         = labels[:, 0:5] # without conf_obj
    if(len(labels) > 0):
        labels_tensor = plt2tensor(labels).to(device)
        p_img_batch, fake_images_denorm  = eval_rowPtach(generator=None, batch_size=batch_size, device=device
                                            , latent_shift=None, alpah_latent = None
                                            , input_imgs=imm_tensor, label=labels_tensor, patch_scale=patch_scale, cls_id_attacked=cls_id_attacked
                                            , denormalisation = False
                                            , model_name = model_name, detector = detector
                                            , patch_transformer = patch_transformer, patch_applier = patch_applier
                                            , by_rectangle = by_rectangle
                                            , enable_rotation = enable_rotation
                                            , enable_randomLocation = enable_randomLocation
                                            , enable_crease = enable_crease
                                            , enable_projection = enable_projection
                                            , enable_rectOccluding = enable_rectOccluding
                                            , enable_blurred = enable_blurred
                                            , enable_with_bbox = enable_with_bbox
                                            , enable_show_plt = enable_show_plt
                                            , enable_clear_output = enable_clear_output
                                            , cls_conf_threshold = cls_conf_threshold
                                            , enable_no_random = enable_no_random
                                            , fake_images_default = torch.zeros(3,128,128).cuda())
                                                                
        img_output = p_img_batch

    # change color channel
    permute_color = [2, 1, 0]
    img_output = img_output[:, permute_color,:,:]
    # resize
    img_output = F.interpolate(img_output, size=img_size_output)
    
    frame = img_output[0].cpu().permute(1,2,0).contiguous().numpy()
    
    cv2.imshow('frame', frame)
    cv2.waitKey(1)
    # if cv2.waitKey(1) & 0xFF == ord('q'): break

cv2.release()
cv2.destroyAllWindows()