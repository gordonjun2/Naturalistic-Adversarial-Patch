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
from tqdm import tqdm
from torch import autograd
from ensemble_tool.utils import *
from ensemble_tool.model import train_rowPtach, eval_rowPtach

from pytorchYOLOv4.tool.utils import load_class_names
# from pytorchVae.vae_sample import training_loader as dataloader_cifar10
from PyTorchYOLOv3.detect import DetectorYolov3
from pytorchYOLOv4.demo import DetectorYolov4
from adversarialYolo.demo import DetectorYolov2
from adversarialYolo.load_data import InriaDataset, PatchTransformer, PatchApplier
from pytorch_pretrained_detection import FasterrcnnResnet50, MaskrcnnResnet50

import pylab
import imageio

import count_map.main as eval_map
from ipdb import set_trace as st
import argparse


### -----------------------------------------------------------    Setting     ---------------------------------------------------------------------- ###
Gparser = argparse.ArgumentParser(description='Advpatch evaluation')
Gparser.add_argument('--model', default='yolov4', type=str, help='options : yolov2, yolov3, yolov4, fasterrcnn')
Gparser.add_argument('--tiny', action='store_true', help='options :True or False')
Gparser.add_argument('--patch', default='', help='patch location')
apt1, unpar = Gparser.parse_known_args()
print(apt1)
print()


### -----------------------------------------------------------    Setting     ---------------------------------------------------------------------- ###
model_name                 = apt1.model        # yolov4, yolov3, yolov2, fasterrcnn
yolo_tiny                  = apt1.tiny            # only yolov4, yolov3
by_rectangle               = True
# transformation options
enable_rotation            = False 
enable_randomLocation      = False
enable_crease              = False
enable_projection          = False 
enable_rectOccluding       = False 
enable_blurred             = False 
# output images with bbox
enable_with_bbox           = True            # outputs with bbox 
# other setting
enable_show_plt            = False           # check output images during testing by human
enable_no_random           = True            # NOT randon patch "light and shadow changes"
enable_check_patch         = False           # check input patch by human
# patch
cls_id_attacked            = 0               # ID of the object to which the patch is posted
patch_scale                = 0.2             # patch size
max_labels_per_img         = 14              # maximum number of objects per image
patch_mode                 = 0               # options: 0(patch), 1(white), 2(gray), 3(randon)
# fake_images_path           = "../adversarial-attack-ensemble/patch_sample/3output.png"
# fake_images_path           = "../adversarial-attack-ensemble/exp/exp07/generated/generated-images-1000.png"
fake_images_path = apt1.patch
# st()
# fake_images_path           = "../adversarial-attack-ensemble/exp/exp1/generated/cv.png"

# data source
video_name                 = "WIN_20210113_18_36_46_Pro"     # WIN_20200903_16_52_27_Pro, WIN_20200903_17_17_34_Pro, WIN_20210113_18_36_46_Pro
video_folder               = "./dataset/video/"
source_folder              = "./dataset/inria/Test/pos/"    # ../dataset/inria/Train/pos/   ,   ../dataset/inria/Test/pos/
# video or folder
source_key                 = 1     # 1:inria     0:video

# MAP
enable_count_map           = True
if yolo_tiny==True and model_name!="yolov2":
    sss = model_name+"tiny"
else:
    sss=model_name
label_labelRescale_folder = "./dataset/inria/Test/pos/yolo-labels-rescale_"+sss
enable_show_map_process    = False

# sss = sss+'_'+fake_images_path[35:40] # -6:-4
temp_f = fake_images_path.split('/')[2]
if temp_f[0]=='exp':
    sss = sss+'_'+temp_f
else:
    sss = sss+'_'+'temp_f'

# st()
# output path
output_video_name          = "video_output"
output_folder              = "eval_output/"+sss+"/"
output_video_foler         = output_folder + "video/"
output_imgs_foler          = output_folder + "output_imgs/"
output_labels_foler        = output_folder + "output_imgs/yolo-labels/"
outout_labelRescale_folder = output_folder + "output_imgs/yolo-labels-rescale/"

# output mode
output_mode                = 1               # options:  0(training data. no-patch and label without confidence)   /   1(evalution. with-pacth and label with confidence)
enable_output_data         = True            # options:  True (output bbox labels and images (clear & rescale) and video)   /    False (only video)


### ----------------------------------------------------------- Initialization ---------------------------------------------------------------------- ###
# init
plt2tensor = transforms.Compose([
        transforms.ToTensor()]) 
device = get_default_device()

# init output folder name
tiny_str = ""
if(yolo_tiny):
    if(model_name == "yolov3" or model_name == "yolov4"):
        tiny_str = "tiny"
if(model_name == "yolov2"):
    output_labels_foler        = output_labels_foler[:-1] + "_yolov2" + tiny_str + output_labels_foler[-1]
    outout_labelRescale_folder = outout_labelRescale_folder[:-1] + "_yolov2" + tiny_str + output_labels_foler[-1]
elif(model_name == "yolov3"):
    output_labels_foler        = output_labels_foler[:-1] + "_yolov3" + tiny_str + output_labels_foler[-1]
    outout_labelRescale_folder = outout_labelRescale_folder[:-1] + "_yolov3" + tiny_str + output_labels_foler[-1]
elif(model_name == "yolov4"):
    output_labels_foler        = output_labels_foler[:-1] + "_yolov4" + tiny_str + output_labels_foler[-1]
    outout_labelRescale_folder = outout_labelRescale_folder[:-1] + "_yolov4" + tiny_str + output_labels_foler[-1]
if(model_name == "fasterrcnn"):
    output_labels_foler        = output_labels_foler[:-1] + "_fasterrcnn" + tiny_str + output_labels_foler[-1]
    outout_labelRescale_folder = outout_labelRescale_folder[:-1] + "_fasterrcnn" + tiny_str + output_labels_foler[-1]

# init cls_conf_threshold
# options:  Test (labels-rescale contain [confidence])    /    Train (labels-rescale doesn't contain [confidence])
if(output_mode == 1):
    output_data_type = "Test"          
elif(output_mode == 0):
    output_data_type = "Train"
if(output_data_type == "Train"):
    cls_conf_threshold = 0.0
    enable_clear_output = True
elif(output_data_type == "Test"):
    cls_conf_threshold = 0.5
    enable_clear_output = False

# init patch_transformer and patch_applier
if torch.cuda.is_available():
    patch_transformer = PatchTransformer().cuda()
    patch_applier = PatchApplier().cuda()
else:
    patch_transformer = PatchTransformer()
    patch_applier = PatchApplier()

# make output folder
if(enable_output_data):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_video_foler, exist_ok=True)
    os.makedirs(output_imgs_foler, exist_ok=True)
    os.makedirs(output_labels_foler, exist_ok=True)
    os.makedirs(outout_labelRescale_folder, exist_ok=True)


### -----------------------------------------------------------  Source Images ---------------------------------------------------------------------- ###
# read source images
print(f"***** Load the Patch from {fake_images_path} *****")
start_r = time.time()
source_data = None
fps         = None
output_name = None
if(source_key == 0):
    print("Start to read images from video")
    # init
    filename = video_folder+str(video_name)+".mp4"
    vid = imageio.get_reader(filename,  'ffmpeg')
    fps = vid.get_meta_data()['fps']
    print("input video fps : "+str(fps))
    # number of frames
    nframes     = (len(list(enumerate(vid))))
    source_data = vid
    output_name = [video_name]
elif(source_key == 1):
    # read images
    print("Start to read images from folder")
    images = []
    filenames = []
    for filename in os.listdir(source_folder):
        if(filename.endswith('.jpg') or filename.endswith('.png') ):
            image = imageio.imread(source_folder+filename)
            images.append(image)
            filenames.append(filename[:-4])
    # number of frames
    nframes     = len(images)
    source_data = images
    output_name = filenames
finish_r = time.time()
print('Finish reading images in %f seconds.' % (finish_r - start_r))


### -----------------------------------------------------------  Patch  Image  ---------------------------------------------------------------------- ###
# Read patch image 
fake_images_input = Image.open(fake_images_path).convert('RGB')
f_width, f_height = fake_images_input.size  
new_side = max(f_width, f_height)
newsize = (new_side, new_side) 
fake_images_input = fake_images_input.resize(newsize) 
if(enable_check_patch):
    # Ckeck Images 
    fake_images_input.show()
# plt to tensor
plt2tensor = transforms.Compose([
            transforms.ToTensor()])
fake_images_input = plt2tensor(fake_images_input).unsqueeze(0)
fake_images_input = fake_images_input.to(device, torch.float)

if(patch_mode == 1):
    # white
    fake_images_input = torch.ones((3, fake_images_input.size()[-2], fake_images_input.size()[-1]), device=device).to(torch.float).unsqueeze(0)
elif(patch_mode == 2):
    # gray
    fake_images_input = torch.zeros((3, fake_images_input.size()[-2], fake_images_input.size()[-1]), device=device).to(torch.float).unsqueeze(0) +0.5
elif(patch_mode == 3):
    # randon
    fake_images_input = torch.rand((3, fake_images_input.size()[-2], fake_images_input.size()[-1]), device=device).to(torch.float).unsqueeze(0)


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
if(model_name == "fasterrcnn"):
    # just use fasterrcnn directly
    detector = None


### -----------------------------------------------------------  Output Video  ---------------------------------------------------------------------- ###
# output video
batch_size   = 1 # one by one
if(fps == None):
    fps = 2
video_writer = imageio.get_writer(output_video_foler + output_video_name + ".mp4", fps=fps)
for i, imm in tqdm(enumerate(source_data), desc=f'Output video ',total=nframes):
    # resize
    # if i>3: break
    imm = np.asarray(imm)
    img = Image.fromarray(imm, 'RGB')
    img = img.resize((416,416))
    # to tensor
    imm_tensor = plt2tensor(img).unsqueeze(0)
    imm_tensor = imm_tensor.to(device, torch.float)
    img_side   = imm_tensor.size()[-1]
    img_output = imm_tensor
    # print("imm_tensor size : "+str(imm_tensor.size()))

    # get clear label of input images
    # detect image. # Be always with bbox
    if(model_name == "yolov4"):
        max_prob_obj_cls, overlap_score, bboxes = detector.detect(input_imgs=imm_tensor, cls_id_attacked=cls_id_attacked, with_bbox=True)
    if(model_name == "yolov3"):
        max_prob_obj_cls, overlap_score, bboxes = detector.detect(input_imgs=imm_tensor, cls_id_attacked=cls_id_attacked, with_bbox=True)
    if(model_name == "yolov2"):
        max_prob_obj_cls, overlap_score, bboxes = detector.detect(input_imgs=imm_tensor, cls_id_attacked=cls_id_attacked, with_bbox=True)
    if(model_name == "fasterrcnn"):
        max_prob, max_prob, bboxes = fasterrcnn(tensor_image_inputs=imm_tensor, device=device, cls_id_attacked=cls_id_attacked, threshold=0.5)

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
    elif(model_name == "fasterrcnn"):
        for b in bbox:
            if (int(b[-1]) == int(cls_id_attacked)):
                label          = np.array([b[-1], (b[0]+b[2])/2.0, (b[1]+b[3])/2.0, (b[2]-b[0]), (b[3]-b[1]), b[4]], dtype=np.float32)
                labels.append(label)
                b[:-3] = b[:-3] * img_side
                label_rescale  = np.array([b[-1], b[-2], b[0], b[1], b[2], b[3]], dtype=np.float32)
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
    
    # set output name
    if(len(output_name) == 1):
        iname = output_name[0]+"_"+str(i)
    else:
        iname = output_name[i]

    # eval_rowPtach
    if(len(labels) > 0):
        labels_tensor = plt2tensor(labels).to(device)
        p_img_batch, fake_images_denorm, bboxes = eval_rowPtach(generator=None, batch_size=batch_size, device=device
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
                                                        , patch_mode=patch_mode
                                                        , enable_no_random = enable_no_random
                                                        , fake_images_default = fake_images_input)
                                                                
        img_output = p_img_batch
        if not(enable_clear_output):
            # get bbox label.
            labels = []          # format:  (label, x_center, y_center, w, h)  ex:(0 0.5 0.6 0.07 0.22)
            labels_rescale = []  # format:  (label, confendence, left, top, right, bottom)  ex:(person 0.76 0.6 183.1 113.5 240.3 184.7)
            if(len(bboxes) == batch_size):
                ## ONLY batch_size = 1
                bbox = bboxes[0]
            if(model_name == "yolov3" or model_name == "yolov4" or model_name == "fasterrcnn"):
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
    
    # output data
    if(enable_output_data):
        # save clear imgs
        output_path = str(output_imgs_foler)+'%s.png' % (iname)
        save_image(img_output, output_path)
        # save bbox
        output_path = str(output_labels_foler)+'%s.txt' % (iname)
        np.savetxt(output_path, labels, fmt='%.6f')
    if(enable_output_data):
        # save recale bbox
        output_path = outout_labelRescale_folder + iname + ".txt"
        labelfile_rescale = open(output_path,'w+') #read label
        for bbox in labels_rescale:
            if(output_data_type == "Train"):
                labelfile_rescale.write("person" + str(f' {bbox[2]} {bbox[3]} {bbox[4]} {bbox[5]}\n'))            # left, top, right, bottom
            elif(output_data_type == "Test"):
                labelfile_rescale.write("person" + str(f' {bbox[1]} {bbox[2]} {bbox[3]} {bbox[4]} {bbox[5]}\n'))  # confendence left, top, right, bottom
        labelfile_rescale.close()

    # output video
    img_output = img_output[0].cpu().detach().numpy()
    img_output = np.transpose(img_output, (1, 2, 0))
    img_output = 255 * img_output # Now scale by 255
    img_output = img_output.astype(np.uint8)
    video_writer.append_data(img_output)
video_writer.close()
# 
# st()
# MAP
if(enable_count_map):
    if not(enable_show_map_process):
        output_imgs_foler=None
    # st()
    output_map = eval_map.count(path_ground_truth=label_labelRescale_folder, 
                                path_detection_results=outout_labelRescale_folder, 
                                path_images_optional=output_imgs_foler)
    # save
    # with open("./"+output_folder+"map.txt", "w") as text_file:
    #     text_file.write(str(output_map))

print(fake_images_path)
if yolo_tiny==True and model_name!='yolov2':
    model_name = model_name+'_tiny'
print(model_name)
print('================ finish ================\n\n')

