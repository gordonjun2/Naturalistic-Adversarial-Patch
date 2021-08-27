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
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import time
from tqdm import tqdm
from torch import autograd
from ensemble_tool.utils import *
from ensemble_tool.model import train_rowPtach, TotalVariation
import random

from GANLatentDiscovery.loading import load_from_dir

from PyTorchYOLOv3.detect import DetectorYolov3
from pytorch_pretrained_detection import FasterrcnnResnet50, MaskrcnnResnet50
from pytorchYOLOv4.demo import DetectorYolov4
from adversarialYolo.demo import DetectorYolov2
from adversarialYolo.train_patch import PatchTrainer
from adversarialYolo.load_data import InriaDataset, PatchTransformer, PatchApplier
from GANLatentDiscovery.utils import is_conditional
from pathlib import Path
from stylegan2_pytorch import run_generator
from ipdb import set_trace as st
import argparse
import sys

"""
version: 2021.2.19.1200

yolov2-img-size: B,3,416,416
yolov3-img-size: B,3,416,416
yolov4-img-size: B,3,608,608

yolov3-tiny-img-size: B,3,416,416
yolov4-tiny-img-size: B,3,416,416
"""
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.enabled = False
### -----------------------------------------------------------    Setting     ---------------------------------------------------------------------- ###
Gparser = argparse.ArgumentParser(description='Advpatch Training')
Gparser.add_argument('--seed', default='15089',type=int, help='choose seed') 
Gparser.add_argument('--model', default='yolov4', type=str, help='options : yolov2, yolov3, yolov4, fasterrcnn')
Gparser.add_argument('--classBiggan', default=259, type=int, help='class in big gan') # 84:peacock
Gparser.add_argument('--tiny', action='store_true', help='options :True or False')
apt = Gparser.parse_known_args()[0]
print(apt)
print()
# st()

### -----------------------------------------------------------    Setting     ---------------------------------------------------------------------- ###

model_name            = apt.model  # options : yolov2, yolov3, yolov4, fasterrcnn
yolo_tiny             = apt.tiny      # hint    : only yolov3 and yolov4
dataset_second        = "inria"   # options : inria, test
by_rectangle          = True      # True: The patch on the character is "rectangular". / False: The patch on the character is "square"
# transformation options
enable_rotation       = False
enable_randomLocation = False
enable_crease         = False
enable_projection     = False
enable_rectOccluding  = False
enable_blurred        = False
# output images with bbox
enable_with_bbox      = True      # hint    : It is very time consuming. So, the result is only with bbox at checkpoint.
# other setting
enable_show_plt       = False     # check output images during training  by human
enable_clear_output   = False     # True: training data without any patch
multi_score           = True     # True: detection score is "class score * objectness score" for yolo.  /  False: detection score is only "objectness score" for yolo.
# loss weight
weight_loss_tv        = 0.1       # total variation loss rate    ([0-0.1])
weight_loss_overlap   = 0.0       # total bbox overlap loss rate ([0-0.1])
# training setting
retrain_gan           = False     # whether use pre-trained checkpoint 
patch_scale           = 0.2       # the scale of the patch attached to persons
n_epochs              = 1000      # training total epoch
start_epoch           = 1         # from what epoch to start training
learing_rate          = 0.02      # training learning rate. (hint v3~v4(~0.02) v2(~0.01))
epoch_save            = 10001       # from how many A to save a checkpoint
cls_id_attacked       = 0         # the class attacked. (0: person). List: https://gist.github.com/AruniRC/7b3dadd004da04c80198557db5da4bda
cls_id_generation     = apt.classBiggan       # the class generated at patch. (259: pomeranian) List: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
alpha_latent          = 1.0       # weight latent space. z = (alpha_latent * z) + ((1-alpha_latent) * rand_z); std:0.99
rowPatch_size         = 128       # the size of patch without gan. It's just like "https://openaccess.thecvf.com/content_CVPRW_2019/html/CV-COPS/Thys_Fooling_Automated_Surveillance_Cameras_Adversarial_Patches_to_Attack_Person_Detection_CVPRW_2019_paper.html"
method_num            = 2         # options : 0 (rowPatch without GAN. randon) / 2 (BigGAN) / 3 (styleGAN2)
# parameters of BigGAN
enable_shift_deformator           = False   # True: patch = G(deformator(z))  /  False: patch = G(z) 
enable_human_annotated_directions = False   # True: only vectors that annotated by human  /   False: all latent vectors
max_value_latent_item             = 10       # the max value of latent vectors
# enable_latent_clipping            = True    # added by kung. To clip the latent code when optimize
# pre-trained checkpoint
checkpoint_path       = "checkpoint/gan_params_10.pt"        # if "retrain_gan" equal "True", and then use this path.
# pre latent vectors
enable_init_latent_vectors       = False     # True: patch = G(init_z)  /  False: patch = G(randon_z) 
latent_vectors_path   = "../tool/GANLatentDiscovery/reverse_gan_output/g_z.npy"
enable_show_init_patch           = False     # check init-patch by human
enable_discriminator  =  False


### ----------------------------------------------------------- Initialization ---------------------------------------------------------------------- ###
# set random seed 
Seed = apt.seed  # 37564 7777
torch.manual_seed(Seed)
torch.cuda.manual_seed(Seed)
torch.cuda.manual_seed_all(Seed)
np.random.seed(Seed)
random.seed(Seed)
device = get_default_device() # cuda or cpu

# no enable_shift_deformator no enable_human_annotated_directions
if not(enable_shift_deformator):
    enable_human_annotated_directions = False
print("setting: enable_shift_deformator           : "+str(enable_shift_deformator))
print("setting: enable_human_annotated_directions : "+str(enable_human_annotated_directions))

# confirm folder
global_dir = increment_path(Path('./exp') / 'exp', exist_ok=False) # 'checkpoint'
global_dir = Path(global_dir)
checkpoint_dir = global_dir / 'checkpoint'
checkpoint_dir.mkdir(parents=True, exist_ok=True)
sample_dir = global_dir / 'generated'
sample_dir.mkdir(parents=True, exist_ok=True)
print(f"\n##### The results are saved at {global_dir}. #######\n")
np.savetxt(f"./{global_dir}/{apt}--latent:{max_value_latent_item}_normal.txt",[enable_rotation,enable_randomLocation,enable_crease,enable_projection,enable_rectOccluding,enable_blurred,])


# confirm training data (Second dataset)
label_folder_name = 'yolo-labels_' + str(model_name)
if(model_name == "yolov3" or model_name == "yolov4"):
    if(yolo_tiny):
        label_folder_name = label_folder_name + 'tiny'


# load the pre-trained from GANLatentDiscovery
if(method_num == 2):
    deformator, G, shift_predictor = load_from_dir(
        './GANLatentDiscovery/models/pretrained/deformators/BigGAN/',
        G_weights='./GANLatentDiscovery/models/pretrained/generators/BigGAN/G_ema.pth')
    generator_biggan = G
    if enable_discriminator == True:
        discriminator_biggan = None
    else:
        D = None
        discriminator_biggan = None
    # show discovered_annotation of BigGAN pretrained
    discovered_annotation = ''
    for d in deformator.annotation.items():
        discovered_annotation += '{}: {}\n'.format(d[0], d[1])
    print('setting: human-annotated directions:\n' + discovered_annotation)
    len_z = G.dim_z
    if(enable_human_annotated_directions):
        annotated_idx = list(deformator.annotation.values())
        len_latent = len(annotated_idx)
    else:
        annotated_idx = []
        len_latent = G.dim_z
    print("setting: len_latent : "+str(len_latent))
    print()
elif method_num ==3:
    stylegan_G = run_generator.get_style_gan2()
    len_z = stylegan_G.latent_size
    annotated_idx = []
    len_latent = len_z
    print("setting: len_latent : "+str(len_latent))
    print()
else:
    raise Exception("Only BigGAN and styleGAN you can choose")


# rowPatch.         input = delta
rowPatch            = torch.rand((3, rowPatch_size, rowPatch_size), device=device).requires_grad_(True)                  # the delta
# BigGAN input.     input = ((1-alpha) * fixed) + (alpha * delta)
fixed_latent_biggan = torch.rand(len_z, device=device)                                                                   # the fixed
# latent_shift_biggan = torch.rand(len_latent, device=device).requires_grad_(True) 
latent_shift_biggan = torch.normal(0.0, torch.ones(len_latent)).to(device).requires_grad_(True)                                        # the delta

if(enable_init_latent_vectors):
    # load z_approx (1, 120)
    with open(latent_vectors_path, 'rb') as f:
        z_loaded = np.load(f)
    # to tensor (tesnor size: 120)
    z_loaded_tensor = torch.from_numpy(z_loaded)[0]
    latent_shift_biggan = z_loaded_tensor.to(device).requires_grad_(True)

def show(img):
    npimg = img.numpy()
    fig = plt.imshow(np.transpose(npimg, (1, 2, 0)), 
                                    interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

if(enable_show_init_patch):
    # generator
    if is_conditional(generator_biggan):
        generator_biggan.set_classes(cls_id_generation)
    test_img = generator_biggan(latent_shift_biggan.unsqueeze(0))
    test_img = (test_img + 1) * 0.5
    # show
    show(test_img.cpu().detach()[0])
    plt.show()

### -----------------------------------------------------------    Detector    ---------------------------------------------------------------------- ###
# Detector

# YOLO v2~v4
### imput  size ###
# input_imgs      : torch.Size([batch_size, 3, 416, 416])
# cls_id_attacked : int
# clear_imgs      : torch.Size([batch_size, 3, 416, 416])
# with_bbox       : boolean

### output size ###
# max_prob_obj    : torch.Size([1])
# max_prob_cls    : torch.Size([1])
# overlap_score   : torch.Size([1])
# bboxes          : torch.Size([num_objs, 7])

start = time.time()
if(model_name == "yolov2"):
    detectorYolov2 = DetectorYolov2(show_detail=False)
    detector = detectorYolov2
    batch_size_second      = 8
    learing_rate          = 0.005
    # # ORIGIN
    # detector = PatchTrainer("paper_obj")
if(model_name == "yolov3"):
    detectorYolov3 = DetectorYolov3(show_detail=False, tiny=yolo_tiny)
    detector = detectorYolov3
    batch_size_second      = 16
    learing_rate          = 0.005
    if yolo_tiny==False:
        batch_size_second  = 4
if(model_name == "yolov4"):
    detectorYolov4   = DetectorYolov4(show_detail=False, tiny=yolo_tiny)
    detector = detectorYolov4
    batch_size_second      = 16
    learing_rate          = 0.005
    if yolo_tiny==False:
        batch_size_second=1
if(model_name == "fasterrcnn"):
    # just use fasterrcnn directly
    batch_size_second = 8
    detector = FasterrcnnResnet50()
if(model_name == "maskrcnn"):
    detector = MaskrcnnResnet50()
finish = time.time()
print('Load detector in %f seconds.' % (finish - start))


### -----------------------------------------------------------   DataLoader   ---------------------------------------------------------------------- ###
# DataLoader
# Second dataset
if(dataset_second == "inria"):
# InriaDataset
    ds_image_size_second   = 416
    # batch_size_second      = 8
    train_loader_second = torch.utils.data.DataLoader(InriaDataset(img_dir='./dataset/inria/Train/pos', 
                                                            lab_dir='./dataset/inria/Train/pos/'+str(label_folder_name), 
                                                            max_lab=14,
                                                            imgsize=ds_image_size_second,
                                                            shuffle=True),
                                                batch_size=batch_size_second,
                                                shuffle=True,
                                                num_workers=10)
elif(dataset_second == "test"):
# InriaDataset
    ds_image_size_second   = 416
    batch_size_second      = 16
    train_loader_second = torch.utils.data.DataLoader(InriaDataset(img_dir='./dataset/video/output_imgs', 
                                                            lab_dir='./dataset/video/output_imgs/yolo-labels', 
                                                            max_lab=14,
                                                            imgsize=ds_image_size_second,
                                                            shuffle=True),
                                                batch_size=batch_size_second,
                                                shuffle=True,
                                                num_workers=10)
# init
train_loader_second = DeviceDataLoader(train_loader_second, device)


# st()
# TV
if(device == "cuda"):
    total_variation = TotalVariation().cuda()
else:
    total_variation = TotalVariation()


### ---------------------------------------------------------- Checkpoint & Init -------------------------------------------------------------------- ###
# Training preprocess
epoch_length_second  = len(train_loader_second)
ep_loss_det   = 0
ep_loss_tv    = 0
torch.cuda.empty_cache()
# Create optimizers
opt_ap = torch.optim.Adam([rowPatch], lr=learing_rate, betas=(0.5, 0.999), amsgrad=True)
opt_ld = torch.optim.Adam([latent_shift_biggan], lr=learing_rate, betas=(0.5, 0.999), amsgrad=True)
# opt_ld = torch.optim.SGD([latent_shift_biggan], lr=learing_rate, momentum=0.9)
# optimizer lr_scheduler
scheduler_ap = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_ap, 'min', patience=50)
scheduler_ld = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_ld, 'min', patience=50)
# load checkpoint
if(retrain_gan):
    PATH = checkpoint_path
    #
    checkpoint = torch.load(PATH)
    epoch_start = checkpoint['epoch']
    start_epoch = epoch_start
    latent_shift_biggan = checkpoint['latent_shift_biggan'].to(device).requires_grad_(True)
    opt_ld = torch.optim.Adam([latent_shift_biggan], lr=learing_rate, betas=(0.5, 0.999), amsgrad=True)
    # The reason for DISABLE this is that if we don’t do this, the training results will be very similar.
    # opt_ld.load_state_dict(checkpoint['optimizer_state_dict_biggan']) 

    # optimizer lr_scheduler
    scheduler_ld = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_ld, 'min', patience=50)
writer = init_tensorboard(path = global_dir, name="gan_adversarial")

# init & and show the length of one epoch
print(f'One epoch lenght is {len(train_loader_second)}')
if torch.cuda.is_available():
    patch_transformer = PatchTransformer().cuda()
    patch_applier = PatchApplier().cuda()
p_img_batch = []
fake_images_denorm = []


### -----------------------------------------------------------  Select Method ---------------------------------------------------------------------- ###
# select method
main_generator       = None
main_scheduler       = None
main_optimizer       = None
main_latentShift     = None
main_denormalisation = None
main_deformator      = None
if(method_num == 0):
    # without GAN, just do gradient-descent with patch
    main_scheduler       = scheduler_ap
    main_optimizer       = opt_ap
    main_latentShift     = rowPatch
    main_denormalisation = False
elif(method_num == 2):
    # BigGAN
    main_generator       = generator_biggan
    main_discriminator   = discriminator_biggan
    main_scheduler       = scheduler_ld
    main_optimizer       = opt_ld
    main_latentShift     = latent_shift_biggan
    main_denormalisation = False
    main_deformator      = deformator
    # init
    if is_conditional(main_generator):
        main_generator.set_classes(cls_id_generation)
elif(method_num == 3):
    # stylegan2
    main_generator       = stylegan_G
    main_discriminator   = None
    main_scheduler       = scheduler_ld
    main_optimizer       = opt_ld
    main_latentShift     = latent_shift_biggan
    main_denormalisation = False
    main_deformator      = None
    # init
    if is_conditional(main_generator):
        main_generator.set_classes(cls_id_generation)


### -----------------------------------------------------------    Training    ---------------------------------------------------------------------- ###
for epoch in range(start_epoch, n_epochs+1):
    ep_loss_det = 0
    ep_loss_tv  = 0
    ep_loss_overlap  = 0
    for i_batch, (img_batch, lab_batch) in tqdm(enumerate(train_loader_second), desc=f'2 Running epoch {epoch}',total=epoch_length_second): ## , input_imgs=img_batch, label=lab_batch,
        with autograd.detect_anomaly():
            # only save the patched image, then enable_with_bbox. To reduce time consuming.
            if(epoch % epoch_save == 0):
                enable_with_bbox_dynamic = enable_with_bbox
            else:
                enable_with_bbox_dynamic = False
        
            # Train with GANLatentDiscovery
            # st()
            # opt_ld.zero_grad()
                                            # np.save('gg', latent_shift_biggan.cpu().detach().numpy())   
                                            # np.argwhere(np.load('gg.npy')!=latent_shift_biggan.cpu().detach().numpy())
            latent_shift_biggan.data = torch.round(latent_shift_biggan.data * 10000) * (10**-4)
            loss_det, loss_overlap, loss_tv, p_img_batch, fake_images_denorm, D_loss = train_rowPtach(method_num=method_num, generator=main_generator
                                                                    , discriminator = main_discriminator
                                                                    , opt=main_optimizer, batch_size=batch_size_second, device=device
                                                                    , latent_shift=latent_shift_biggan, alpah_latent=alpha_latent
                                                                    , input_imgs=img_batch, label=lab_batch, patch_scale=patch_scale, cls_id_attacked=cls_id_attacked
                                                                    , denormalisation=main_denormalisation
                                                                    , model_name = model_name, detector=detector
                                                                    , patch_transformer=patch_transformer, patch_applier=patch_applier
                                                                    , total_variation=total_variation
                                                                    , by_rectangle=by_rectangle
                                                                    , enable_rotation=enable_rotation
                                                                    , enable_randomLocation=enable_randomLocation
                                                                    , enable_crease=enable_crease
                                                                    , enable_projection=enable_projection
                                                                    , enable_rectOccluding=enable_rectOccluding
                                                                    , enable_blurred=enable_blurred
                                                                    , enable_with_bbox=enable_with_bbox_dynamic
                                                                    , enable_show_plt=enable_show_plt
                                                                    , enable_clear_output=enable_clear_output
                                                                    , weight_loss_tv=weight_loss_tv
                                                                    , weight_loss_overlap=weight_loss_overlap
                                                                    , multi_score=multi_score
                                                                    , deformator=main_deformator
                                                                    , fixed_latent_biggan=fixed_latent_biggan
                                                                    , max_value_latent_item=max_value_latent_item
                                                                    , enable_shift_deformator=enable_shift_deformator)


            # Tloss.backward()
            # opt_ld.step()
            # # Record loss and score
            ep_loss_det   += loss_det
            ep_loss_overlap += loss_overlap
            ep_loss_tv    += loss_tv
    # if enable_latent_clipping:
        # latent_shift_biggan = torch.clamp(latent_shift_biggan,-3,3)
    ep_loss_det   = ep_loss_det/epoch_length_second
    ep_loss_overlap = ep_loss_overlap/epoch_length_second
    ep_loss_tv    = ep_loss_tv/epoch_length_second

    ep_loss = ep_loss_det + (weight_loss_overlap * ep_loss_overlap)

    main_scheduler.step(ep_loss)

    
    ep_loss_det      = ep_loss_det.detach().cpu().numpy()
    ep_loss_overlap  = ep_loss_overlap.detach().cpu().numpy()
    ep_loss_tv       = ep_loss_tv.detach().cpu().numpy()
    
    writer.add_scalar('ep_loss_det', ep_loss_det, epoch)
    writer.add_scalar('ep_loss_overlap', ep_loss_overlap, epoch)
    writer.add_scalar('ep_loss_tv', ep_loss_tv, epoch)
    writer.add_scalar('D_loss', D_loss, epoch)
    writer.add_scalar('latent_code_inf_norm', torch.max(torch.abs(latent_shift_biggan)), epoch)
    writer.add_scalar('latent_code_1st_norm', torch.norm(latent_shift_biggan, p=1)/latent_shift_biggan.shape[0], epoch)

    print("ep_loss_det      : "+str(ep_loss_det))
    print("ep_loss_overlap  : "+str(ep_loss_overlap))
    print("ep_loss_tv       : "+str(ep_loss_tv))
    print("D_loss           : "+str(D_loss))
    print("latent code:     :'"+f"norn_inf:{torch.max(torch.abs(latent_shift_biggan)):.4f}; norm_1:{torch.norm(latent_shift_biggan, p=1)/latent_shift_biggan.shape[0]:.4f}")

    if(method_num == 0):
        # save patch
        save_samples(index=epoch, sample_dir=sample_dir, patch=rowPatch.cpu())
    if(method_num == 2):
        # save patch
        print(f"Save at: {global_dir}")
        save_samples_GANLatentDiscovery(method_num=method_num,
                                        index=epoch, sample_dir=sample_dir, 
                                        deformator=deformator, G=main_generator, 
                                        latent_shift=latent_shift_biggan, param_rowPatch_latent=alpha_latent, fixed_rand_latent=fixed_latent_biggan, 
                                        max_value_latent_item=max_value_latent_item, 
                                        enable_shift_deformator=enable_shift_deformator, 
                                        device=device)
    elif method_num == 3:
        save_samples_GANLatentDiscovery(method_num=method_num,
                                        index=epoch, sample_dir=sample_dir, 
                                        deformator=None, G=main_generator, 
                                        latent_shift=latent_shift_biggan, param_rowPatch_latent=alpha_latent, fixed_rand_latent=fixed_latent_biggan, 
                                        max_value_latent_item=max_value_latent_item, 
                                        enable_shift_deformator=enable_shift_deformator, 
                                        device=device)

    if(epoch % epoch_save == 0):
        # # save the patched image
        # print(f"@{global_dir}")
        save_the_patched(index=epoch, the_patched=p_img_batch, sample_dir=sample_dir, show=False)
        # # save checkpoint
        # Additional information
        EPOCH  = epoch
        PATH   = str(checkpoint_dir) + "/gan_params_"+str(epoch)+".pt"
        torch.save({
                    'epoch': EPOCH,
                    'optimizer_state_dict_biggan': opt_ld.state_dict(),
                    'latent_shift_biggan':latent_shift_biggan.data,
                    'alpha_latent':alpha_latent,
                    'annotated_idx':annotated_idx,
                    'enable_shift_deformator':enable_shift_deformator,
                    'enable_human_annotated_directions':enable_human_annotated_directions,
                    'ep_loss_det':ep_loss_det,
                    'ep_loss_overlap':ep_loss_overlap,
                    'ep_loss_tv':ep_loss_tv
                    }, PATH)
        print(f"save checkpoint: "+str(PATH))
writer.close()
