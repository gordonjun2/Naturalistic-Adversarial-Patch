"""
Training code for Adversarial patch training


"""

import PIL
import load_data
from tqdm import tqdm

from load_data import *
import gc
import matplotlib.pyplot as plt
from torch import autograd
from torchvision import transforms
from tensorboardX import SummaryWriter
import subprocess

import patch_config
import sys
import time
import random

import os
import os.path
from ckp import save_ckp, load_ckp
from utils import get_region_boxes, nms, do_detect
from PIL import Image, ImageDraw

class PatchTrainer(object):
    def __init__(self, mode):
        self.config = patch_config.patch_configs[mode]()

        self.darknet_model = Darknet(self.config.cfgfile)
        self.darknet_model.load_weights(self.config.weightfile)
        self.model_iheight = self.darknet_model.height
        self.model_iwidth  = self.darknet_model.width
        # self.darknet_model = torch.nn.DataParallel(self.darknet_model)
        self.darknet_model = self.darknet_model.eval().cuda() # TODO: Why eval?
        
        self.patch_applier = PatchApplier().cuda()
        self.patch_transformer = PatchTransformer().cuda()
        self.prob_extractor = MaxProbExtractor(0, 80, self.config).cuda()
        print("self.config.patch_size : "+str(self.config.patch_size))
        self.nps_calculator_local = NPSCalculator(self.config.printfile, int(self.config.patch_size/2)).cuda()
        self.nps_calculator_global = NPSCalculator(self.config.printfile, self.config.patch_size).cuda()
        self.css_calculator = CSSCalculator(sample_img=self.read_image(self.config.sampleimgfile)).cuda()
        self.total_variation = TotalVariation().cuda()

        self.output_file_name = "init"

        self.writer = self.init_tensorboard(mode)

    def init_tensorboard(self, name=None):
        # subprocess.Popen(['tensorboard', '--logdir=runs'])
        if name is not None:
            # time_str = time.strftime("%Y%m%d-%H%M%S")
            # time_str = time.strftime("%Y%m%d_28")
            time_str = time.strftime("%Y%m%d_test")
            self.output_file_name = time_str + "_paper_obj"
            print("init_tensorboard / time: "+str(self.output_file_name))
            return SummaryWriter(f'runs/{time_str}_{name}')
        else:
            print("init_tensorboard ("+str(name)+")")
            return SummaryWriter()

    def train(self):
        """
        Optimize a patch to generate an adversarial example.
        :return: Nothing
        """

        img_size = self.model_iheight
        batch_size = self.config.batch_size
        # n_epochs = 10000
        n_epochs = 600
        save_epochs = 1
        start_local_epoch = 0 # 0: It's from global (-1), and then the next is local (1~3). (For cyclic_update_mode = True or False)
        start_epoch = 1
        max_lab = 14 # Inira: 14, COCO: 7
        global_patch_scale = 0.2
        local_patch_scale  = 0.15
        local_mode = 1 # 0: Sequential / 1: Random
        cyclic_update_mode = True
        cyclic_update_step = 150
        local_patch_area_per_image_rate = 0.6
        p_nps = 0.01
        p_tv  = 2.5
        p_css = 50
        enable_loss_css = True
        _local_patch_size = int(self.config.patch_size / 2)
        _num_local_side = int(int(int(self.config.patch_size - _local_patch_size) / cyclic_update_step) + 1)
        _num_local = _num_local_side * _num_local_side
        _flag_no_avaliable_size_obj = True
        

        time_str = time.strftime("%Y%m%d-%H%M%S")

        # Generate stating point
        adv_patch_cpu = self.generate_patch("file", dim=3)  # torch.Size([3, 300, 300])
        # adv_patch_cpu = self.read_image("saved_patches/patch11.jpg")
        # print("adv_patch_cpu size : "+str(adv_patch_cpu.size()))

        adv_patch_cpu.requires_grad_(True)
        _previous_adv_patch_local_set = []
        # # show
        # im = transforms.ToPILImage('RGB')(adv_patch_cpu)
        # plt.imshow(im)
        # plt.pause(0.001)  # in 0.001 sec
        # plt.ioff()  # After displaying, be sure to use plt.ioff() to close the interactive mode, otherwise there may be strange problems
        # plt.clf()  # Clear the image
        # plt.close()  # Clear the window

        train_loader = torch.utils.data.DataLoader(InriaDataset(self.config.img_dir, self.config.lab_dir, max_lab, img_size,
                                                                shuffle=True),
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=10)
        self.epoch_length = len(train_loader)
        print(f'One epoch is {len(train_loader)}')

        # init
        optimizer = optim.Adam([adv_patch_cpu], lr=self.config.start_learning_rate, amsgrad=True)
        scheduler = self.config.scheduler_factory(optimizer)
        # if os.path.exists(self.config.target_checkpoint_path):
        #     print(f'Load ckeck point {str(self.config.target_checkpoint_path)}')
        #     model, scheduler, start_epoch, valid_loss_min, ep_det_loss , ep_nps_loss, ep_tv_loss, ep_loss = load_ckp(self.config.target_checkpoint_path, self.darknet_model, scheduler)
        #     self.darknet_model = model

        index_list_local = [i for i in range(0,_num_local,1)]
        print("init index_list_local : "+str(index_list_local))
        index_list_local = []
        index_local = 0
        ep_det_loss_g = 1
        ep_det_loss_l0 = 1
        ep_det_loss_l1 = 1
        ep_det_loss_l2 = 1
        ep_det_loss_l3 = 1
        ep_nps_loss_g = 1
        ep_nps_loss_l0 = 1
        ep_nps_loss_l1 = 1
        ep_nps_loss_l2 = 1
        ep_nps_loss_l3 = 1
        ep_tv_loss_g = 1
        ep_tv_loss_l0 = 1
        ep_tv_loss_l1 = 1
        ep_tv_loss_l2 = 1
        ep_tv_loss_l3 = 1
        local_det_loss_0 = torch.tensor(0).cuda()
        local_nps_loss_0 = torch.tensor(0).cuda()
        local_tv_loss_0  = torch.tensor(0).cuda()
        local_det_loss_1 = torch.tensor(0).cuda()
        local_nps_loss_1 = torch.tensor(0).cuda()
        local_tv_loss_1  = torch.tensor(0).cuda()
        local_det_loss_2 = torch.tensor(0).cuda()
        local_nps_loss_2 = torch.tensor(0).cuda()
        local_tv_loss_2  = torch.tensor(0).cuda()
        local_det_loss_3 = torch.tensor(0).cuda()
        local_nps_loss_3 = torch.tensor(0).cuda()
        local_tv_loss_3  = torch.tensor(0).cuda()
        global_det_loss = torch.tensor(0).cuda()
        global_nps_loss = torch.tensor(0).cuda()
        global_tv_loss  =torch.tensor(0).cuda()
        et0 = time.time()
        for epoch in range(start_epoch, n_epochs+1):
            ep_det_loss = 0
            ep_nps_loss = 0
            ep_css_loss = 0
            ep_tv_loss = 0
            ep_loss = 0
            
            bt0 = time.time()
            
            if(epoch > start_local_epoch):
                if(local_mode == 0):
                    # Sequential
                    index_local = epoch % _num_local
                elif(local_mode == 1):
                    # Random
                    if(len(index_list_local) == 0):
                        if(index_local == -1):
                            # reset, after final global test
                            index_list_local = [i for i in range(0,_num_local,1)]
                        else:
                            # set final global test
                            if(index_local > -1): 
                                index_local = -1
                            else:
                                index_local = index_local -1
                        
                    if not(len(index_list_local) == 0):
                        index_local = random.choice(index_list_local)
                        index_list_local.remove(index_local)
                else:
                    raise(NotImplementedError)
            else:
                index_local = -1

            for i_batch, (img_batch, lab_batch) in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}',
                                                        total=self.epoch_length):
                with autograd.detect_anomaly():
                    img_batch = img_batch.cuda()
                    lab_batch = lab_batch.cuda()
                    #print('TRAINING EPOCH %i, BATCH %i'%(epoch, i_batch))
                    adv_patch = adv_patch_cpu.cuda()
                    # print("adv_patch size: "+str(adv_patch.size()))  ##  adv_patch size: torch.Size([3, 300, 300])
                    # print("lab_batch size: "+str(lab_batch.size()))  ##  lab_batch size: torch.Size([8, 14, 5])
                    # print("img_size: "+str(img_size)) ## img_size: 416
                    
                    adv_patch_local_set = []
                    patch_unit = int(_local_patch_size)
                    for x_unit in range(0,(int(self.config.patch_size - _local_patch_size)+1),cyclic_update_step):
                        _s_xp = int(x_unit)
                        for y_unit in range(0,(int(self.config.patch_size - _local_patch_size)+1),cyclic_update_step):
                            _s_yp = int(y_unit)
                            adv_patch_part = torch.narrow(torch.narrow(adv_patch, 1, _s_xp, patch_unit), 2, _s_yp, patch_unit)
                            adv_patch_local_set.append(adv_patch_part)
                    adv_patch_input = adv_patch_local_set[-1]

                    patch_unit = int(self.config.patch_size / 2)
                    adv_patch_01 = torch.narrow(torch.narrow(adv_patch, 1, 0, patch_unit), 2, 0, patch_unit)
                    adv_patch_02 = torch.narrow(torch.narrow(adv_patch, 1, 0, patch_unit), 2, patch_unit, patch_unit)
                    adv_patch_03 = torch.narrow(torch.narrow(adv_patch, 1, patch_unit, patch_unit), 2, 0, patch_unit)
                    adv_patch_04 = torch.narrow(torch.narrow(adv_patch, 1, patch_unit, patch_unit), 2, patch_unit, patch_unit)
                    # adv_patch_input = adv_patch_04
                    
                    # print("adv_patch_01 size : "+str(adv_patch_01.size()))  # torch.Size([3, 150, 150])
                    # print("adv_patch_02 size : "+str(adv_patch_02.size()))
                    # print("adv_patch_03 size : "+str(adv_patch_03.size()))
                    # print("adv_patch_04 size : "+str(adv_patch_04.size()))

                    
                    # if not(index_local <= -1): # for local
                    #     max_bbox_aera_list = []
                    #     image_weight_list = []
                    #     for lab_per_image in lab_batch:
                    #         max_bbox_aera_per_img = -999
                    #         for lab_per_bbox in lab_per_image:
                    #             if(lab_per_bbox[0] == 0): # person
                    #                 x_center = lab_per_bbox[1]
                    #                 y_center = lab_per_bbox[2]
                    #                 width    = lab_per_bbox[3]
                    #                 height   = lab_per_bbox[4]
                    #                 bbox_area = width * height
                    #                 if(max_bbox_aera_per_img < bbox_area):
                    #                     max_bbox_aera_per_img = bbox_area
                    #         max_bbox_aera_list.append(max_bbox_aera_per_img)
                    #         if(max_bbox_aera_per_img > local_patch_area_per_image_rate):
                    #             local_patch_weight = 1
                    #             _flag_no_avaliable_size_obj = False
                    #         else:
                    #             local_patch_weight = 0
                    #         image_weight_list.append(local_patch_weight)
                    #     # print("max_bbox_aera_list size : "+str((max_bbox_aera_list)))  # 8
                    #     # print("image_weight_list : "+str(image_weight_list))  ##  torch.Size([8])
                    #     if(_flag_no_avaliable_size_obj):
                    #         if(cyclic_update_mode):
                    #             print("skip")
                    #             continue
                    # # print("alive~~~~~~~~~~~~~~~~~~")
                    
                    def get_patch_det_loss(adv_patch_input, patch_scale=0.2, need_patch_set = False, isLocal=False, with_bbox=False):

                        adv_batch_t, adv_patch_set = self.patch_transformer(adv_patch_input, lab_batch, img_size, do_rotate=False, rand_loc=False, TRAIN_LOCAL=True, scale_rate = patch_scale, with_crease=False, with_projection=False)

                        # print("img_batch size: "+str(img_batch.size()))  ##  torch.Size([8, 3, 416, 416])
                        # print("adv_batch_t size: "+str(adv_batch_t.size()))  ##  torch.Size([8, 14, 3, 416, 416])
                        # print("adv_patch_set size: "+str(adv_patch_set.size()))  ##  torch.Size([3, 300, 300])

                        # # set gray img
                        # img_batch = img_batch.cpu()
                        # img_batch[:,0,:,:] = img_batch[:,0,:,:] * 0.2989 +  img_batch[:,1,:,:] * 0.5870 + img_batch[:,2,:,:] * 0.1140
                        # img_batch[:,1,:,:] = img_batch[:,0,:,:]
                        # img_batch[:,2,:,:] = img_batch[:,0,:,:]
                        # img_batch = img_batch.cuda()


                        p_img_batch = self.patch_applier(img_batch, adv_batch_t)

                        # print("p_img_batch patch_applier size: "+str(p_img_batch.size()))  ##  torch.Size([8, 3, 416, 416])
                        p_img_batch = F.interpolate(p_img_batch, (self.model_iheight, self.model_iwidth))
                        # p_img_batch0 = F.interpolate(p_img_batch0, (self.darknet_model.height, self.darknet_model.width))
                        # p_img_batch1 = F.interpolate(p_img_batch1, (self.darknet_model.height, self.darknet_model.width))
                        # p_img_batch2 = F.interpolate(p_img_batch2, (self.darknet_model.height, self.darknet_model.width))
                        # p_img_batch3 = F.interpolate(p_img_batch3, (self.darknet_model.height, self.darknet_model.width))
                        # p_img_batch.show()
                        # print("p_img_batch size: "+str(p_img_batch.size()))  ##  torch.Size([8, 3, 416, 416])

                        img = p_img_batch[0, :, :,]  # torch.Size([3, 416, 416])
                        # print("img size: "+str(img.size()))
                        img = img.detach().cpu()
                        if(with_bbox):
                            trans_2pilimage = transforms.ToPILImage()
                            img_pil = trans_2pilimage(img)
                            boxes = do_detect(self.darknet_model, img_pil, 0.4, 0.4, True)
                            boxes = nms(boxes, 0.4)
                            for box in boxes:
                                cls_id = box[6]
                                if(cls_id == 0):   #if person
                                    det_score   = box[4]
                                    c_cla_score = box[5]
                                    cla_score   = det_score * c_cla_score
                                    if(cla_score > 0.5): # detection confidence
                                        x_center    = box[0]
                                        y_center    = box[1]
                                        width       = box[2]
                                        height      = box[3]
                                        left        = (x_center.item() - width.item() / 2) * img_pil.size[0]
                                        right       = (x_center.item() + width.item() / 2) * img_pil.size[0]
                                        top         = (y_center.item() - height.item() / 2) * img_pil.size[0]
                                        bottom      = (y_center.item() + height.item() / 2) * img_pil.size[0]
                                        # img with prediction
                                        draw = ImageDraw.Draw(img_pil)
                                        shape_ = [(x_center.item() - width.item() / 2),
                                                (y_center.item() - height.item() / 2), 
                                                (x_center.item() + width.item() / 2),
                                                (y_center.item() + height.item() / 2)]
                                        shape = [ tt * img_pil.size[0]  for tt in shape_]
                                        draw.rectangle(shape, outline ="red")
                                        # text
                                        color = [255,0,0]
                                        sentence = "person\n(" + str(round(float(det_score), 2)) + ", " + str(round(float(c_cla_score), 2)) + ")"
                                        position = [((x_center.item() - width.item() / 2) * img_pil.size[0]),
                                                ((y_center.item() - height.item() / 2) * img_pil.size[0])]
                                        draw.text(tuple(position), sentence, tuple(color))
                            trans_2tensor = transforms.ToTensor()
                            img = trans_2tensor(img_pil)
                            # print("img size: "+str(img.size()))


                        # img_plt = transforms.ToPILImage()(img)
                        # img_plt.show()
                        # print("img_plt ToPILImage size: "+str(img_plt.size))

                        img_b = adv_batch_t[0, 0, :, :,]
                        img_b = img_b.detach().cpu()
                        # img_plt = transforms.ToPILImage()(img_b)
                        # img_plt.show()
                        # time.sleep(100)

                        # print("p_img_batch size: "+str(p_img_batch.size()))  ##  torch.Size([8, 3, 416, 416])
                        output = self.darknet_model(p_img_batch)
                        max_prob = self.prob_extractor(output)
                        del output, p_img_batch
                        # output0 = self.darknet_model(p_img_batch0)
                        # max_prob0 = self.prob_extractor(output0)
                        # del output0, p_img_batch0
                        # output1 = self.darknet_model(p_img_batch1)
                        # max_prob1 = self.prob_extractor(output1)
                        # del output1, p_img_batch1
                        # output2 = self.darknet_model(p_img_batch2)
                        # max_prob2 = self.prob_extractor(output2)
                        # del output2, p_img_batch2
                        # output3 = self.darknet_model(p_img_batch3)
                        # max_prob3 = self.prob_extractor(output3)
                        # del output3, p_img_batch3

                        # max_prob = max_prob + max_prob0 + max_prob1 + max_prob2 + max_prob3

                        # print("output size: "+str(output.size()))  ##  torch.Size([8, 425, 13, 13])
                        # #
                        # for output_per_img in output:
                        #     output_per_img = output_per_img.unsqueeze(0)
                        #     print("output_per_img size : "+str(output_per_img.size()))
                        #     conf_thresh = 0.4
                        #     nms_thresh = 0.4
                        #     boxes = get_region_boxes(output, conf_thresh, self.darknet_model.num_classes, self.darknet_model.anchors, self.darknet_model.num_anchors)[0]
                        #     boxes = nms(boxes, nms_thresh)
                        #     print("boxes size : "+str(len(boxes)))
                        #     for box in boxes:
                        #         cls_id = box[6]
                        #         if(cls_id == 0):   #if person
                        #             x_center    = box[0]
                        #             y_center    = box[1]
                        #             width       = box[2]
                        #             height      = box[3]
                        #             det_score   = box[4]
                        #             c_cla_score = box[5]
                        #             cla_score   = det_score * c_cla_score
                        #             bbox_area = width * height
                        #             print("bbox_area : "+str(bbox_area))
                        #             bbox_area_score = min(0.2, bbox_area)
                        #             local_patch_weight = min(1.0, (bbox_area_score * 5))
                        #             global_patch_weight = 1.0-local_patch_weight
                        #
                        
                        # print("A max_prob size: "+str(max_prob))  ##  torch.Size([8])
                        
                        # if(cyclic_update_mode or isLocal):
                        #     if not(index_local <= -1): # local
                        #         if(len(max_prob) == len(image_weight_list)):
                        #             for t in range(len(max_prob)):
                        #                 max_prob[t] = image_weight_list[t] * max_prob[t]
                        #         # print("B max_prob size: "+str(max_prob))  ##  torch.Size([8])
                        #         max_prob = max_prob[torch.nonzero(max_prob)]
                        #         # print("C max_prob size: "+str(max_prob))  ##  torch.Size([8])
                        #         if(max_prob.size()[0]==0):
                        #             max_prob = torch.zeros(1)
                        #         # print("D max_prob size: "+str(max_prob))  ##  torch.Size([8])
                        det_loss = torch.mean(max_prob)
                        # print("det_loss: "+str(det_loss))

                        if not(need_patch_set):
                            del adv_batch_t, max_prob, adv_patch_set
                            return det_loss, torch.zeros(1), img, img_b
                        else:
                            del adv_batch_t, max_prob
                            return det_loss, adv_patch_set, img, img_b

                    adv_patch_input = adv_patch
                    if(cyclic_update_mode):
                        patch_scale = local_patch_scale
                        if(index_local <= -1):
                            adv_patch_input = adv_patch
                            patch_scale = global_patch_scale
                        else:
                            adv_patch_input = adv_patch_local_set[index_local]
                            # print("adv_patch_input size : "+str(adv_patch_input.size()))
                            # patch_scale = 0.15
                        # if(index_local == 0):
                        #     adv_patch_input = adv_patch_01
                        #     patch_scale = 0.15
                        # elif(index_local == 1):
                        #     adv_patch_input = adv_patch_02
                        #     patch_scale = 0.15
                        # elif(index_local == 2):
                        #     adv_patch_input = adv_patch_03
                        #     patch_scale = 0.15
                        # elif(index_local == 3):
                        #     adv_patch_input = adv_patch_04
                        #     patch_scale = 0.15
                        # elif(index_local == -1):
                        #     adv_patch_input = adv_patch
                        #     patch_scale = 0.3
                        det_loss, adv_patch_set, g_img_patched, g_img_patch = get_patch_det_loss(adv_patch_input, patch_scale, need_patch_set = True)
                    else:
                        if(epoch > start_local_epoch):
                            patch_scale = local_patch_scale
                            det_loss_1, adv_patch_set_1, g_img_patched_1, g_img_patch_1 = get_patch_det_loss(adv_patch_01, patch_scale, need_patch_set = True, isLocal=True)
                            det_loss_2, adv_patch_set_2, g_img_patched_2, g_img_patch_2 = get_patch_det_loss(adv_patch_02, patch_scale, need_patch_set = True, isLocal=True)
                            det_loss_3, adv_patch_set_3, g_img_patched_3, g_img_patch_3 = get_patch_det_loss(adv_patch_03, patch_scale, need_patch_set = True, isLocal=True)
                            det_loss_4, adv_patch_set_4, g_img_patched_4, g_img_patch_4 = get_patch_det_loss(adv_patch_04, patch_scale, need_patch_set = True, isLocal=True)
                            patch_scale = global_patch_scale
                            det_loss_g, adv_patch_set, g_img_patched, g_img_patch = get_patch_det_loss(adv_patch, patch_scale, need_patch_set = True)
                            det_loss = (det_loss_1 + det_loss_2 + det_loss_3 + det_loss_4 + det_loss_g) / 5
                        else:
                            patch_scale = global_patch_scale
                            det_loss_g, adv_patch_set, g_img_patched, g_img_patch = get_patch_det_loss(adv_patch, patch_scale, need_patch_set = True)
                            det_loss = det_loss_g

                    if(enable_loss_css):
                        if(cyclic_update_mode):
                            if not(index_local <= -1): # local
                                if(len(_previous_adv_patch_local_set)>0):
                                    _previous_part_patch = _previous_adv_patch_local_set[index_local]
                                    css = self.css_calculator(adv_patch_input, _previous_part_patch)
                                    _previous_adv_patch_local_set = adv_patch_local_set
                            else:
                                if(len(_previous_adv_patch_local_set)==0):
                                    # init
                                    _previous_adv_patch_local_set = adv_patch_local_set
                        else:
                            if(epoch > start_local_epoch):
                                if(len(_previous_adv_patch_local_set)>0):
                                    _previous_adv_patch_01 = _previous_adv_patch_local_set[0]
                                    _previous_adv_patch_02 = _previous_adv_patch_local_set[1]
                                    _previous_adv_patch_03 = _previous_adv_patch_local_set[2]
                                    _previous_adv_patch_04 = _previous_adv_patch_local_set[3]
                                    css_1 = self.css_calculator(adv_patch_01, _previous_adv_patch_01)
                                    css_2 = self.css_calculator(adv_patch_02, _previous_adv_patch_02)
                                    css_3 = self.css_calculator(adv_patch_03, _previous_adv_patch_03)
                                    css_4 = self.css_calculator(adv_patch_04, _previous_adv_patch_04)
                                    css = (css_1 + css_2 + css_3 + css_4) / 4
                                    _previous_adv_patch_local_set = adv_patch_local_set
                                else:
                                    if(len(_previous_adv_patch_local_set)==0):
                                        # init
                                        _previous_adv_patch_local_set = adv_patch_local_set


                    if(cyclic_update_mode):
                        if(index_local <= -1):
                            nps = self.nps_calculator_global(adv_patch_input)
                        else:
                            nps = self.nps_calculator_local(adv_patch_input)
                    else:
                        if(epoch > start_local_epoch):
                            nps_1 = self.nps_calculator_local(adv_patch_01)
                            nps_2 = self.nps_calculator_local(adv_patch_02)
                            nps_3 = self.nps_calculator_local(adv_patch_03)
                            nps_4 = self.nps_calculator_local(adv_patch_04)
                            nps_g = self.nps_calculator_global(adv_patch)
                            nps = (nps_1 + nps_2 + nps_3 + nps_4 + nps_g) / 5
                        else:
                            nps_g = self.nps_calculator_global(adv_patch)
                            nps = nps_g

                    if(cyclic_update_mode):
                        tv = self.total_variation(adv_patch_input)
                    else:
                        if(epoch > start_local_epoch):
                            tv_1 = self.total_variation(adv_patch_01)
                            tv_2 = self.total_variation(adv_patch_02)
                            tv_3 = self.total_variation(adv_patch_03)
                            tv_4 = self.total_variation(adv_patch_04)
                            tv_g = self.total_variation(adv_patch)
                            tv = (tv_1 + tv_2 + tv_3 + tv_4 + tv_g) / 5
                        else:
                            tv_g = self.total_variation(adv_patch)
                            tv = tv_g
                    # print("tv size: "+str(tv.size()))  ##  torch.Size([])
                    # print("tv     : "+str(tv.data))


                    nps_loss = nps*p_nps
                    if(enable_loss_css):
                        if not(index_local <= -1): # local
                            css_loss = css*p_css
                    tv_loss = tv*p_tv

                    if(cyclic_update_mode):
                        if(index_local <= -1):
                            global_det_loss = det_loss
                            global_nps_loss = nps_loss
                            global_tv_loss  = tv_loss
                        else:
                            local_det_loss_0 = det_loss
                            local_nps_loss_0 = nps_loss
                            local_tv_loss_0  = tv_loss

                        # if(index_local == 0):
                        #     local_det_loss_0 = det_loss
                        #     local_nps_loss_0 = nps_loss
                        #     local_tv_loss_0  = tv_loss
                        # elif(index_local == 1):
                        #     local_det_loss_1 = det_loss
                        #     local_nps_loss_1 = nps_loss
                        #     local_tv_loss_1  = tv_loss
                        # elif(index_local == 2):
                        #     local_det_loss_2 = det_loss
                        #     local_nps_loss_2 = nps_loss
                        #     local_tv_loss_2  = tv_loss
                        # elif(index_local == 3):
                        #     local_det_loss_3 = det_loss
                        #     local_nps_loss_3 = nps_loss
                        #     local_tv_loss_3  = tv_loss
                        # elif(index_local == -1):
                        #     global_det_loss = det_loss
                        #     global_nps_loss = nps_loss
                        #     global_tv_loss  = tv_loss
                    else:
                        global_det_loss = det_loss
                        global_nps_loss = nps_loss
                        global_tv_loss  = tv_loss
                    # det_loss = global_det_loss * 0.1 + local_det_loss_0 * 0.9
                    if(enable_loss_css):
                        if not(index_local <= -1): # local
                            loss = det_loss + nps_loss + css_loss + torch.max(tv_loss, torch.tensor(0.1).cuda())
                        else:
                            loss = det_loss + nps_loss + torch.max(tv_loss, torch.tensor(0.1).cuda())
                    else:
                        loss = det_loss + nps_loss + torch.max(tv_loss, torch.tensor(0.1).cuda())
                    # loss = det_loss + css_loss + torch.max(tv_loss, torch.tensor(0.1).cuda())

                    ep_det_loss    += det_loss.detach().cpu().numpy()
                    ep_det_loss_g  += global_det_loss.detach().cpu().numpy()
                    ep_det_loss_l0 += local_det_loss_0.detach().cpu().numpy()
                    ep_det_loss_l1 += local_det_loss_1.detach().cpu().numpy()
                    ep_det_loss_l2 += local_det_loss_2.detach().cpu().numpy()
                    ep_det_loss_l3 += local_det_loss_3.detach().cpu().numpy()
                    ep_nps_loss    += nps_loss.detach().cpu().numpy()
                    ep_nps_loss_g  += global_nps_loss.detach().cpu().numpy()
                    ep_nps_loss_l0 += local_nps_loss_0.detach().cpu().numpy()
                    ep_nps_loss_l1 += local_nps_loss_1.detach().cpu().numpy()
                    ep_nps_loss_l2 += local_nps_loss_2.detach().cpu().numpy()
                    ep_nps_loss_l3 += local_nps_loss_3.detach().cpu().numpy()
                    if(enable_loss_css):
                        if not(index_local <= -1): # local
                            ep_css_loss += css_loss.detach().cpu().numpy()
                    ep_tv_loss    += tv_loss.detach().cpu().numpy()
                    ep_tv_loss_g  += global_tv_loss.detach().cpu().numpy()
                    ep_tv_loss_l0 += local_tv_loss_0.detach().cpu().numpy()
                    ep_tv_loss_l1 += local_tv_loss_1.detach().cpu().numpy()
                    ep_tv_loss_l2 += local_tv_loss_2.detach().cpu().numpy()
                    ep_tv_loss_l3 += local_tv_loss_3.detach().cpu().numpy()
                    ep_loss       += loss

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    adv_patch_cpu.data.clamp_(0,1)       #keep patch in image range

                    bt1 = time.time()
                    # if i_batch%5 == 0:
                    #     iteration = self.epoch_length * epoch + i_batch

                    #     self.writer.add_scalar('total_loss', loss.detach().cpu().numpy(), iteration)
                    #     self.writer.add_scalar('loss/det_loss', det_loss.detach().cpu().numpy(), iteration)
                    #     self.writer.add_scalar('loss/nps_loss', nps_loss.detach().cpu().numpy(), iteration)
                    #     self.writer.add_scalar('loss/css_loss', css_loss.detach().cpu().numpy(), iteration)
                    #     self.writer.add_scalar('loss/tv_loss', tv_loss.detach().cpu().numpy(), iteration)
                    #     self.writer.add_scalar('misc/epoch', epoch, iteration)
                    #     self.writer.add_scalar('misc/learning_rate', optimizer.param_groups[0]["lr"], iteration)

                    #     self.writer.add_image('patch', adv_patch_cpu, iteration)

                    #     self.writer.flush()
                    if i_batch + 1 >= len(train_loader):
                        print('\n')
                    else:
                        # del adv_batch_t, output, max_prob, det_loss, p_img_batch, nps_loss, css_loss, tv_loss, loss
                        # del nps_loss, css_loss, tv_loss, loss
                        del nps_loss, tv_loss, loss
                        torch.cuda.empty_cache()
                    bt0 = time.time()

            et1 = time.time()
            ep_det_loss = ep_det_loss/len(train_loader)
            ep_det_loss_g = ep_det_loss_g/len(train_loader)
            ep_det_loss_l0 = ep_det_loss_l0/len(train_loader)
            ep_det_loss_l1 = ep_det_loss_l1/len(train_loader)
            ep_det_loss_l2 = ep_det_loss_l2/len(train_loader)
            ep_det_loss_l3 = ep_det_loss_l3/len(train_loader)
            ep_nps_loss = ep_nps_loss/len(train_loader)
            ep_nps_loss_g = ep_nps_loss_g/len(train_loader)
            ep_nps_loss_l0 = ep_nps_loss_l0/len(train_loader)
            ep_nps_loss_l1 = ep_nps_loss_l1/len(train_loader)
            ep_nps_loss_l2 = ep_nps_loss_l2/len(train_loader)
            ep_nps_loss_l3 = ep_nps_loss_l3/len(train_loader)
            ep_css_loss = ep_css_loss/len(train_loader)
            ep_tv_loss = ep_tv_loss/len(train_loader)
            ep_tv_loss_g = ep_tv_loss_g/len(train_loader)
            ep_tv_loss_l0 = ep_tv_loss_l0/len(train_loader)
            ep_tv_loss_l1 = ep_tv_loss_l1/len(train_loader)
            ep_tv_loss_l2 = ep_tv_loss_l2/len(train_loader)
            ep_tv_loss_l3 = ep_tv_loss_l3/len(train_loader)
            ep_loss = ep_loss/len(train_loader)

            iteration = epoch
            self.writer.add_scalar('total_loss', ep_loss.detach().cpu().numpy(), iteration)
            self.writer.add_scalar('loss/det_loss', ep_det_loss, iteration)
            self.writer.add_scalar('loss/det_loss_global', ep_det_loss_g, iteration)
            self.writer.add_scalar('loss/det_loss_local_0', ep_det_loss_l0, iteration)
            self.writer.add_scalar('loss/det_loss_local_1', ep_det_loss_l1, iteration)
            self.writer.add_scalar('loss/det_loss_local_2', ep_det_loss_l2, iteration)
            self.writer.add_scalar('loss/det_loss_local_3', ep_det_loss_l3, iteration)
            self.writer.add_scalar('loss/nps_loss', ep_nps_loss, iteration)
            self.writer.add_scalar('loss/nps_loss_g', ep_nps_loss_g, iteration)
            self.writer.add_scalar('loss/nps_loss_l0', ep_nps_loss_l0, iteration)
            self.writer.add_scalar('loss/nps_loss_l1', ep_nps_loss_l1, iteration)
            self.writer.add_scalar('loss/nps_loss_l2', ep_nps_loss_l2, iteration)
            self.writer.add_scalar('loss/nps_loss_l3', ep_nps_loss_l3, iteration)
            self.writer.add_scalar('loss/css_loss', ep_css_loss, iteration)
            self.writer.add_scalar('loss/tv_loss', ep_tv_loss, iteration)
            self.writer.add_scalar('loss/tv_loss_g', ep_tv_loss_g, iteration)
            self.writer.add_scalar('loss/tv_loss_l0', ep_tv_loss_l0, iteration)
            self.writer.add_scalar('loss/tv_loss_l1', ep_tv_loss_l1, iteration)
            self.writer.add_scalar('loss/tv_loss_l2', ep_tv_loss_l2, iteration)
            self.writer.add_scalar('loss/tv_loss_l3', ep_tv_loss_l3, iteration)
            self.writer.add_scalar('misc/epoch', epoch, iteration)
            self.writer.add_scalar('misc/learning_rate', optimizer.param_groups[0]["lr"], iteration)

            self.writer.add_image('patch', adv_patch_cpu, iteration)
            self.writer.add_image('g_patched', g_img_patched, iteration)
            # self.writer.add_image('l0_patched', l0_img_patched, iteration)
            self.writer.add_image('g_patch', g_img_patch, iteration)
            if(not cyclic_update_mode):
                self.writer.add_image('g_img_patched_1', g_img_patched_1, iteration)
                self.writer.add_image('g_img_patch_1', g_img_patch_1, iteration)
                self.writer.add_image('g_img_patched_2', g_img_patched_2, iteration)
                self.writer.add_image('g_img_patch_2', g_img_patch_2, iteration)
                self.writer.add_image('g_img_patched_3', g_img_patched_3, iteration)
                self.writer.add_image('g_img_patch_3', g_img_patch_3, iteration)
                self.writer.add_image('g_img_patched_4', g_img_patched_4, iteration)
                self.writer.add_image('g_img_patch_4', g_img_patch_4, iteration)
            # self.writer.add_image('l0_patch', l0_img_patch, iteration)


            #im = transforms.ToPILImage('RGB')(adv_patch_cpu)
            #plt.imshow(im)
            #plt.savefig(f'pics/{time_str}_{self.config.patch_name}_{epoch}.png')

            # # checkpoint
            # checkpoint = {
            #     'epoch': epoch,
            #     'ep_det_loss': ep_det_loss,
            #     'ep_nps_loss': ep_nps_loss,
            #     'ep_tv_loss': ep_tv_loss,
            #     'ep_loss': ep_loss,
            #     'valid_loss_min': ep_loss,
            #     'state_dict': self.darknet_model.state_dict(),
            #     'scheduler': scheduler.state_dict(),
            # }
            # # save checkpoint
            # save_ckp(checkpoint, False, str(self.config.checkpoint_path+"ckp_"+str(epoch)+".pt"), self.config.best_model_path)

            scheduler.step(ep_loss)
            if True:
                print('  EPOCH NR: ', epoch),
                print('EPOCH LOSS: ', ep_loss)
                print('  DET LOSS: ', ep_det_loss)
                print('  NPS LOSS: ', ep_nps_loss)
                print('  CSS LOSS: ', ep_css_loss)
                print('   TV LOSS: ', ep_tv_loss)
                print('EPOCH TIME: ', et1-et0)
                print('INDEX LOCAL:', index_local)
                if(epoch % save_epochs == 0):
                    adv_patch_cpu_ori = adv_patch_cpu.detach()
                    adv_patch_cpu_ = adv_patch_set.detach().cpu()
                    im_origin = transforms.ToPILImage('RGB')(adv_patch_cpu_ori)
                    im_noised = transforms.ToPILImage('RGB')(adv_patch_cpu_)
                    # plt.imshow(im)
                    # plt.show()
                    if not(os.path.exists("saved_patches/"+str(self.output_file_name))):
                        os.mkdir("saved_patches/"+str(self.output_file_name)) 
                    im_origin.save("saved_patches/"+str(self.output_file_name)+"/patch_"+str(epoch)+".jpg")
                    im_noised.save("saved_patches/"+str(self.output_file_name)+"/patched_"+str(epoch)+".jpg")
                    plt.imshow(im_origin)
                    plt.pause(0.001)  # in 0.001 sec
                    plt.ioff()  # After displaying, be sure to use plt.ioff() to close the interactive mode, otherwise there may be strange problems
                    plt.clf()  # Clear the image
                    plt.close()  # Clear the window
                if not(_flag_no_avaliable_size_obj):
                    del nps_loss, tv_loss, loss
                torch.cuda.empty_cache()
            et0 = time.time()
        self.writer.close()

    def generate_patch(self, type, dim=3):
        """
        Generate a random patch as a starting point for optimization.

        :param type: Can be 'gray' or 'random'. Whether or not generate a gray or a random patch.
        :return:
        """
        if type == 'gray':
            if(dim ==3):
                adv_patch_cpu = torch.full((3, self.config.patch_size, self.config.patch_size), 0.5)
            else:
                adv_patch_cpu = torch.full((1, self.config.patch_size, self.config.patch_size), 0.5)
        elif type == 'random':
            if(dim ==3):
                adv_patch_cpu = torch.rand((3, self.config.patch_size, self.config.patch_size))
            else:
                adv_patch_cpu = torch.rand((1, self.config.patch_size, self.config.patch_size))
        elif type == 'file':
            adv_patch_cpu = Image.open(self.config.init_patch).convert('RGB')
            tf = transforms.Resize((self.config.patch_size,self.config.patch_size))
            adv_patch_cpu = tf(adv_patch_cpu)
            tf = transforms.ToTensor()
            adv_patch_cpu = tf(adv_patch_cpu)

        return adv_patch_cpu

    def read_image(self, path, dim=3):
        """
        Read an input image to be used as a patch

        :param path: Path to the image to be read.
        :return: Returns the transformed patch as a pytorch Tensor.
        """
        patch_img = Image.open(path).convert('RGB')
        tf = transforms.Resize((self.config.patch_size, self.config.patch_size))
        patch_img = tf(patch_img)
        tf = transforms.ToTensor()

        adv_patch_cpu = tf(patch_img)
        if(dim ==3):
            return adv_patch_cpu ##  torch.Size([3, 300, 300])
        else:
            return adv_patch_cpu[0].unsqueeze(0) ##  torch.Size([1, 300, 300])


def main():
    if len(sys.argv) != 2:
        print('You need to supply (only) a configuration mode.')
        print('Possible modes are:')
        print(patch_config.patch_configs)


    trainer = PatchTrainer(sys.argv[1])
    trainer.train()

if __name__ == '__main__':
    main()


