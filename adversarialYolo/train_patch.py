"""
Training code for Adversarial patch training


"""

import PIL
import adversarialYolo.load_data
from tqdm import tqdm

from adversarialYolo.load_data import *
import gc
import matplotlib.pyplot as plt
from torch import autograd
from torchvision import transforms
from tensorboardX import SummaryWriter
import subprocess

import adversarialYolo.patch_config as patch_config
import sys
import time
import random
from enum import Enum

import os
import os.path
from adversarialYolo.utils import get_region_boxes, nms, do_detect, SwedishFlag_generator
from PIL import Image, ImageDraw
from adversarialYolo.NeuralStyleAlgorithm_api import get_style_model_and_losses

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
        self.nps_calculator_local  = NPSCalculator(patch_side=int(self.config.patch_size/2), 
                                                   printability_file_1=self.config.printfile
                                                   ).cuda()
        self.nps_calculator_global = NPSCalculator(patch_side=self.config.patch_size, 
                                                   printability_file_1=self.config.printfile
                                                   ).cuda()
        self.nps_calculator_global_countryflag = NPSCalculator(patch_side=self.config.patch_size, 
                                                   printability_file_1=self.config.printfile_blue,
                                                   printability_file_2=self.config.printfile_yellow
                                                   ).cuda()
        self.css_calculator = CSSCalculator(sample_img=self.read_image(self.config.sampleimgfile)).cuda()
        self.total_variation = TotalVariation().cuda()

        self.output_file_name = "init"
        self.writer = self.init_tensorboard(mode)
        
        self.cyclic_update_mode = False
        self.enable_loss_css    = False
        self.enable_countryflag = False
        self.enable_mask        = False
        self.n_epochs           = 1000      # Total epech
        self.save_epochs        = 1         # Save patch per "save_epochs"
        self.start_local_epoch  = 0         # 0: It's from global (-1), and then the others are with local patches. If it equal n_epochs, and the there is no local patch
        self.global_epoch_times = 1         # times of global epoch with local patch
        self.max_lab            = 14        # Inira: 14, COCO: 7
        self.global_patch_scale = 0.2       # size of global patch
        self.local_patch_scale  = 0.15      # size of local pacth
        self.local_index_mode   = 1         # 0: Sequential / 1: Random
        self.cyclic_update_step = 150       # local patch step. 300(patch size) / 150(step) = 2, and the total are 4(2x2) local patches
        self.p_nps     = 0.01               # coefficient nps loss. origin: 0.01   
        self.p_tv      = 2.5                # coefficient tv  loss. origin: 2.5   
        self.p_css     = 500                # coefficient css loss.
        self.p_style   = 100
        self.p_content = 0.05
        self.enable_rotation       = True
        self.enable_randomLocation = False
        self.enable_projection     = False
        self.enable_crease         = False
        self.enable_rectOccluding  = False
        self.enable_styleTransfer  = False
        self.enable_with_bbox      = False
        self.init_trainmode(4)              # origin: 0   

    def init_tensorboard(self, name=None):
        # subprocess.Popen(['tensorboard', '--logdir=runs'])
        if name is not None:
            # time_str = time.strftime("%Y%m%d-%H%M%S")
            time_str = time.strftime("%Y%m%d_45")
            # time_str = time.strftime("%Y%m%d_test")
            self.output_file_name = time_str + "_paper_obj"
            print("init_tensorboard / time: "+str(self.output_file_name))
            return SummaryWriter(f'runs/{time_str}_{name}')
        else:
            print("init_tensorboard ("+str(name)+")")
            return SummaryWriter()

    class trainmode(Enum):
        globalPatch        = 0  # origin
        fourDivisionsSimul = 1  # Simultaneously. Divide A into four equal parts 
        fourDivisionsSeq   = 2  # Sequentially.   Divide A into four equal parts 
        maskPatchSimul     = 3  # Simultaneously. Form a specific pattern through masks
        maskPatchSeq       = 4  # Sequentially.   Form a specific pattern through masks

    def init_trainmode(self, tmkey):
        tm = self.trainmode(tmkey)
        print("Trainmode : "+str(tm))
        if(tm == self.trainmode.globalPatch):
            self.cyclic_update_mode = True
            self.enable_mask        = False
            self.start_local_epoch  = self.n_epochs

        elif(tm == self.trainmode.fourDivisionsSimul):
            self.cyclic_update_mode = False
            self.enable_mask        = False
            self.cyclic_update_step = 150

        elif(tm == self.trainmode.fourDivisionsSeq):
            self.cyclic_update_mode = True
            self.enable_mask        = False
            self.cyclic_update_step = 150

        elif(tm == self.trainmode.maskPatchSimul):
            self.cyclic_update_mode = False
            self.enable_mask        = True
            self.start_local_epoch  = self.n_epochs

        elif(tm == self.trainmode.maskPatchSeq):
            self.cyclic_update_mode = True
            self.enable_mask        = True
            self.start_local_epoch  = self.n_epochs

        else:
            raise(NotImplementedError)

    def train(self):
        """
        Optimize a patch to generate an adversarial example.
        :return: Nothing
        """

        img_size = self.model_iheight
        batch_size = self.config.batch_size
        start_epoch = 1
        _local_patch_size = int(self.config.patch_size / 2)
        _num_local_side = int(int(int(self.config.patch_size - _local_patch_size) / self.cyclic_update_step) + 1)
        _num_local = _num_local_side * _num_local_side
        _flag_no_avaliable_size_obj = True
        
        time_str = time.strftime("%Y%m%d-%H%M%S")

        # Generate stating point
        adv_patch_cpu = self.generate_patch("gray", dim=3)  # torch.Size([3, 300, 300])
        mask=[]
        if(self.enable_mask):
            mask_cpu = self.generate_mask() # torch.Size([3, 300, 300])
            mask = mask_cpu.cuda()
        if(self.enable_styleTransfer):
            style_img_cpu = self.generate_style() # torch.Size([3, 300, 300])
            style_img = style_img_cpu.cuda()
            adv_patch_cpu = style_img_cpu
        
        # adv_patch_cpu = self.generate_patch("file", dim=3)  # torch.Size([3, 300, 300])
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

        train_loader = torch.utils.data.DataLoader(InriaDataset(self.config.img_dir, self.config.lab_dir, self.max_lab, img_size,
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
        ep_tv_loss_g = 1
        ep_tv_loss_l0 = 1
        ep_styleT_loss = 0
        ep_style_loss = 0
        ep_content_loss = 0
        local_det_loss_0 = torch.tensor(0).cuda()
        local_nps_loss_0 = torch.tensor(0).cuda()
        local_tv_loss_0  = torch.tensor(0).cuda()
        local_det_loss_1 = torch.tensor(0).cuda()
        local_det_loss_2 = torch.tensor(0).cuda()
        local_det_loss_3 = torch.tensor(0).cuda()
        global_det_loss = torch.tensor(0).cuda()
        global_nps_loss = torch.tensor(0).cuda()
        global_tv_loss  = torch.tensor(0).cuda()
        loss_style_content = torch.tensor(0).cuda()
        style_score        = torch.tensor(0).cuda()
        content_score      = torch.tensor(0).cuda()
        et0 = time.time()

        def get_patch_det_loss(adv_patch_input, img_batch_input, patch_scale=0.2, need_patch_set = False, isLocal=False, with_bbox=False):
            adv_batch_t, adv_patch_set, msk_batch = self.patch_transformer(adv_patch_input, 
                                                                            lab_batch, img_size,
                                                                            patch_mask=mask, 
                                                                            do_rotate=self.enable_rotation, 
                                                                            rand_loc=self.enable_randomLocation, 
                                                                            with_black_trans=True, 
                                                                            scale_rate = patch_scale, 
                                                                            with_crease=self.enable_crease, 
                                                                            with_projection=self.enable_projection,
                                                                            with_rectOccluding=self.enable_rectOccluding)
            # print("img_batch size: "+str(img_batch.size()))  ##  torch.Size([8, 3, 416, 416])
            # print("adv_batch_t size: "+str(adv_batch_t.size()))  ##  torch.Size([8, 14, 3, 416, 416])
            # print("adv_patch_set size: "+str(adv_patch_set.size()))  ##  torch.Size([3, 300, 300])

            ## get the part covered by patch
            # img_batch_ = img_batch_input[0, :, :, :,].detach().cpu()
            # img_plt = transforms.ToPILImage()(img_batch_)
            # img_plt.show()
            # time.sleep(5)
            img_batch_input_ = img_batch_input.unsqueeze(1)
            img_batch_input__ = img_batch_input_.expand(-1,msk_batch.size()[1],-1,-1,-1) 
            img_batch_covered = img_batch_input__* msk_batch
            # img_batch_covered_ = img_batch_covered[0, 0, :, :, :,].detach().cpu()
            # img_plt = transforms.ToPILImage()(img_batch_covered_)
            # img_plt.show()
            # time.sleep(5)
            ## get main colors
            # bf_palette, bf_counts = split_color(img_batch_covered, n_colors=5)

            p_img_batch = self.patch_applier(img_batch_input, adv_batch_t)
            #
            p_img_batch = F.interpolate(p_img_batch, (self.model_iheight, self.model_iwidth)) ##  torch.Size([8, 3, 416, 416])
            #
            output = self.darknet_model(p_img_batch)
            # img: output sample image for checking
            # sample first image
            img = p_img_batch[0, :, :,].detach().cpu()  # torch.Size([3, 416, 416])
            #
            if(with_bbox):
                trans_2pilimage = transforms.ToPILImage()
                img_pil = trans_2pilimage(img)
                # sample first image
                boxes = get_region_boxes(output[0], conf_thresh=0.4, num_classes=self.darknet_model.num_classes, anchors=self.darknet_model.anchors, num_anchors=self.darknet_model.num_anchors)[0]
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
            # img_plt = transforms.ToPILImage()(img)
            # img_plt.show()
            ## btach mask: get batch location of image
            img_b = adv_batch_t[0, 0, :, :,].detach().cpu()
            # img_plt = transforms.ToPILImage()(img_b)
            # img_plt.show()
            # time.sleep(100)
            max_prob = self.prob_extractor(output)
            det_loss = torch.mean(max_prob)  ## config loss function.
            if not(need_patch_set):
                del output, p_img_batch, max_prob, adv_patch_set, 
                return det_loss, torch.zeros(1), img, img_b, adv_batch_t, img_batch_covered
            else:
                del output, p_img_batch, max_prob
                return det_loss, adv_patch_set, img, img_b, adv_batch_t, img_batch_covered

        """
        train epoch
        """
        for epoch in range(start_epoch, self.n_epochs+1):
            ep_det_loss = 0
            ep_nps_loss = 0
            ep_css_loss = 0
            ep_tv_loss = 0
            ep_loss = 0
            
            bt0 = time.time()
            
            """
            index of local patch:  
            <=-1 are global. 
            >= 0 are local.
            """
            if(epoch > self.start_local_epoch):
                if(self.local_index_mode == 0):
                    # Sequential
                    index_local = epoch % _num_local
                elif(self.local_index_mode == 1):
                    # Random
                    if(len(index_list_local) == 0):
                        if(index_local == -self.global_epoch_times):
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
                if(self.enable_countryflag):
                    if not(index_local == -1):
                        index_local = -1
                    else:
                        index_local = -2
                else:
                    index_local = -1

            """
            train batch
            """
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
                    

                    # patch_unit = int(self.config.patch_size / 2)
                    # adv_patch_01 = torch.narrow(torch.narrow(adv_patch, 1, 0, patch_unit), 2, 0, patch_unit)
                    # adv_patch_02 = torch.narrow(torch.narrow(adv_patch, 1, 0, patch_unit), 2, patch_unit, patch_unit)
                    # adv_patch_03 = torch.narrow(torch.narrow(adv_patch, 1, patch_unit, patch_unit), 2, 0, patch_unit)
                    # adv_patch_04 = torch.narrow(torch.narrow(adv_patch, 1, patch_unit, patch_unit), 2, patch_unit, patch_unit)
                    adv_patch_local_set = []
                    patch_unit = int(_local_patch_size)
                    for x_unit in range(0,(int(self.config.patch_size - _local_patch_size)+1),self.cyclic_update_step):
                        _s_xp = int(x_unit)
                        for y_unit in range(0,(int(self.config.patch_size - _local_patch_size)+1),self.cyclic_update_step):
                            _s_yp = int(y_unit)
                            adv_patch_part = torch.narrow(torch.narrow(adv_patch, 1, _s_xp, patch_unit), 2, _s_yp, patch_unit)
                            adv_patch_local_set.append(adv_patch_part)
                    adv_patch_input = adv_patch_local_set[-1]

                    if(self.enable_countryflag):
                        adv_patch_SwedishFlag_outer, adv_patch_SwedishFlag_inner = SwedishFlag_generator(adv_patch)

                    # output_clear = self.darknet_model(img_batch)
                    # # print("output_clear size : "+str(output_clear.size()))
                    # boxes_0 = get_region_boxes(output_clear[0], conf_thresh=0.4, num_classes=self.darknet_model.num_classes, anchors=self.darknet_model.anchors, num_anchors=self.darknet_model.num_anchors)[0]
                    # boxes_1 = nms(boxes_0, 0.4)

                    """
                    det_loss
                    """
                    adv_patch_input = adv_patch
                    if(self.cyclic_update_mode):
                        patch_scale = self.local_patch_scale
                        if(index_local <= -1):
                            if(self.enable_countryflag):
                                if(index_local == -1):
                                    adv_patch_input = adv_patch_SwedishFlag_outer
                                else:
                                    adv_patch_input = adv_patch_SwedishFlag_inner
                            else:
                                adv_patch_input = adv_patch
                            patch_scale = self.global_patch_scale
                        else:
                            adv_patch_input = adv_patch_local_set[index_local]
                        det_loss, adv_patch_set, g_img_patched, g_img_patch, adv_batch_t, img_batch_covered = get_patch_det_loss(adv_patch_input, img_batch, patch_scale, need_patch_set = True, with_bbox=self.enable_with_bbox)
                    else:
                        if(epoch > self.start_local_epoch):
                            patch_scale = self.local_patch_scale
                            local_det_loss_0, adv_patch_set_1, g_img_patched_1, g_img_patch_1,_,_ = get_patch_det_loss(adv_patch_local_set[0], img_batch, patch_scale, need_patch_set = True, isLocal=True)
                            local_det_loss_1, adv_patch_set_2, g_img_patched_2, g_img_patch_2,_,_ = get_patch_det_loss(adv_patch_local_set[1], img_batch, patch_scale, need_patch_set = True, isLocal=True)
                            local_det_loss_2, adv_patch_set_3, g_img_patched_3, g_img_patch_3,_,_ = get_patch_det_loss(adv_patch_local_set[2], img_batch, patch_scale, need_patch_set = True, isLocal=True)
                            local_det_loss_3, adv_patch_set_4, g_img_patched_4, g_img_patch_4,_,_ = get_patch_det_loss(adv_patch_local_set[3], img_batch, patch_scale, need_patch_set = True, isLocal=True)
                            patch_scale = self.global_patch_scale
                            det_loss_g, adv_patch_set, g_img_patched, g_img_patch, adv_batch_t, img_batch_covered = get_patch_det_loss(adv_patch, img_batch, patch_scale, need_patch_set = True)
                            det_loss = (local_det_loss_0 + local_det_loss_1 + local_det_loss_2 + local_det_loss_3 + det_loss_g) / 5
                        else:
                            patch_scale = self.global_patch_scale
                            det_loss_g, adv_patch_set, g_img_patched, g_img_patch, adv_batch_t, img_batch_covered = get_patch_det_loss(adv_patch, img_batch, patch_scale, need_patch_set = True)
                            det_loss = det_loss_g

                    """
                    css_loss
                    """
                    if(self.enable_loss_css):
                        # ### let the local be like as the global
                        # if(self.cyclic_update_mode):
                        #     if not(index_local <= -1): # local
                        #         if(len(_previous_adv_patch_local_set)>0):
                        #             _previous_part_patch = _previous_adv_patch_local_set[index_local]
                        #             css = self.css_calculator(adv_patch_input, _previous_part_patch)
                        #             _previous_adv_patch_local_set = adv_patch_local_set
                        #     else:
                        #         if(len(_previous_adv_patch_local_set)==0):
                        #             # init
                        #             _previous_adv_patch_local_set = adv_patch_local_set
                        # else:
                        #     if(epoch >= self.start_local_epoch):
                        #         if(len(_previous_adv_patch_local_set)>0):
                        #             _previous_adv_patch_01 = _previous_adv_patch_local_set[0]
                        #             _previous_adv_patch_02 = _previous_adv_patch_local_set[1]
                        #             _previous_adv_patch_03 = _previous_adv_patch_local_set[2]
                        #             _previous_adv_patch_04 = _previous_adv_patch_local_set[3]
                        #             css_1 = self.css_calculator(adv_patch_local_set[0], _previous_adv_patch_01)
                        #             css_2 = self.css_calculator(adv_patch_local_set[1], _previous_adv_patch_02)
                        #             css_3 = self.css_calculator(adv_patch_local_set[2], _previous_adv_patch_03)
                        #             css_4 = self.css_calculator(adv_patch_local_set[3], _previous_adv_patch_04)
                        #             css = (css_1 + css_2 + css_3 + css_4) / 4
                        #             _previous_adv_patch_local_set = adv_patch_local_set
                        #         else:
                        #             if(len(_previous_adv_patch_local_set)==0):
                        #                 # init
                        #                 _previous_adv_patch_local_set = adv_patch_local_set
                        ### let the global or the local be like as the part covered by patch
                        css = self.css_calculator(adv_batch_t, img_batch_covered)

                    """
                    nps_loss
                    """
                    if(self.cyclic_update_mode):
                        if(index_local <= -1):
                            if(self.enable_countryflag):
                                if(index_local == -1):
                                    nps = self.nps_calculator_global_countryflag(adv_patch_input, key=1)
                                else:
                                    nps = self.nps_calculator_global_countryflag(adv_patch_input, key=2)
                            else:
                                nps = self.nps_calculator_global(adv_patch_input, key=1)
                        else:
                            nps = self.nps_calculator_local(adv_patch_input)
                    else:
                        if(epoch > self.start_local_epoch):
                            nps_1 = self.nps_calculator_local(adv_patch_local_set[0])
                            nps_2 = self.nps_calculator_local(adv_patch_local_set[1])
                            nps_3 = self.nps_calculator_local(adv_patch_local_set[2])
                            nps_4 = self.nps_calculator_local(adv_patch_local_set[3])
                            nps_g = self.nps_calculator_global(adv_patch)
                            nps = (nps_1 + nps_2 + nps_3 + nps_4 + nps_g) / 5
                        else:
                            nps_g = self.nps_calculator_global(adv_patch)
                            nps = nps_g

                    """
                    tv_loss
                    """
                    if(self.cyclic_update_mode):
                        tv = self.total_variation(adv_patch_input)
                    else:
                        if(epoch > self.start_local_epoch):
                            tv_1 = self.total_variation(adv_patch_local_set[0])
                            tv_2 = self.total_variation(adv_patch_local_set[1])
                            tv_3 = self.total_variation(adv_patch_local_set[2])
                            tv_4 = self.total_variation(adv_patch_local_set[3])
                            tv_g = self.total_variation(adv_patch)
                            tv = (tv_1 + tv_2 + tv_3 + tv_4 + tv_g) / 5
                        else:
                            tv_g = self.total_variation(adv_patch)
                            tv = tv_g
                    # print("tv size: "+str(tv.size()))  ##  torch.Size([])
                    # print("tv     : "+str(tv.data))


                    nps_loss = nps*self.p_nps
                    # ### let the local be like as the global
                    # if(self.enable_loss_css):
                    #     if not(index_local <= -1): # local
                    #         css_loss = css*self.p_css
                    ### let the global or the local be like as the part covered by patch 
                    if(self.enable_loss_css):
                        css_loss = css*self.p_css
                    tv_loss = tv*self.p_tv

                    if(self.cyclic_update_mode):
                        if(index_local <= -1):
                            global_det_loss = det_loss
                            global_nps_loss = nps_loss
                            global_tv_loss  = tv_loss
                        else:
                            local_det_loss_0 = det_loss
                            local_nps_loss_0 = nps_loss
                            local_tv_loss_0  = tv_loss
                    else:
                        global_det_loss = det_loss
                        global_nps_loss = nps_loss
                        global_tv_loss  = tv_loss
                    # det_loss = global_det_loss * 0.1 + local_det_loss_0 * 0.9
                    if(self.enable_loss_css):
                        # ### let the local be like as the global
                        # if not(index_local <= -1): # local
                        #     loss = det_loss + nps_loss + css_loss + torch.max(tv_loss, torch.tensor(0.1).cuda())
                        # else:
                        #     loss = det_loss + nps_loss + torch.max(tv_loss, torch.tensor(0.1).cuda())
                        ### let the global or the local be like as the part covered by patch
                        loss = det_loss + nps_loss + css_loss + torch.max(tv_loss, torch.tensor(0.1).cuda())
                    else:
                        loss = det_loss + nps_loss + torch.max(tv_loss, torch.tensor(0.1).cuda())
                    # loss = det_loss + css_loss + torch.max(tv_loss, torch.tensor(0.1).cuda())

                    if(self.enable_styleTransfer):
                        style_img = style_img
                        content_img = adv_patch_input
                        imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu
                        if(style_img.size()[-1] != imsize):
                            style_img   = F.interpolate(style_img.unsqueeze(0), size=imsize)
                        if(content_img.size()[-1] != imsize):
                            content_img = F.interpolate(content_img.unsqueeze(0), size=imsize)
                        model, style_losses, content_losses = get_style_model_and_losses(style_img, 
                                                                                        content_img)
                        model(content_img)
                        #
                        style_score   = 0
                        content_score = 0
                        for sl in style_losses:
                            style_score += sl.loss
                        for cl in content_losses:
                            content_score += cl.loss
                        #
                        style_score *= self.p_style
                        content_score *= self.p_content
                        #
                        loss_style_content = style_score + content_score
                        #
                        loss = loss + loss_style_content

                    ep_det_loss     += det_loss.detach().cpu().numpy()
                    ep_det_loss_g   += global_det_loss.detach().cpu().numpy()
                    ep_det_loss_l0  += local_det_loss_0.detach().cpu().numpy()
                    ep_det_loss_l1  += local_det_loss_1.detach().cpu().numpy()
                    ep_det_loss_l2  += local_det_loss_2.detach().cpu().numpy()
                    ep_det_loss_l3  += local_det_loss_3.detach().cpu().numpy()
                    ep_nps_loss     += nps_loss.detach().cpu().numpy()
                    ep_nps_loss_g   += global_nps_loss.detach().cpu().numpy()
                    ep_nps_loss_l0  += local_nps_loss_0.detach().cpu().numpy()
                    ep_styleT_loss  += loss_style_content.detach().cpu().numpy()
                    ep_style_loss   += style_score.detach().cpu().numpy()
                    ep_content_loss += content_score.detach().cpu().numpy()
                    if(self.enable_loss_css):
                        # # ### let the local be like as the global
                        # if not(index_local <= -1): # local
                        #     ep_css_loss += css_loss.detach().cpu().numpy()
                        ### let the global or the local be like as the part covered by patch
                        ep_css_loss += css_loss.detach().cpu().numpy()
                    ep_tv_loss    += tv_loss.detach().cpu().numpy()
                    ep_tv_loss_g  += global_tv_loss.detach().cpu().numpy()
                    ep_tv_loss_l0 += local_tv_loss_0.detach().cpu().numpy()
                    ep_loss       += loss

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    if(self.enable_mask):
                        adv_patch_cpu.data = adv_patch_cpu.data * mask_cpu.data   # mask
                    adv_patch_cpu.data.clamp_(0,1)                            #keep patch in image range

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
                        if(self.enable_loss_css):
                            del css_loss
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
            ep_css_loss = ep_css_loss/len(train_loader)
            ep_tv_loss = ep_tv_loss/len(train_loader)
            ep_tv_loss_g = ep_tv_loss_g/len(train_loader)
            ep_tv_loss_l0 = ep_tv_loss_l0/len(train_loader)
            ep_styleT_loss = ep_styleT_loss/len(train_loader)
            ep_style_loss = ep_style_loss/len(train_loader)
            ep_content_loss = ep_content_loss/len(train_loader)
            ep_loss = ep_loss/len(train_loader)

            iteration = epoch
            self.writer.add_scalar('total_loss', ep_loss.detach().cpu().numpy(), iteration)
            self.writer.add_scalar('loss/det_loss', ep_det_loss, iteration)
            self.writer.add_scalar('loss/det_loss_global', ep_det_loss_g, iteration)
            self.writer.add_scalar('loss/det_loss_local_0', ep_det_loss_l0, iteration)
            self.writer.add_scalar('loss/det_loss_local_1', ep_det_loss_l1, iteration)
            self.writer.add_scalar('loss/det_loss_local_2', ep_det_loss_l2, iteration)
            self.writer.add_scalar('loss/det_loss_local_3', ep_det_loss_l3, iteration)
            self.writer.add_scalar('loss/ep_styleT_loss', ep_styleT_loss, iteration)
            self.writer.add_scalar('loss/ep_style_loss', ep_style_loss, iteration)
            self.writer.add_scalar('loss/ep_content_loss', ep_content_loss, iteration)
            self.writer.add_scalar('loss/nps_loss', ep_nps_loss, iteration)
            self.writer.add_scalar('loss/nps_loss_g', ep_nps_loss_g, iteration)
            self.writer.add_scalar('loss/nps_loss_l0', ep_nps_loss_l0, iteration)
            self.writer.add_scalar('loss/css_loss', ep_css_loss, iteration)
            self.writer.add_scalar('loss/tv_loss', ep_tv_loss, iteration)
            self.writer.add_scalar('loss/tv_loss_g', ep_tv_loss_g, iteration)
            self.writer.add_scalar('loss/tv_loss_l0', ep_tv_loss_l0, iteration)
            self.writer.add_scalar('misc/epoch', epoch, iteration)
            self.writer.add_scalar('misc/learning_rate', optimizer.param_groups[0]["lr"], iteration)

            self.writer.add_image('patch', adv_patch_cpu, iteration)
            self.writer.add_image('g_patched', g_img_patched, iteration)
            # self.writer.add_image('l0_patched', l0_img_patched, iteration)
            self.writer.add_image('g_patch', g_img_patch, iteration)
            if(not self.cyclic_update_mode):
                if(epoch > self.start_local_epoch):
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
                print('    EPOCH NR: ', epoch),
                print('  EPOCH LOSS: ', ep_loss)
                print('    DET LOSS: ', ep_det_loss)
                print('    NPS LOSS: ', ep_nps_loss)
                print('    CSS LOSS: ', ep_css_loss)
                print('     TV LOSS: ', ep_tv_loss)
                print('     ST LOSS: ', ep_styleT_loss)
                print('  STYLE LOSS: ', ep_style_loss)
                print('CONTENT LOSS: ', ep_content_loss)
                print('  EPOCH TIME: ', et1-et0)
                print(' INDEX LOCAL:', index_local)
                if(epoch % self.save_epochs == 0):
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
        elif type == 'yellow':
            if(dim ==3):
                adv_patch_cpu = torch.full((3, self.config.patch_size, self.config.patch_size), 1.0)
                adv_patch_cpu[2,:,:] = 0
            else:
                adv_patch_cpu = torch.full((1, self.config.patch_size, self.config.patch_size), 0.5)
        elif type == 'black':
            if(dim ==3):
                adv_patch_cpu = torch.full((3, self.config.patch_size, self.config.patch_size), 0.0)
            else:
                adv_patch_cpu = torch.full((1, self.config.patch_size, self.config.patch_size), 0.0)
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
    
    def generate_mask(self, dim=3):
        mask_cpu = Image.open(self.config.mask01).convert('RGB')
        tf = transforms.Resize((self.config.patch_size,self.config.patch_size))
        mask_cpu = tf(mask_cpu)
        tf = transforms.ToTensor()
        mask_cpu = tf(mask_cpu)
        return mask_cpu

    def generate_style(self, dim=3):
        img_cpu = Image.open(self.config.style).convert('RGB')
        tf = transforms.Resize((self.config.patch_size,self.config.patch_size))
        img_cpu = tf(img_cpu)
        tf = transforms.ToTensor()
        img_cpu = tf(img_cpu)
        return img_cpu

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


