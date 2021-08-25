import sys
import time
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageDraw
from utils import *
from darknet import *
from load_data import PatchTransformer, PatchApplier, InriaDataset
import json
import matplotlib.pyplot as plt
import shutil


if __name__ == '__main__':
    print("Setting everything up")
    cfgfile = "cfg/yolo.cfg"
    weightfile = "weights/yolo.weights"
    patchfile = "patches/object_score.png"
    # patchfile = "saved_patches/patch11.jpg"
    # patchfile = "saved_patches/original/v2/patch_500.jpg"
    # patchfile = "saved_patches/20200722_12_g1_5_l1_5_paper_obj/patch_1000.jpg"
    # patchfile = "saved_patches/20200820_22_1g_4l_paper_obj/patch_450.jpg"
    # patchfile = "saved_patches/20201027_44_paper_obj/patch_1000.jpg"
    # patchfile = "saved_patches/20201027_test_paper_obj/patch_1000.jpg"
    # patchfile = "/home/wvr/Pictures/individualImage_upper_body.png"
    #patchfile = "/home/wvr/Pictures/class_only.png"
    #patchfile = "/home/wvr/Pictures/class_transfer.png"
    savedir = "testing/"

    """
    output:
    imgs                   : Images with overlapping ground-truth files (and filtered by roi_rate and marked with bbox)
    yolo-labels            : The prediction  label in yolo format
    yolo-labels_rescale    : The prediction  label in yolo format with coordinates from [0,1] to [0,(w,h)]
    yolo-labels_gt_rescale : The groundtruth label in yolo format with coordinates from [0,1] to [0,(w,h)]
    """

    ## ---------------------------------------------- ##
    data_mode = "test"                       # mode: test, train, or create
    local_index = -1                         # 0~3 : part patch, -1 : all patch
    patch_scale = 0.2
    enable_roi_filter = False                # Filter images where roi_rate is less than roi_thres.
    roi_thres = 0.4
    enable_numfeature_filter = True          # only for prediction
    max_num_feature = 14                     # max number of objects in one image
    feature_id = 0                           # person
    enable_save_person_part = False          # save the person part of bbox as new image
    person_part_thre = 0.0625                # area > thre (maybe 1/9, 0.111), them save them
    person_part_rescale=[0.8,                # Scale the independent person_part to these dimensions 
                        0.7,
                        0.6,
                        0.5,
                        0.4,
                        0.3,
                        0.2]

    enable_proper_part = True                # True: save the images with patch
    enable_random_part = False

    enable_show_roirate_distribution = True  # for the clear, the proper and the random
    enable_show_predection_bbox = True       # for the clear, the proper and the random
    save_clean_imgs = True                   # True: save the images without patch

    # imgdir    : Image location to be recognized
    # labeldir  : Groudtruth label location. (None : No output 'yolo-labels_gt_rescale', 
    #                                                No cpoy images fittered by roi_rate
    #                                                No 'roirate_distribution'
    if(data_mode.lower() == "test"):
        imgdir = "inria/Test/pos"
        # imgdir = "inria/Test/pos_020"
        labeldir = "inria/Test/pos/labels/"
    elif(data_mode.lower() == "train"):
        imgdir = "inria/Train/pos"
        # # imgdir = "inria/Train/pos_020"
        labeldir = "inria/Train/pos/labels/"
        # """ person part: no original labels """
        # imgdir = "inria/Train/pos_person_00625"
        # labeldir = None
    elif(data_mode.lower() == "create"):
        imgdir =   "../convert2Yolo-master/COCO/coco_val2017"
        labeldir = "../convert2Yolo-master/COCO/coco_val2017/labels/"
        # labeldir = None
    ## ---------------------------------------------- ##

    darknet_model = Darknet(cfgfile)
    darknet_model.load_weights(weightfile)
    darknet_model = darknet_model.eval().cuda()
    patch_applier = PatchApplier().cuda()
    patch_transformer = PatchTransformer().cuda()

    
    img_size = darknet_model.height
    _confidence = 0.5
    _image_filtered = False
    patch_size = 300

    patch_img = Image.open(patchfile).convert('RGB')
    tf = transforms.Resize((patch_size,patch_size))
    patch_img = tf(patch_img)
    tf = transforms.ToTensor()
    adv_patch_cpu = tf(patch_img)
    adv_patch = adv_patch_cpu.cuda()

    clean_results = []
    noise_results = []
    patch_results = []

    def bb_intersection_over_union(boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou

    # Create folders
    _FILES_PATH_list = [os.path.abspath(os.path.join(savedir, 'skip/')),
                        os.path.abspath(os.path.join(savedir, 'create/')),
                        os.path.abspath(os.path.join(savedir, 'clean/', 'imgs/')),
                        os.path.abspath(os.path.join(savedir, 'clean/', 'imgs_person/')),
                        os.path.abspath(os.path.join(savedir, 'clean/', 'yolo-labels/')),
                        os.path.abspath(os.path.join(savedir, 'clean/', 'yolo-labels_rescale/')),
                        os.path.abspath(os.path.join(savedir, 'clean/', 'yolo-labels_gt_rescale/')),
                        os.path.abspath(os.path.join(savedir, 'clean/', 'imgs_person_yolo-labels/')),
                        os.path.abspath(os.path.join(savedir, 'clean/', 'imgs_person-labels_gt/')),
                        os.path.abspath(os.path.join(savedir, 'clean/', 'imgs_person-labels_gt_rescale/')),
                        os.path.abspath(os.path.join(savedir, 'proper_patched/', 'imgs/')),
                        os.path.abspath(os.path.join(savedir, 'proper_patched/', 'yolo-labels/')),
                        os.path.abspath(os.path.join(savedir, 'proper_patched/', 'yolo-labels_rescale/')),
                        os.path.abspath(os.path.join(savedir, 'proper_patched/', 'yolo-labels_gt_rescale/')),
                        os.path.abspath(os.path.join(savedir, 'random_patched/', 'imgs/')),
                        os.path.abspath(os.path.join(savedir, 'random_patched/', 'yolo-labels/')),
                        os.path.abspath(os.path.join(savedir, 'random_patched/', 'yolo-labels_rescale/')),
                        os.path.abspath(os.path.join(savedir, 'random_patched/', 'yolo-labels_gt_rescale/'))
                        ]
    for _FILES_PATH in _FILES_PATH_list:
        if not os.path.exists(_FILES_PATH): # if it doesn't exist already
            os.makedirs(_FILES_PATH)
    _IMG_FILTTED_FILES_PATH_list = [os.path.abspath(os.path.join(savedir, 'test/')),
                                    os.path.abspath(os.path.join(savedir, 'train/'))]
    for _IMG_FILTTED_FILES_PATH in _IMG_FILTTED_FILES_PATH_list:
        if not os.path.exists(_IMG_FILTTED_FILES_PATH): # if it doesn't exist already
            os.makedirs(_IMG_FILTTED_FILES_PATH)
        else:
            shutil.rmtree(_IMG_FILTTED_FILES_PATH) 
            os.makedirs(_IMG_FILTTED_FILES_PATH)
    
    print("Done")
    #Loop over cleane beelden (Walk over clean images)
    bbox_rates = []
    bbox_rates_per_img = []
    num_img = 1
    _skip_num_img = 0
    _total_num_img = len(os.listdir(imgdir))
    img_name = ""
    label = np.ones([5])
    for imgfile in os.listdir(imgdir):
        _skip_img = False # Skip imgs without lable
        print("new image : "+str(num_img) +"/"+str(_total_num_img) + "     skip : "+str(_skip_num_img), end ="\r")
        num_img = num_img + 1
        if imgfile.endswith('.jpg'):
            name = os.path.splitext(imgfile)[0]
            img_name = name + '.jpg'
        elif imgfile.endswith('.png'):
            name = os.path.splitext(imgfile)[0]
            img_name = name + '.png'
        if imgfile.endswith('.jpg') or imgfile.endswith('.png'):
            name = os.path.splitext(imgfile)[0]    #image name w/o extension
            #
            txtname = name + '.txt'
            txtpath = os.path.abspath(os.path.join(savedir, 'clean/', 'yolo-labels/', txtname))
            txtpath_rescale = os.path.abspath(os.path.join(savedir, 'clean/', 'yolo-labels_rescale/', txtname))

            if not(labeldir == None):
                # Rescale lable to match the format of map calculation.
                # And, fiter image which over threshold, bbox/image_aera.

                #
                labelname = name + '.txt'
                labelpath = os.path.abspath(os.path.join(labeldir, labelname))
                labelpath_rescale = os.path.abspath(os.path.join(savedir, 'clean/', 'yolo-labels_gt_rescale/', txtname))
                #
                try:
                    labelfile = open(labelpath,'r')
                except:
                    _skip_img = True
                    _skip_num_img = _skip_num_img + 1
                    skipname = txtname
                    skippath = os.path.abspath(os.path.join(savedir, 'skip/', skipname))
                    skipfile = open(skippath,'w+')
                    skipfile.write(f'{"No path: "+str(labelpath)}\n')
                    skipfile.close()
                    continue
                bbox_labels = [] # per file
                # print("labelname : "+str(labelname))
                for lfc in labelfile:
                    # print("lfc               : "+str(lfc))
                    # print("lfc.split(" ")[0] : "+str(lfc.split(" ")[0]))
                    if(lfc.split(" ")[0] == "None"):
                        # skip the None object
                        continue
                    bbox_label = [round(float(item), 5) for item in lfc.split(" ")] 
                    # print("bbox_label : "+str(bbox_label))
                    bbox_labels.append(bbox_label)
                # #
                # if(enable_numfeature_filter):
                #     _count_feature = 0
                #     for bbox_label in bbox_labels:
                #         cls_id  = bbox_label[0]
                #         if(cls_id == feature_id):
                #             _count_feature = _count_feature + 1
                #     if(_count_feature>max_num_feature):
                #         _skip_img = True
                #         _skip_num_img = _skip_num_img + 1
                #         print("\n too many feature (ground_truth) : "+str(txtname))
                #         continue

            def save_image_person_part(padded_img_input, left, top, right, bottom, new_scale, index_bbox, cls_id, with_save_gt=False, with_save_yoloLables=False, with_no_detected_filtter=False):
                new_scale_codename = str(new_scale)[2:]
                resize = transforms.Resize((img_size,img_size))
                padded_img_input = resize(padded_img_input)
                padded_img_input_person_ = padded_img_input.crop((left, top, right, bottom))
                w,h = padded_img_input_person_.size
                if(w > h):
                    max_side = w
                    target_max_side = int(padded_img_input.size[0]*new_scale)
                    im_rate = float(target_max_side / max_side)
                    newsize = (target_max_side, int(im_rate*h))
                    padded_img_input_person_= padded_img_input_person_.resize(newsize)
                    w, h = padded_img_input_person_.size
                    padding_y = (padded_img_input.size[0]  - h) / 2
                    padding_x = (padded_img_input.size[0] - target_max_side) / 2
                else:
                    max_side = h
                    target_max_side = int(padded_img_input.size[0]*new_scale)
                    im_rate = float(target_max_side / max_side)
                    newsize = (int(im_rate*w) , target_max_side)
                    padded_img_input_person_ = padded_img_input_person_.resize(newsize)
                    w, h = padded_img_input_person_.size
                    padding_x = (padded_img_input.size[0]  - w) / 2
                    padding_y = (padded_img_input.size[0] - target_max_side) / 2  
                # gt rescale
                w_rescale = w
                h_rescale = h
                x_center_rescale = padding_x + int(w_rescale/2)
                y_center_rescale = padding_y + int(h_rescale/2)
                left_rescale   = padding_x
                top_rescale    = padding_y
                right_rescale  = left_rescale + w_rescale
                bottom_rescale = top_rescale  + h_rescale
                # gt
                w_normalized = float(w_rescale / padded_img_input.size[0])
                h_normalized = float(h_rescale / padded_img_input.size[0])
                x_center_normalized = float(x_center_rescale / padded_img_input.size[0])
                y_center_normalized = float(y_center_rescale / padded_img_input.size[0])
                # create new image woth person_part
                padded_img_input_person = Image.new('RGB', padded_img_input.size, color=(127,127,127))
                padded_img_input_person.paste(padded_img_input_person_, (int(padding_x), int(padding_y)))
                # init
                x_center_detected    = 0
                y_center_detected    = 0
                width_detected       = 0
                height_detected      = 0
                det_score_detected   = 0
                c_cla_score_detected = 0
                # do detect with new image
                boxes = do_detect(darknet_model, padded_img_input_person, 0.4, 0.4, True)
                # get max bbox one
                max_aera             = 0
                for box in boxes:
                    cls_id = box[6]
                    if(cls_id == 0):   #if person
                        det_score   = box[4]
                        c_cla_score = box[5]
                        cla_score   = det_score * c_cla_score
                        if(cla_score > _confidence): # detection confidence
                            x_center    = box[0]
                            y_center    = box[1]
                            width       = box[2]
                            height      = box[3]
                            aera        = width * height
                            det_score_detected   = det_score
                            c_cla_score_detected = c_cla_score
                            if(aera > max_aera):
                                max_aera    = aera
                                x_center_detected = x_center
                                y_center_detected = y_center
                                width_detected    = width
                                height_detected   = height
                if(enable_show_predection_bbox):
                    left    = (x_center_detected - width_detected  / 2) * padded_img_input_person.size[0]
                    right   = (x_center_detected + width_detected  / 2) * padded_img_input_person.size[0]
                    top     = (y_center_detected - height_detected / 2) * padded_img_input_person.size[0]
                    bottom  = (y_center_detected + height_detected / 2) * padded_img_input_person.size[0]
                    # img with prediction
                    draw = ImageDraw.Draw(padded_img_input_person)
                    shape = [left,
                                top, 
                                right,
                                bottom]
                    draw.rectangle(shape, outline ="green")
                    # text
                    color = [0,255,0]
                    sentence = "person\n(" + str(round(float(det_score_detected), 2)) + ", " + str(round(float(c_cla_score_detected), 2)) + ")"
                    position = [left,
                                top]
                    draw.text(tuple(position), sentence, tuple(color))
                    # img with groudtruth
                    draw = ImageDraw.Draw(padded_img_input_person)
                    shape = [left_rescale,
                                top_rescale, 
                                right_rescale,
                                bottom_rescale]
                    draw.rectangle(shape, outline ="white")
                # check whether to save them
                _save_data = True
                if(with_no_detected_filtter):
                    if((width_detected*height_detected)<=0):
                        _save_data = False
                if(_save_data):
                    # save images
                    cleanname_person = name+"_"+str(index_bbox)+"_"+str(new_scale_codename) +".png"
                    padded_img_input_person.save(os.path.join(savedir, 'clean/imgs_person/', cleanname_person))
                    # save gt-labels and yolo-labels
                    if(with_save_gt):
                        txtname = name+"_"+str(index_bbox)+"_"+str(new_scale_codename) + '.txt'
                        # gt rescale
                        path_person_part_gt_rescale = os.path.abspath(os.path.join(savedir, 'clean/', 'imgs_person-labels_gt_rescale/', txtname))
                        labelfile_person_part_rescale = open(path_person_part_gt_rescale,'w+') #read label
                        labelfile_person_part_rescale.write("person" + str(f' {left_rescale} {top_rescale} {right_rescale} {bottom_rescale}\n'))  # left, top, right, bottom
                        labelfile_person_part_rescale.close()
                        # gt
                        path_person_part_gt = os.path.abspath(os.path.join(savedir, 'clean/', 'imgs_person-labels_gt/', txtname))
                        labelfile_person_part = open(path_person_part_gt,'w+') #read label
                        labelfile_person_part.write(f'{cls_id} {x_center_normalized} {y_center_normalized} {w_normalized} {h_normalized}\n')
                        labelfile_person_part.close()
                    if(with_save_yoloLables):
                        txtname = name+"_"+str(index_bbox)+"_"+str(new_scale_codename) + '.txt'
                        path_person_part_yoloLabels = os.path.abspath(os.path.join(savedir, 'clean/', 'imgs_person_yolo-labels/', txtname))
                        labelfile_person_part_yoloLabels = open(path_person_part_yoloLabels,'w+')
                        labelfile_person_part_yoloLabels.write(f'{cls_id} {x_center_detected} {y_center_detected} {width_detected} {height_detected}\n')
                        labelfile_person_part_yoloLabels.close()

            # open beeld en pas aan naar yolo input size (open image and adjust to yolo input size)
            imgfile = os.path.abspath(os.path.join(imgdir, imgfile))
            img = Image.open(imgfile).convert('RGB')
            w,h = img.size
            # rescale the ground-truth from [0~1] to [0~(w,h)]
            bbox_labels_rescale = []
            if w==h:
                padded_img = img
            else:
                dim_to_pad = 1 if w<h else 2
                if dim_to_pad == 1:
                    padding = (h - w) / 2
                    padded_img = Image.new('RGB', (h,h), color=(127,127,127))
                    padded_img.paste(img, (int(padding), 0))
                    #
                    if not(labeldir == None):
                        index_bbox = 0
                        for bbox_label in bbox_labels:
                            cls_id      = bbox_label[0]
                            x_center    = bbox_label[1]
                            y_center    = bbox_label[2]
                            width       = bbox_label[3]
                            height      = bbox_label[4]
                            #
                            large_side = h
                            small_side = w
                            l_temp = (x_center - width / 2) * img_size
                            r_temp = (x_center + width / 2) * img_size
                            left        = ((x_center - width / 2) * w + padding) * (img_size/h)
                            right       = ((x_center + width / 2) * w + padding) * (img_size/h)
                            #
                            top         = (y_center - height / 2) * img_size
                            bottom      = (y_center + height / 2) * img_size
                            #
                            if(cls_id == 0):
                                bbox_labels_rescale.append([left, top, right, bottom])
                                if(enable_save_person_part):
                                    if((width*height) > person_part_thre):
                                        for imag_scale in person_part_rescale:
                                            save_image_person_part(padded_img, left, top, right, bottom,
                                                                    new_scale=imag_scale,
                                                                    index_bbox=index_bbox,
                                                                    cls_id=cls_id, 
                                                                    with_save_gt=True,
                                                                    with_save_yoloLables=True,
                                                                    with_no_detected_filtter=True)
                            index_bbox = index_bbox + 1
                else:
                    padding = (w - h) / 2
                    padded_img = Image.new('RGB', (w, w), color=(127,127,127))
                    padded_img.paste(img, (0, int(padding)))
                    if not(labeldir == None):
                        index_bbox = 0
                        for bbox_label in bbox_labels:
                            cls_id      = bbox_label[0]
                            x_center    = bbox_label[1]
                            y_center    = bbox_label[2]
                            width       = bbox_label[3]
                            height      = bbox_label[4]
                            #
                            left        = (x_center - width / 2) * img_size
                            right       = (x_center + width / 2) * img_size
                            #
                            large_side = w
                            small_side = h
                            t_temp = (y_center - height / 2) * img_size
                            b_temp = (y_center + height / 2) * img_size
                            top         = ((y_center - height / 2) * h + padding) * (img_size/w)
                            bottom      = ((y_center + height / 2) * h + padding) * (img_size/w)
                            #
                            if(cls_id == 0):
                                bbox_labels_rescale.append([left, top, right, bottom])
                                if(enable_save_person_part):
                                    if((width*height) > person_part_thre):
                                        for imag_scale in person_part_rescale:
                                            save_image_person_part(padded_img, left, top, right, bottom,
                                                                    new_scale=imag_scale,
                                                                    index_bbox=index_bbox,
                                                                    cls_id=cls_id, 
                                                                    with_save_gt=True,
                                                                    with_save_yoloLables=True,
                                                                    with_no_detected_filtter=True)
                            index_bbox = index_bbox + 1
                
            #
            resize = transforms.Resize((img_size,img_size))
            padded_img = resize(padded_img)
            padded_img_original = padded_img.copy()
            cleanname = name + ".png"
            # sla dit beeld op (save this image)
            # padded_img.save(os.path.join(savedir, 'clean/', cleanname))

            # Clear images            
            t0 =  time.time()
            # genereer een label file voor het gepadde beeld (generate a label file for the pathed image)
            boxes = do_detect(darknet_model, padded_img, 0.4, 0.4, True)
            # check num feature
            _count_feature = 0
            for box in boxes:
                cls_id = box[6]
                if(cls_id == feature_id):
                    if(box[4].item() > _confidence): # detection confidence
                        _count_feature = _count_feature + 1
            if(_count_feature>max_num_feature):
                _skip_img = True
                _skip_num_img = _skip_num_img + 1
                # print("\n too many feature (prediction) ("+str(_count_feature)+" / "+str(len(boxes))+") : "+str(txtname))
                skipname = txtname
                skippath = os.path.abspath(os.path.join(savedir, 'skip/', skipname))
                skipfile = open(skippath,'w+')
                skipfile.write(f'{str(str(_count_feature)+" / "+str(len(boxes)))} {str(skipname)}\n')
                skipfile.close()
                continue
            t1 =  time.time()

            # save images filtered to disk
            if not(labeldir == None):
                # img_size bbox rate
                img_area = img_size * img_size
                max_bbox_rate = -0.1
                for bbox in bbox_labels_rescale:
                    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    bbox_rate = bbox_area / img_area
                    if(max_bbox_rate < bbox_rate):
                        max_bbox_rate = bbox_rate
                    bbox_rates.append(bbox_rate)
                bbox_rates_per_img.append(max_bbox_rate)
                if(enable_roi_filter):
                    _image_filtered = False
                    if(max_bbox_rate > roi_thres):
                        _image_filtered = True
                        src = imgfile
                        if(data_mode.lower() == "test"):
                            dst = os.path.abspath(os.path.join(savedir, 'test/', img_name))
                        elif(data_mode.lower() == "train"):
                            dst = os.path.abspath(os.path.join(savedir, 'train/', img_name))
                        elif(data_mode.lower() == "create"):
                            dst = os.path.abspath(os.path.join(savedir, 'create/', img_name))
                        shutil.copyfile(src, dst)

            _save_data = False
            if(enable_roi_filter):
                if(_image_filtered):
                    _save_data = True
            else:
                _save_data = True
            if(_save_data):
                # save ground truth lables
                if not(labeldir == None):
                    if(len(bbox_labels_rescale) > 0):
                        labelfile_rescale = open(labelpath_rescale,'w+') #read label
                        for bbox in bbox_labels_rescale:
                            labelfile_rescale.write("person" + str(f' {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n'))  # left, top, right, bottom
                        labelfile_rescale.close()
                    else:
                        _skip_img = True
                        _skip_num_img = _skip_num_img + 1
                        # print("\n no any target class object (prediction) ("+str(_count_feature)+" / "+str(len(boxes))+") : "+str(txtname))
                        skipname = txtname
                        skippath = os.path.abspath(os.path.join(savedir, 'skip/', skipname))
                        skipfile = open(skippath,'w+')
                        skipfile.write(f'{str(str(_count_feature)+" / "+str(len(boxes)))} {str(skipname)}\n')
                        skipfile.close()
                        continue

            # print("boxes[0] len :\n"+str(len(boxes[0])))
            # print("boxes[0]     :\n"+str(boxes[0]))
            _save_data = False
            if(enable_roi_filter):
                if(_image_filtered):
                    _save_data = True
            else:
                _save_data = True
            if(_save_data):
                textfile = open(txtpath,'w+')
                text_rescalefile = open(txtpath_rescale,'w+')
                for box in boxes:
                    cls_id = box[6]
                    if(cls_id == 0):   #if person
                        det_score   = box[4]
                        c_cla_score = box[5]
                        cla_score   = det_score * c_cla_score
                        if(cla_score > _confidence): # detection confidence
                            x_center    = box[0]
                            y_center    = box[1]
                            width       = box[2]
                            height      = box[3]
                            left        = (x_center.item() - width.item() / 2) * padded_img.size[0]
                            right       = (x_center.item() + width.item() / 2) * padded_img.size[0]
                            top         = (y_center.item() - height.item() / 2) * padded_img.size[0]
                            bottom      = (y_center.item() + height.item() / 2) * padded_img.size[0]
                            #
                            text_rescalefile.write("person" + str(f' {cla_score} {left} {top} {right} {bottom}\n'))
                            textfile.write(f'{cls_id} {x_center} {y_center} {width} {height}\n')
                            clean_results.append({'image_id': name, 'bbox': [x_center.item() - width.item() / 2,
                                                                            y_center.item() - height.item() / 2,
                                                                            width.item(),
                                                                            height.item()],
                                                'score': box[4].item(),
                                                'category_id': 1})
                            if(enable_show_predection_bbox):
                                # img with prediction
                                draw = ImageDraw.Draw(padded_img)
                                shape_ = [(x_center.item() - width.item() / 2),
                                        (y_center.item() - height.item() / 2), 
                                        (x_center.item() + width.item() / 2),
                                        (y_center.item() + height.item() / 2)]
                                shape = [ tt * padded_img.size[0]  for tt in shape_]
                                draw.rectangle(shape, outline ="green")
                                # text
                                color = [0,255,0]
                                sentence = "person\n(" + str(round(float(det_score), 2)) + ", " + str(round(float(c_cla_score), 2)) + ")"
                                position = [((x_center.item() - width.item() / 2) * padded_img.size[0]),
                                        ((y_center.item() - height.item() / 2) * padded_img.size[0])]
                                draw.text(tuple(position), sentence, tuple(color))
                if(enable_show_predection_bbox):
                    # img with groudtruth
                    for bbox_label_rescale in bbox_labels_rescale:
                        draw = ImageDraw.Draw(padded_img)
                        shape = [(bbox_label_rescale[0]),
                                (bbox_label_rescale[1]), 
                                (bbox_label_rescale[2]),
                                (bbox_label_rescale[3])]
                        draw.rectangle(shape, outline ="white")
                if(save_clean_imgs):
                    padded_img.save(os.path.join(savedir, 'clean/imgs/', cleanname))
                textfile.close()
                text_rescalefile.close()

                # lees deze labelfile terug in als tensor (read this label file back as a tensor)    
                if os.path.getsize(txtpath):       #check to see if label file contains data. 
                    label = np.loadtxt(txtpath)
                else:
                    label = np.ones([5])
                label = torch.from_numpy(label).float()
                if label.dim() == 1:
                    label = label.unsqueeze(0)
                
            t2 =  time.time()

            # # print("Time do_detect : "+str(t1-t0))
            # # print("Time bbox      : "+str(t2-t1))

            

            if(enable_proper_part):
                # pre-process
                padded_img = padded_img_original
                transform = transforms.ToTensor()
                padded_img = transform(padded_img).cuda()
                img_fake_batch = padded_img.unsqueeze(0)
                lab_fake_batch = label.unsqueeze(0).cuda()
                
                # transformeer patch en voeg hem toe aan beeld (transform patch and add it to image)
                adv_patch_01 = torch.narrow(torch.narrow(adv_patch, 1, 0, 150), 2, 0, 150)
                adv_patch_02 = torch.narrow(torch.narrow(adv_patch, 1, 0, 150), 2, 150, 150)
                adv_patch_03 = torch.narrow(torch.narrow(adv_patch, 1, 150, 150), 2, 0, 150)
                adv_patch_04 = torch.narrow(torch.narrow(adv_patch, 1, 150, 150), 2, 150, 150)
                if(local_index == 0):
                    adv_patch_input = adv_patch_01
                elif(local_index == 1):
                    adv_patch_input = adv_patch_02
                elif(local_index == 2):
                    adv_patch_input = adv_patch_03
                elif(local_index == 3):
                    adv_patch_input = adv_patch_04
                elif(local_index == -1):
                    adv_patch_input = adv_patch
                adv_batch_t, _, _ = patch_transformer(adv_patch_input, lab_fake_batch, img_size, do_rotate=True, rand_loc=False, with_black_trans=True, scale_rate = patch_scale)
                # adv_batch_t, _ = patch_transformer(adv_patch, lab_fake_batch, img_size, do_rotate=True, rand_loc=False)
                p_img_batch = patch_applier(img_fake_batch, adv_batch_t)
                p_img_batch = p_img_batch.clamp_(0,1)       #keep patch in image range
                # # test
                # print("adv_batch_t size: "+str(adv_batch_t.size()))
                # test_img = adv_batch_t[0].detach().cpu()
                # test_img_plt = transforms.ToPILImage()(test_img)
                # test_img_plt.show()
                # #
                p_img = p_img_batch.squeeze(0)
                p_img_pil = transforms.ToPILImage('RGB')(p_img.cpu())
                properpatchedname = name + ".png"
                # p_img_pil.save(os.path.join(savedir, 'proper_patched/', properpatchedname))
                
                _save_data = False
                if(enable_roi_filter):
                    if(_image_filtered):
                        _save_data = True
                else:
                    _save_data = True
                if(_save_data):
                    # genereer een label file voor het beeld met sticker (generate a label file for the image with sticker)
                    txtname = properpatchedname.replace('.png', '.txt')
                    txtpath = os.path.abspath(os.path.join(savedir, 'proper_patched/', 'yolo-labels/', txtname))
                    txtpath_rescale = os.path.abspath(os.path.join(savedir, 'proper_patched/', 'yolo-labels_rescale/', txtname))
                    boxes = do_detect(darknet_model, p_img_pil, 0.01, 0.4, True)
                    textfile = open(txtpath,'w+')
                    text_rescalefile = open(txtpath_rescale,'w+')
                    for box in boxes:
                        cls_id = box[6]
                        if(cls_id == 0):   #if person
                            det_score   = box[4]
                            c_cla_score = box[5]
                            cla_score   = det_score * c_cla_score
                            if(cla_score > _confidence): # detection confidence
                                x_center    = box[0]
                                y_center    = box[1]
                                width       = box[2]
                                height      = box[3]
                                left        = (x_center.item() - width.item() / 2) * img_size
                                right       = (x_center.item() + width.item() / 2) * img_size
                                top         = (y_center.item() - height.item() / 2) * img_size
                                bottom      = (y_center.item() + height.item() / 2) * img_size
                                #
                                text_rescalefile.write("person" + str(f' {cla_score} {left} {top} {right} {bottom}\n'))
                                textfile.write(f'{cls_id} {x_center} {y_center} {width} {height}\n')
                                patch_results.append({'image_id': name, 'bbox': [x_center.item() - width.item() / 2, y_center.item() - height.item() / 2, width.item(), height.item()], 'score': box[4].item(), 'category_id': 1})
                                if(enable_show_predection_bbox):
                                    # img with prediction
                                    draw = ImageDraw.Draw(p_img_pil)
                                    shape_ = [(x_center.item() - width.item() / 2),
                                            (y_center.item() - height.item() / 2), 
                                            (x_center.item() + width.item() / 2),
                                            (y_center.item() + height.item() / 2)]
                                    shape = [ tt * p_img_pil.size[0]  for tt in shape_]
                                    draw.rectangle(shape, outline ="green")
                                    # text
                                    color = [0,255,0]
                                    sentence = "person\n(" + str(round(float(det_score), 2)) + ", " + str(round(float(c_cla_score), 2)) + ")"
                                    position = [((x_center.item() - width.item() / 2) * p_img_pil.size[0]),
                                            ((y_center.item() - height.item() / 2) * p_img_pil.size[0])]
                                    draw.text(tuple(position), sentence, tuple(color))
                    p_img_pil.save(os.path.join(savedir, 'proper_patched/imgs/', properpatchedname))
                    textfile.close()
                    text_rescalefile.close()


            if(enable_random_part):
                # maak een random patch, transformeer hem en voeg hem toe aan beeld (make a random patch, transform it and add it to image)
                random_patch = torch.rand(adv_patch_cpu.size()).cuda()
                adv_batch_t, _ = patch_transformer(random_patch, lab_fake_batch, img_size, do_rotate=True, rand_loc=False)
                p_img_batch = patch_applier(img_fake_batch, adv_batch_t)
                p_img = p_img_batch.squeeze(0)
                p_img_pil = transforms.ToPILImage('RGB')(p_img.cpu())
                properpatchedname = name + ".png"
                # p_img_pil.save(os.path.join(savedir, 'random_patched/', properpatchedname))
                _save_data = False
                if(enable_roi_filter):
                    if(_image_filtered):
                        _save_data = True
                else:
                    _save_data = True
                if(_save_data):
                    # genereer een label file voor het beeld met random patch (generate a label file for the image with random patch)
                    txtname = properpatchedname.replace('.png', '.txt')
                    txtpath = os.path.abspath(os.path.join(savedir, 'random_patched/', 'yolo-labels/', txtname))
                    txtpath_rescale = os.path.abspath(os.path.join(savedir, 'random_patched/', 'yolo-labels_rescale/', txtname))
                    boxes = do_detect(darknet_model, p_img_pil, 0.01, 0.4, True)
                    textfile = open(txtpath,'w+')
                    text_rescalefile = open(txtpath_rescale,'w+')
                    for box in boxes:
                        cls_id = box[6]
                        if(cls_id == 0):   #if person
                            det_score   = box[4]
                            c_cla_score = box[5]
                            cla_score   = det_score * c_cla_score
                            if(cla_score > _confidence): # detection confidence
                                x_center    = box[0]
                                y_center    = box[1]
                                width       = box[2]
                                height      = box[3]
                                left        = (x_center.item() - width.item() / 2) * img_size
                                right       = (x_center.item() + width.item() / 2) * img_size
                                top         = (y_center.item() - height.item() / 2) * img_size
                                bottom      = (y_center.item() + height.item() / 2) * img_size
                                text_rescalefile.write("person" + str(f' {cla_score} {left} {top} {right} {bottom}\n'))
                                textfile.write(f'{cls_id} {x_center} {y_center} {width} {height}\n')
                                noise_results.append({'image_id': name, 'bbox': [x_center.item() - width.item() / 2, y_center.item() - height.item() / 2, width.item(), height.item()], 'score': box[4].item(), 'category_id': 1})
                                if(enable_show_predection_bbox):
                                    # img with prediction
                                    draw = ImageDraw.Draw(p_img_pil)
                                    shape_ = [(x_center.item() - width.item() / 2),
                                            (y_center.item() - height.item() / 2), 
                                            (x_center.item() + width.item() / 2),
                                            (y_center.item() + height.item() / 2)]
                                    shape = [ tt * p_img_pil.size[0]  for tt in shape_]
                                    draw.rectangle(shape, outline ="green")
                                    # text
                                    color = [0,255,0]
                                    sentence = "person\n(" + str(round(float(det_score), 2)) + ", " + str(round(float(c_cla_score), 2)) + ")"
                                    position = [((x_center.item() - width.item() / 2) * p_img_pil.size[0]),
                                            ((y_center.item() - height.item() / 2) * p_img_pil.size[0])]
                                    draw.text(tuple(position), sentence, tuple(color))
                    p_img_pil.save(os.path.join(savedir, 'random_patched/imgs/', properpatchedname))
                    textfile.close()
                    text_rescalefile.close()

    if(enable_show_roirate_distribution):
        # show rate plt
        # red dashes, blue squares and green triangles
        bbox_rates.sort()
        bbox_rates_per_img.sort()
        print("")
        print("bbox_rates len: "+str(len(bbox_rates)))
        plt.plot((np.arange(0, int(len(bbox_rates)), 1)), bbox_rates, 'g^')
        plt.title("bbox_rates")
        plt.show()
        print("bbox_rates_per_img len: "+str(len(bbox_rates_per_img)))
        plt.plot((np.arange(0, int(len(bbox_rates_per_img)), 1)), bbox_rates_per_img, 'g^')
        plt.title("bbox_rates_per_img")
        plt.show()

    with open(savedir+'clean_results.json', 'w') as fp:
        json.dump(clean_results, fp)
    with open(savedir+'noise_results.json', 'w') as fp:
        json.dump(noise_results, fp)
    with open(savedir+'patch_results.json', 'w') as fp:
        json.dump(patch_results, fp)

    print("Finish......")