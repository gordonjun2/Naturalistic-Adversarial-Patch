# -*- coding: utf-8 -*-
'''
@Time          : 20/04/25 15:49
@Author        : huguanghao
@File          : demo.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :
'''

# import sys
# import time
# from PIL import Image, ImageDraw
# from models.tiny_yolo import TinyYoloNet
if not(__name__ == "demo") and not(__name__ == "__main__"):
    import sys
    sys.path.append('pytorchYOLOv4/')
from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
import torch.nn.functional as F
import argparse
from ipdb import set_trace as st


"""hyper parameters"""
use_cuda = True

class MaxProbExtractor(torch.nn.Module):
    """MaxProbExtractor: extracts max class probability for class from YOLO output.

    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

    """

    def __init__(self, cls_id=0, num_cls=80):
        super(MaxProbExtractor, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls

    def set_cls_id_attacked(self, cls_id):
        self.cls_id = cls_id

    def forward(self, YOLOoutput):
        ## YOLOoutput size : torch.Size([1, 22743, 80])
        YOLOoutput = YOLOoutput[:,:,self.cls_id]
        max_conf_target_obj_cls, max_conf_indexes  = torch.max(YOLOoutput, dim=1)
        # st()
        return max_conf_target_obj_cls

class DetectorYolov4():
    def __init__(self, cfgfile='./pytorchYOLOv4/cfg/yolov4.cfg', weightfile='./pytorchYOLOv4/weight/yolov4.weights', show_detail=False, tiny=False):
        if tiny:
            cfgfile    = './pytorchYOLOv4/cfg/yolov4-tiny.cfg'
            weightfile = './pytorchYOLOv4/weight/yolov4-tiny.weights'
        self.show_detail = show_detail
        if(self.show_detail):
            start_init        = time.time()
            self.m = Darknet(cfgfile)
            finish_init       = time.time()
            start_w           = time.time()
            # m.print_network()
            self.m.load_weights(weightfile)
            finish_w          = time.time()
            print('Loading Yolov4 weights from %s... Done!' % (weightfile))
            start_d           = time.time()
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.device == "cuda":
                use_cuda = True
                self.m.cuda()
                # init MaxProbExtractor
                self.max_prob_extractor = MaxProbExtractor().cuda()
            else:
                use_cuda = False
                # init MaxProbExtractor
                self.max_prob_extractor = MaxProbExtractor()
            finish_d          = time.time()
            
            print('Yolov4 init model  Predicted in %f seconds.' % (finish_init - start_init))
            print('Yolov4 load weight Predicted in %f seconds.' % (finish_w - start_w))
            print('Yolov4 load device Predicted in %f seconds.' % (finish_d - start_d))
            print('Total time :%f ' % (finish_d - start_init))
        else:
            self.m = Darknet(cfgfile)
            # m.print_network()
            self.m.load_weights(weightfile)
            print('Loading Yolov4 weights from %s... Done!' % (weightfile))
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.device == "cuda":
                use_cuda = True
                self.m.cuda()
                 # init MaxProbExtractor
                self.max_prob_extractor = MaxProbExtractor().cuda()
            else:
                use_cuda = False
                # init MaxProbExtractor
                self.max_prob_extractor = MaxProbExtractor()
        # 
    def detect(self, input_imgs, cls_id_attacked, clear_imgs=None, with_bbox=True):
        # resize image
        # input_imgs_ori = input_imgs.clone() # np.save('gg', input_imgs_ori.cpu().detach().numpy())   
                                            #  gg=np.load('gg.npy')    gg2=input_imgs_ori.cpu().detach().numpy() np.argwhere(gg!=gg2)
        input_imgs        = F.interpolate(input_imgs, size=self.m.width).to(self.device)
        # bbox
        if(with_bbox):
            boxes         = do_detect(self.m, input_imgs, 0.4, 0.6, use_cuda)
            # print("boxes size : "+str(np.array(boxes).shape))
            # print("boxes      : "+str(boxes))
            bbox          = [torch.Tensor(box) for box in boxes] ## [torch.Size([3, 7]), torch.Size([2, 7]), ...]
        else:
            self.m.eval()
        # detections_tensor
        output            = self.m(input_imgs)
        detections_tensor_xy    = output[0]    # xy1xy2, torch.Size([1, 22743, 1, 4])
        detections_tensor_class = output[1] # conf, torch.Size([1, 22743, 80])
        # st() # np.save('gg', input_imgs.cpu().detach().numpy())  gg=np.load('gg.npy')  gg2=input_imgs.cpu().detach().numpy() np.argwhere(gg!=gg2)
        # print("input_imgs",input_imgs[0,0,:10,0])
        # print("detections_tensor_xy:",detections_tensor_xy[0,:10,0,0])
        # print("detections_tensor_class:",detections_tensor_class[0,:10,0])
        # st()
        # Get probility
        self.max_prob_extractor.set_cls_id_attacked(cls_id_attacked)
        max_prob_obj_cls = self.max_prob_extractor(detections_tensor_class)
        # print("detections_tensor_class:",detections_tensor_class[1,:10,0])
        # print("max_prob_obj_cls",max_prob_obj_cls)

        # get overlap_score
        if not(clear_imgs == None):
            # resize image
            clear_imgs = F.interpolate(clear_imgs, size=self.m.width).to(self.device)
            # detections_tensor
            output_clear = self.m(clear_imgs)
            detections_tensor_xy_clear    = output_clear[0]    # xy1xy2, torch.Size([1, 22743, 1, 4])
            detections_tensor_class_clear = output_clear[1] # conf, torch.Size([1, 22743, 80])
            #
            max_prob_obj_cls_clear = self.max_prob_extractor(detections_tensor_class_clear)
            # count overlap
            output_score       = max_prob_obj_cls
            output_score_clear = max_prob_obj_cls_clear
            overlap_score      = torch.abs(output_score-output_score_clear)
        else:
            overlap_score = torch.tensor(0).to(self.device)
        # st()
        if(with_bbox):
            return max_prob_obj_cls, overlap_score, bbox
        else:
            return max_prob_obj_cls, overlap_score, []
    

def detect_cv2(cfgfile, weightfile, imgfile):
    import cv2
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/x.names'
    class_names = load_class_names(namesfile)

    img = cv2.imread(imgfile)
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

    plot_boxes_cv2(img, boxes[0], savename='predictions.jpg', class_names=class_names)


def detect_cv2_camera(cfgfile, weightfile):
    import cv2
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("./test.mp4")
    cap.set(3, 1280)
    cap.set(4, 720)
    print("Starting the YOLO loop...")

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/x.names'
    class_names = load_class_names(namesfile)

    while True:
        ret, img = cap.read()
        sized = cv2.resize(img, (m.width, m.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        start = time.time()
        boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
        finish = time.time()
        print('Predicted in %f seconds.' % (finish - start))

        result_img = plot_boxes_cv2(img, boxes[0], savename=None, class_names=class_names)

        cv2.imshow('Yolo demo', result_img)
        cv2.waitKey(1)

    cap.release()


def detect_skimage(cfgfile, weightfile, imgfile):
    from skimage import io
    from skimage.transform import resize
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/x.names'
    class_names = load_class_names(namesfile)

    img = io.imread(imgfile)
    sized = resize(img, (m.width, m.height)) * 255

    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.4, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

    plot_boxes_cv2(img, boxes, savename='predictions.jpg', class_names=class_names)


def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('-cfgfile', type=str, default='./cfg/yolov4.cfg',
                        help='path of cfg file', dest='cfgfile')
    parser.add_argument('-weightfile', type=str,
                        default='./checkpoints/Yolov4_epoch1.pth',
                        help='path of trained model.', dest='weightfile')
    parser.add_argument('-imgfile', type=str,
                        default='./data/mscoco2017/train2017/190109_180343_00154162.jpg',
                        help='path of your image file.', dest='imgfile')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    if args.imgfile:
        detect_cv2(args.cfgfile, args.weightfile, args.imgfile)
        # detect_imges(args.cfgfile, args.weightfile)
        # detect_cv2(args.cfgfile, args.weightfile, args.imgfile)
        # detect_skimage(args.cfgfile, args.weightfile, args.imgfile)
    else:
        detect_cv2_camera(args.cfgfile, args.weightfile)
