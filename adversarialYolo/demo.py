from adversarialYolo.utils import do_detect, load_class_names, plot_boxes_cv2
import cv2
from adversarialYolo.darknet import Darknet
import torch.nn.functional as F
import torch
import sys
import numpy as np
from ipdb import set_trace as st

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
        ## YOLOoutput size : torch.Size([8, 425, 13, 13])
        if YOLOoutput.dim() == 3:
            YOLOoutput = YOLOoutput.unsqueeze(0)
        batch = YOLOoutput.size()[0]
        assert (YOLOoutput.size()[1] == (5 + self.num_cls ) * 5)
        # print("YOLOoutput        size : "+str(YOLOoutput.size()))
        output = YOLOoutput.view(batch, 5, 5 + self.num_cls , -1)  # [batch, 5, 85, 361]
        # print("output            size : "+str(output.size()))
        output = output.transpose(1, 2).contiguous()  # [batch, 85, 5, 361]
        # print("output            size : "+str(output.size()))
        output = output.view(batch, 5 + self.num_cls , -1)  # [batch, 85, 1805]
        # print("output            size : "+str(output.size()))
        output_objectness = output[:, 4, :]
        output_objectness = torch.sigmoid(output_objectness)
        # print("output_objectness size : "+str(output_objectness.size()))
        output_class = output[: , 5:(5 + self.num_cls), :]
        # print("output_class      size : "+str(output_class.size()))
        output_class = torch.nn.Softmax(dim=1)(output_class)
        output_class = output_class[:, self.cls_id, :]
        # print("output_class      size : "+str(output_class.size()))
        
        # print("max_conf_target_obj size : "+str(max_conf_target_obj.size()))
        # print("max_conf_target_cls size : "+str(max_conf_target_cls.size()))
        # print("max_conf_target_obj    : "+str(max_conf_target_obj))
        # print("max_conf_target_cls    : "+str(max_conf_target_cls))

        ##
        # YOLOoutput        size : torch.Size([8, 425, 13, 13])
        # output            size : torch.Size([8, 5, 85, 169])
        # output            size : torch.Size([8, 85, 5, 169])
        # output            size : torch.Size([8, 85, 845])
        # output_objectness size : torch.Size([8, 845])
        # output_class      size : torch.Size([8, 80, 845])
        # output_class      size : torch.Size([8, 845])
        # max_conf_target_obj size : torch.Size([8])
        # max_conf_target_cls size : torch.Size([8])


        return output_objectness, output_class

class DetectorYolov2():
    def __init__(self, cfgfile='./adversarialYolo/cfg/yolo.cfg', weightfile='./adversarialYolo/weights/yolo.weights', show_detail=False):
        self.show_detail = show_detail
        self.darknet_model = Darknet(cfgfile)
        self.darknet_model.load_weights(weightfile)
        self.model_iheight = self.darknet_model.height
        self.model_iwidth  = self.darknet_model.width

        print('Loading Yolov2 weights from %s... Done!' % (weightfile))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            self.use_cuda = True
            self.darknet_model.cuda()
            # init MaxProbExtractor
            self.max_prob_extractor = MaxProbExtractor().cuda()
        else:
            self.use_cuda = False
            # init MaxProbExtractor
            self.max_prob_extractor = MaxProbExtractor()

    def detect(self, input_imgs, cls_id_attacked, clear_imgs=None, with_bbox=True):
        self.darknet_model.eval()
        # resize image
        if not(input_imgs.size()[-1] == self.model_iwidth):
            input_imgs = F.interpolate(input_imgs, size=self.model_iwidth).to(self.device)
        else:
            input_imgs = input_imgs.to(self.device)
        # bbox
        if(with_bbox):
            # print("input_imgs size : "+str(input_imgs.size()))
            boxes  = do_detect(self.darknet_model, input_imgs, 0.4, 0.6, self.use_cuda)
            # print("boxes size : "+str(np.array(boxes).shape))
            # print("boxes      : "+str(boxes))
            bbox   = [torch.Tensor(box) for box in boxes] ## [torch.Size([3, 7]), torch.Size([2, 7]), ...]
            # print("len(bbox)  : "+str(len(bbox)))
            # print("bbox       : "+str(bbox))
        else:
            self.darknet_model.eval()
        # detections_tensor
        output = self.darknet_model(input_imgs) # output: torch.Size([1, 425, 13, 13])
        # init cls_id_attacked
        self.max_prob_extractor.set_cls_id_attacked(cls_id_attacked)
        output_objectness, output_class = self.max_prob_extractor(output)
        # probability
        # st()
        output_obj_cls = output_objectness * output_class           # output_objectness = output_class = (8,845)
        max_prob_obj_cls, max_conf_idx_target_obj_cls = torch.max(output_obj_cls, dim=1)
        # max_prob_obj, max_conf_idx_target_obj = torch.max(output_objectness, dim=1)
        # max_prob_cls, max_conf_idx_target_cls = torch.max(output_class, dim=1)

        # get overlap_score
        if not(clear_imgs == None):
            # resize image
            if not(input_imgs.size()[-1] == self.model_iwidth):
                clear_imgs = F.interpolate(clear_imgs, size=self.model_iwidth).to(self.device)
            else:
                clear_imgs = clear_imgs.to(self.device)
            # detections_tensor
            output_clear = self.darknet_model(clear_imgs)
            #
            output_objectness_clear, output_class_clear = self.max_prob_extractor(output_clear)
            # count overlap
            output_score       = (output_objectness * output_class) # torch.Size([8, 845])
            output_score_clear = (output_objectness_clear * output_class_clear)
            overlap_score      = torch.sum(torch.abs(output_score-output_score_clear), dim=1)
        else:
            overlap_score = torch.tensor(0).to(self.device)

        # return
        if(with_bbox):
            # return max_prob_obj, max_prob_cls, overlap_score, bbox
            return max_prob_obj_cls, overlap_score, bbox
        else:
            # return max_prob_obj, max_prob_cls, overlap_score, [[]]
            return max_prob_obj_cls, overlap_score, [[]]

def demo(cfgfile, weightfile):
    m = Darknet(cfgfile)
    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'
    class_names = load_class_names(namesfile)
 
    use_cuda = 1
    if use_cuda:
        m.cuda()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to open camera")
        exit(-1)

    while True:
        res, img = cap.read()
        if res:
            sized = cv2.resize(img, (m.width, m.height))
            bboxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
            print('------')
            draw_img = plot_boxes_cv2(img, bboxes, None, class_names)
            cv2.imshow(cfgfile, draw_img)
            cv2.waitKey(1)
        else:
             print("Unable to read image")
             exit(-1) 

############################################
if __name__ == '__main__':
    if len(sys.argv) == 3:
        cfgfile = sys.argv[1]
        weightfile = sys.argv[2]
        demo(cfgfile, weightfile)
        #demo('cfg/tiny-yolo-voc.cfg', 'tiny-yolo-voc.weights')
    else:
        print('Usage:')
        print('    python demo.py cfgfile weightfile')
        print('')
        print('    perform detection on camera')
