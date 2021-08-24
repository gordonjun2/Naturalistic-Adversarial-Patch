from __future__ import division

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from ipdb import set_trace as st

if __name__ == "__main__":
    from models import *
    from utils.utils import *
    from utils.datasets import *

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode

    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))
        # print("input_imgs size : "+str(input_imgs.size())) ## torch.Size([1, 3, 416, 416])

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print("\nSaving images:")
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        print("(%d) Image: '%s'" % (img_i, path))

        # Create plot
        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                box_w = x2 - x1
                box_h = y2 - y1

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(
                    x1,
                    y1,
                    s=classes[int(cls_pred)],
                    color="white",
                    verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                )

        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = path.split("/")[-1].split(".")[0]
        plt.savefig(f"output/{filename}.png", bbox_inches="tight", pad_inches=0.0)
        plt.close()
else:
    sys.path.append('PyTorchYOLOv3/')
    from models import *
    from utils.utils import *
    from utils.datasets import *
    
class MaxProbExtractor(nn.Module):
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
        ## YOLOoutput size : torch.Size([4, 2535, 85])
        batch = YOLOoutput.size()[0]
        # print("YOLOoutput        size : "+str(YOLOoutput.size()))
        output_objectness = YOLOoutput[:, :, 4]
        # print("output_objectness size : "+str(output_objectness.size()))
        output_class = YOLOoutput[: , :, 5:(5 + self.num_cls)]
        # print("output_class      size : "+str(output_class.size()))
        output_class_new = output_class[:, :, self.cls_id]
        # print("output_class      size : "+str(output_class.size()))
        # max_conf_target_obj, max_conf_idx_target_obj = torch.max(output_objectness, dim=1)
        # max_conf_target_cls, max_conf_idx_target_cls = torch.max(output_class, dim=1)
        # output_cls_obj = output_objectness*output_class_new
        # max_conf_target_cls_obj, max_conf_idx_target_cls_obj = torch.max(output_cls_obj, dim=1)
        # st()
        # print("max_conf_target_obj size : "+str(max_conf_target_obj.size()))
        # print("max_conf_target_cls size : "+str(max_conf_target_cls.size()))
        # print("max_conf_target_obj    : "+str(max_conf_target_obj))
        # print("max_conf_target_cls    : "+str(max_conf_target_cls))

        # #
        # YOLOoutput        size : torch.Size([8, 2535, 85])
        # output_objectness size : torch.Size([8, 2535])
        # output_class      size : torch.Size([8, 2535, 80])
        # output_class      size : torch.Size([8, 2535])


        # return max_conf_target_obj, max_conf_target_cls
        return output_objectness, output_class_new

class DetectorYolov3():
    def __init__(self, cfgfile="PyTorchYOLOv3/config/yolov3.cfg", weightfile="PyTorchYOLOv3/weights/yolov3.weights", show_detail=False, tiny=False):
        #
        start_t = time.time()

        if(tiny):
            cfgfile    = "PyTorchYOLOv3/config/yolov3-tiny.cfg"
            weightfile = "PyTorchYOLOv3/weights/yolov3-tiny.weights"

        self.show_detail = show_detail
        # check whether cuda or cpu
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Set up model
        self.model = Darknet(cfgfile)
        self.model.load_darknet_weights(weightfile)
        self.img_size = self.model.img_size
        
        print('Loading Yolov3 weights from %s... Done!' % (weightfile))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            self.use_cuda = True
            self.model.cuda()
            # init MaxProbExtractor
            self.max_prob_extractor = MaxProbExtractor().cuda()
        else:
            self.use_cuda = False
            # init MaxProbExtractor
            self.max_prob_extractor = MaxProbExtractor()
        
        finish_t = time.time()
        if self.show_detail:
            print('Total init time :%f ' % (finish_t - start_t))
    def detect(self, input_imgs, cls_id_attacked, clear_imgs=None, with_bbox=True):
        start_t = time.time()
        # resize image
        input_imgs = F.interpolate(input_imgs, size=self.img_size).to(self.device)

        # Get detections
        self.model.eval()
        detections = self.model(input_imgs) ## v3tiny:torch.Size([8, 2535, 85]), v3:torch.Size([8, 10647, 85])
        if not(detections[0] == None):
            # init cls_id_attacked
            self.max_prob_extractor.set_cls_id_attacked(cls_id_attacked)
            # max_prob_obj, max_prob_cls = self.max_prob_extractor(detections)
            output_objectness, output_class = self.max_prob_extractor(detections)
            # print(output_objectness.shape, output_class.shape)
            output_cls_obj = torch.mul(output_objectness,output_class)
            max_prob_obj_cls, max_prob_obj_cls_index = torch.max(output_cls_obj, dim=1)
            # print(max_prob_obj_cls.shape)
            if(with_bbox):
                bboxes = non_max_suppression(detections, 0.4, 0.6) ## <class 'list'>. 
                # only non None. Replace None with torch.tensor([])
                bboxes = [torch.tensor([]) if bbox is None else bbox for bbox in bboxes]
                bboxes = [rescale_boxes(bbox, self.img_size, [1,1]) if bbox.dim() == 2 else bbox for bbox in bboxes] # shape [1,1] means the range of value is [0,1]
                # print("bboxes size : "+str(len(bboxes)))
                # print("bboxes      : "+str(bboxes))

            # get overlap_score
            if not(clear_imgs == None):
                # resize image
                input_imgs_clear = F.interpolate(clear_imgs, size=self.img_size).to(self.device)
                # detections_tensor
                detections_clear = self.model(input_imgs_clear) ## v3tiny:torch.Size([8, 2535, 85]), v3:torch.Size([8, 10647, 85])
                if not(detections_clear[0] == None):
                    #
                    # output_score_clear = self.max_prob_extractor(detections_clear)
                    output_score_obj_clear, output_score_cls_clear = self.max_prob_extractor(detections_clear)
                    output_cls_obj = output_score_obj_clear*output_score_cls_clear
                    # st()
                    output_score_clear, output_score_clear_index = torch.max(output_cls_obj,dim=1)
                    # count overlap
                    output_score       = max_prob_obj_cls
                    # output_score_clear = (max_prob_obj_clear * max_prob_cls_clear)
                    overlap_score      = torch.abs(output_score-output_score_clear)
                else:
                    overlap_score = torch.tensor(0).to(self.device)
            else:
                overlap_score = torch.tensor(0).to(self.device)
        else:
            print("None : "+str(type(detections)))
            print("None : "+str(detections))
            max_prob_obj = []
            max_prob_cls = []
            bboxes       = []
            overlap_score = torch.tensor(0).to(self.device)
        
        finish_t = time.time()
        if self.show_detail:
            print('Total init time :%f ' % (finish_t - start_t))
        if(with_bbox):
            # return max_prob_obj, max_prob_cls, overlap_score, bboxes
            return max_prob_obj_cls, overlap_score, bboxes
        else:
            # return max_prob_obj, max_prob_cls, overlap_score, [[]]
            return max_prob_obj_cls, overlap_score, [[]]
