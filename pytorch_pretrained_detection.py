# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Function `vis_bbox`, `vis_mask`, and `vis_class` are adapted from: 
# https://github.com/facebookresearch/Detectron/blob/7aa91aaa5a85598399dc8d8413e05a06ca366ba7/detectron/utils/vis.py
##############################################################################

"""PyTorch object detection example."""

import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as trns
from PIL import Image

_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_WHITE = (255, 255, 255)

_COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A", "N/A",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "N/A", "dining table",
    "N/A", "N/A", "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "N/A", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


def vis_bbox(image, bbox, color=_GREEN, thick=1):
    """Visualizes a bounding box."""
    image = image.astype(np.uint8)
    bbox = list(map(int, bbox))
    x0, y0, x1, y1 = bbox
    cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness=thick)
    return image


def vis_mask(image, mask, col, alpha=0.4, show_border=True, border_thick=1):
    """Visualizes a single binary mask."""
    image = image.astype(np.float32)

    mask = mask >= 0.5
    mask = mask.astype(np.uint8)
    idx = np.nonzero(mask)

    image[idx[0], idx[1], :] *= 1.0 - alpha
    image[idx[0], idx[1], :] += alpha * col

    if show_border:
        contours = cv2.findContours(
            mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)[-2]
        cv2.drawContours(image, contours, -1, _WHITE,
                         border_thick, cv2.LINE_AA)

    return image.astype(np.uint8)


def vis_class(image, bbox, text, bg_color=_GREEN, text_color=_GRAY, font_scale=0.35):
    """Visualizes the class."""
    image = image.astype(np.uint8)
    x0, y0 = int(bbox[0]), int(bbox[1])

    # Compute text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    ((text_w, text_h), _) = cv2.getTextSize(text, font, font_scale, 1)

    # Place text background
    back_tl = x0, y0 - int(1.3 * text_h)
    back_br = x0 + text_w, y0
    cv2.rectangle(image, back_tl, back_br, bg_color, -1)

    # Show text
    text_tl = x0, y0 - int(0.3 * text_h)
    cv2.putText(image, text, text_tl, font, font_scale,
                text_color, lineType=cv2.LINE_AA)

    return image


def run_object_detection(model, image_path, transforms, threshold=0.5, output_path="out.png"):
    """Inference."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Read image and run prepro
    image = Image.open(image_path).convert("RGB")
    image_tensor = transforms(image)
    print(f"\n\nImage size after transformation: {image_tensor.size()}")

    # Feed input and get results at index 0
    # (input image is at index 0 in the list)
    outputs = model([image_tensor])[0]
    print("outputs :\n"+str(outputs))
    # print("model([image_tensor]) :\n"+str(model([image_tensor])))

    # Result postpro and vis
    display_image = np.array(image)
    outputs = {k: v.detach().cpu().numpy() for k, v in outputs.items()}
    is_mask = True if "masks" in outputs else False
    if is_mask:
        outputs["masks"] = np.squeeze(outputs["masks"], axis=1)

    print("\n\nInference results:")
    for i, (bbox, label, score) in enumerate(zip(outputs["boxes"], outputs["labels"], outputs["scores"])):
        if score < threshold:
            continue

        print(
            f"Label {label}: {_COCO_INSTANCE_CATEGORY_NAMES[label]} ({score:.2f})")

        display_image = vis_bbox(display_image, bbox)
        display_image = vis_class(
            display_image, bbox, _COCO_INSTANCE_CATEGORY_NAMES[label])
        if is_mask:
            display_image = vis_mask(
                display_image, outputs["masks"][i], np.array([0., 0., 255.]))

    plt.figure(figsize=(10, 6))
    plt.imshow(display_image)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(output_path, bbox_inches="tight")

    return

class FasterrcnnResnet50:
    """
    tensor_image_input: torch.Size([3, h, w])
    """
    def __init__(self, device='cuda'):
        self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval().to(device)
        self.device = device
    def detect(self, tensor_image_inputs, cls_id_attacked=0, threshold=0.5):
        bboxes = []
        prof_max_scores = []
        any_max_scores = []
        for tensor_image_input in tensor_image_inputs:
            outputs = self.model([tensor_image_input])[0]
            #
            outputs["boxes"][:,0] = outputs["boxes"][:,0] / tensor_image_input.size()[-2]
            outputs["boxes"][:,1] = outputs["boxes"][:,1] / tensor_image_input.size()[-1]
            outputs["boxes"][:,2] = outputs["boxes"][:,2] / tensor_image_input.size()[-2]
            outputs["boxes"][:,3] = outputs["boxes"][:,3] / tensor_image_input.size()[-1]
            # create bbox with (batch,7). (x1,y1,x2,y2,score,score,class_id)
            batch = outputs["boxes"].size()[0]
            outputs["labels"] = outputs["labels"] - 1 # without class __background__
            bbox = torch.cat((outputs["boxes"], outputs["scores"].resize(batch,1), outputs["scores"].resize(batch,1), outputs["labels"].resize(batch,1)), 1)
            # get items with cls_id_attacked
            any_max_score = torch.max(bbox[:,-2])
            any_max_scores.append(any_max_score)
            bbox = bbox[(bbox[:,-1] == cls_id_attacked)]
            # score > threshold
            bbox = bbox[(bbox[:,-2] >= threshold)]
            if(bbox.size()[0]>0):
                # get max score
                max_score = torch.max(bbox[:,-2])
                # print("max_score : "+str(max_score))

                bboxes.append(bbox)
                prof_max_scores.append(max_score)
            else:
                bboxes.append(torch.tensor([]))
                prof_max_scores.append(torch.tensor(0.0).to(self.device))
        # stack
        if(len(prof_max_scores) > 0):
            prof_max_scores = torch.stack(prof_max_scores, dim=0)
        else:
            prof_max_scores = torch.stack(any_max_scores, dim=0) * 0.01
            if(tensor_image_inputs.is_cuda):
                prof_max_scores = prof_max_scores.cuda()
            else:
                prof_max_scores = prof_max_scores

        return prof_max_scores, bboxes

class MaskrcnnResnet50:
    """
    tensor_image_input: torch.Size([3, h, w])
    """
    def __init__(self, device='cuda'):
        self.model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        self.model.eval().to(device)
        self.device = device
    def detect(self, tensor_image_inputs, cls_id_attacked=0, threshold=0.5):
        bboxes = []
        prof_max_scores = []
        any_max_scores = []
        for tensor_image_input in tensor_image_inputs:
            outputs = self.model([tensor_image_input])[0]
            #
            outputs["boxes"][:,0] = outputs["boxes"][:,0] / tensor_image_input.size()[-2]
            outputs["boxes"][:,1] = outputs["boxes"][:,1] / tensor_image_input.size()[-1]
            outputs["boxes"][:,2] = outputs["boxes"][:,2] / tensor_image_input.size()[-2]
            outputs["boxes"][:,3] = outputs["boxes"][:,3] / tensor_image_input.size()[-1]
            # create bbox with (batch,7). (x1,y1,x2,y2,score,score,class_id)
            batch = outputs["boxes"].size()[0]
            outputs["labels"] = outputs["labels"] - 1 # without class __background__
            bbox = torch.cat((outputs["boxes"], outputs["scores"].resize(batch,1), outputs["scores"].resize(batch,1), outputs["labels"].resize(batch,1)), 1)
            # get items with cls_id_attacked
            any_max_score = torch.max(bbox[:,-2])
            any_max_scores.append(any_max_score)
            bbox = bbox[(bbox[:,-1] == cls_id_attacked)]
            # score > threshold
            bbox = bbox[(bbox[:,-2] >= threshold)]
            if(bbox.size()[0]>0):
                # get max score
                max_score = torch.max(bbox[:,-2])
                # print("max_score : "+str(max_score))

                bboxes.append(bbox)
                prof_max_scores.append(max_score)
            else:
                bboxes.append(torch.tensor([]))
                prof_max_scores.append(torch.tensor(0.0).to(self.device))
        # stack
        if(len(prof_max_scores) > 0):
            prof_max_scores = torch.stack(prof_max_scores, dim=0)
        else:
            prof_max_scores = torch.stack(any_max_scores, dim=0) * 0.01
            if(tensor_image_inputs.is_cuda):
                prof_max_scores = prof_max_scores.cuda()
            else:
                prof_max_scores = prof_max_scores

        return prof_max_scores, bboxes

if __name__ == "__main__":
    # parser = argparse.ArgumentParser("PyTorch Object Detection")
    # parser.add_argument("--image_path", type=str,
    #                     default="./data/imgs/giraffe.jpg", help="path to image")
    # parser.add_argument("--model_type", type=str,
    #                     default="fasterrcnn", help="fasterrcnn or maskrcnn")
    # parser.add_argument("--output_path", type=str,
    #                     default="./pytorch_model_out.png", help="path to save output image")

    # # Parse arguments
    # args = parser.parse_args()

    # # Define image transforms
    # transforms = trns.ToTensor()

    # # Load model
    # if args.model_type == "fasterrcnn":
    #     model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # elif args.model_type == "maskrcnn":
    #     model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    # else:
    #     raise AssertionError

    # print(model)

    # # Set model to eval mode
    # model.eval()

    # # Run model
    # run_object_detection(model, 
    #                     args.image_path,
    #                     transforms, 
    #                     output_path=args.output_path)

    ### test
    # image_path = "./data/imgs/giraffe.jpg"
    image_path_1 = "./crop001029.png"
    image_path_2 = "./crop001030.png"
    transforms = trns.ToTensor()
    image_1 = Image.open(image_path_1).convert("RGB")
    image_tensor_1 = transforms(image_1)
    image_2 = Image.open(image_path_2).convert("RGB")
    image_tensor_2 = transforms(image_2)
    testing_data = torch.cat((image_tensor_1.unsqueeze(0), image_tensor_2.unsqueeze(0)), 0)
    print(f"\n\nImage size after transformation: {testing_data.size()}")
    # fasterrcnn(testing_data)