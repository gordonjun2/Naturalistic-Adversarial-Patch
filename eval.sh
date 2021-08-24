#!/bin/bash
# WHERE = "./exp/exp07/generated/generated-images-1000.png"
# CUDA_VISIBLE_DEVICES=0 python evaluation.py --model='yolov2' --patch="./exp/exp71/generated/generated-images-0750.png"
# CUDA_VISIBLE_DEVICES=0 python evaluation.py --model='yolov3' --patch="./exp/exp71/generated/generated-images-0750.png" --tiny
# CUDA_VISIBLE_DEVICES=0 python evaluation.py --model='yolov3' --patch="./exp/exp71/generated/generated-images-0750.png"
# CUDA_VISIBLE_DEVICES=0 python evaluation.py --model='yolov4' --patch="./exp/exp71/generated/generated-images-0750.png" --tiny
# CUDA_VISIBLE_DEVICES=0 python evaluation.py --model='yolov4' --patch="./exp/exp71/generated/generated-images-0750.png"


# # # 
# CUDA_VISIBLE_DEVICES=9 python evaluation.py --model='yolov2' --patch="./patch/bg76.png"
# CUDA_VISIBLE_DEVICES=9 python evaluation.py --model='yolov3' --patch="./patch/bg76.png" --tiny
# CUDA_VISIBLE_DEVICES=9 python evaluation.py --model='yolov3' --patch="./patch/bg76.png"
# CUDA_VISIBLE_DEVICES=9 python evaluation.py --model='yolov4' --patch="./patch/bg76.png" --tiny
# CUDA_VISIBLE_DEVICES=9 python evaluation.py --model='yolov4' --patch="./patch/bg76.png"

# CUDA_VISIBLE_DEVICES=9 python evaluation.py --model='yolov2' --patch="./patch/generated-images-0354.png"
# CUDA_VISIBLE_DEVICES=9 python evaluation.py --model='yolov3' --patch="./patch/generated-images-0354.png" --tiny
# CUDA_VISIBLE_DEVICES=9 python evaluation.py --model='yolov3' --patch="./patch/generated-images-0354.png"
# CUDA_VISIBLE_DEVICES=9 python evaluation.py --model='yolov4' --patch="./patch/generated-images-0354.png" --tiny
# CUDA_VISIBLE_DEVICES=9 python evaluation.py --model='yolov4' --patch="./patch/generated-images-0354.png"

CUDA_VISIBLE_DEVICES=9 python evaluation.py --model='yolov4' --tiny --patch="./exp/exp74/generated/generated-images-1000.png"
CUDA_VISIBLE_DEVICES=9 python evaluation.py --model='yolov4' --tiny --patch="./exp/exp112/generated/generated-images-1000.png"

# CUDA_VISIBLE_DEVICES=9 python evaluation.py --model='yolov4' --tiny --patch="./exp/exp108/generated/generated-images-1000.png"
# CUDA_VISIBLE_DEVICES=9 python evaluation.py --model='yolov4' --tiny --patch="./exp/exp109/generated/generated-images-1000.png"
# CUDA_VISIBLE_DEVICES=9 python evaluation.py --model='yolov4' --tiny --patch="./exp/exp110/generated/generated-images-1000.png"
# CUDA_VISIBLE_DEVICES=9 python evaluation.py --model='yolov4' --tiny --patch="./exp/exp111/generated/generated-images-1000.png"
# CUDA_VISIBLE_DEVICES=9 python evaluation.py --model='yolov4' --tiny --patch="./exp/exp112/generated/generated-images-1000.png"

