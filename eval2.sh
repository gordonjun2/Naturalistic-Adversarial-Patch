#!/bin/bash
# CUDA_VISIBLE_DEVICES=8 python evaluation.py --model='yolov2' --patch="./exp/exp29/generated/generated-images-1000.png"
# CUDA_VISIBLE_DEVICES=8 python evaluation.py --model='yolov3' --patch="./exp/exp29/generated/generated-images-1000.png"
# CUDA_VISIBLE_DEVICES=8 python evaluation.py --model='yolov3' --patch="./exp/exp29/generated/generated-images-1000.png" --tiny
# CUDA_VISIBLE_DEVICES=8 python evaluation.py --model='yolov4' --patch="./exp/exp29/generated/generated-images-1000.png"
# CUDA_VISIBLE_DEVICES=8 python evaluation.py --model='yolov4' --patch="./exp/exp29/generated/generated-images-1000.png" --tiny


# CUDA_VISIBLE_DEVICES=8 python evaluation.py --model='yolov2' --patch="./exp/exp56/generated/generated-images-1000.png"
# CUDA_VISIBLE_DEVICES=8 python evaluation.py --model='yolov3' --patch="./exp/exp56/generated/generated-images-1000.png" --tiny
# CUDA_VISIBLE_DEVICES=8 python evaluation.py --model='yolov3' --patch="./exp/exp56/generated/generated-images-1000.png"
# CUDA_VISIBLE_DEVICES=8 python evaluation.py --model='yolov4' --patch="./exp/exp56/generated/generated-images-1000.png" --tiny
# CUDA_VISIBLE_DEVICES=8 python evaluation.py --model='yolov4' --patch="./exp/exp56/generated/generated-images-1000.png"

# for ((i=80;i<=85;i++))
# do
#   tt="./exp/exp$i/generated/generated-images-1000.png"
#   echo $tt
#   CUDA_VISIBLE_DEVICES=8 python evaluation.py --model='yolov2' --patch=$tt
#   CUDA_VISIBLE_DEVICES=8 python evaluation.py --model='yolov3' --patch=$tt --tiny
#   CUDA_VISIBLE_DEVICES=8 python evaluation.py --model='yolov3' --patch=$tt
#   CUDA_VISIBLE_DEVICES=8 python evaluation.py --model='yolov4' --patch=$tt --tiny
#   CUDA_VISIBLE_DEVICES=8 python evaluation.py --model='yolov4' --patch=$tt
# done

for ((i=80;i<=85;i++))
do
  tt="./exp/exp$i/generated/generated-images-1000.png"
  echo $tt
  CUDA_VISIBLE_DEVICES=8 python evaluation.py --model='yolov4' --patch=$tt --tiny
done
