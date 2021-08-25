import numpy as np
from PIL import Image, ImageDraw

img2 = np.array(Image.open("./patch.jpg").convert('RGB'))
img3 = np.array(Image.open("./predictions.jpg").convert('RGB'))
img = np.array(Image.open("./DNM65.png").convert('RGB'))

# """
# numpy ver.
# """
# import cv2
# import matplotlib.pyplot as plt

# print("img size : "+str(img.shape))
# # plt.imshow(img)
# # plt.show()

# average = img.mean(axis=0).mean(axis=0)
# print("average : "+str(average/255.0))

# pixels = np.float32(img.reshape(-1, 3))
# print("pixels size : "+str(pixels.shape))

# n_colors = 5
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
# print("criteria : "+str(criteria))
# flags = cv2.KMEANS_RANDOM_CENTERS
# print("flags : "+str(flags))

# _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
# print("labels "+str(labels.shape)+" : "+str(labels))
# print("palette "+str(palette.shape)+" : \n"+str(palette/255.0))
# _, counts = np.unique(labels, return_counts=True)
# print("counts "+str(counts.shape)+" : "+str(counts))

# dominant = palette[np.argmax(counts)]
# print("dominant "+str(dominant.shape)+" : "+str(dominant/255.0))

# avg_patch = np.ones(shape=img.shape, dtype=np.uint8)*np.uint8(average)
# print("avg_patch "+str(avg_patch.shape))

# print("*********counts "+str(counts.shape))
# print("*********avg_patch "+str(avg_patch.shape))
# print("*********palette "+str(palette.shape))

# indices = np.argsort(counts)[::-1]
# print("indices "+str(indices.shape)+" : "+str(indices))
# freqs = np.cumsum(np.hstack([[0], counts[indices]/counts.sum()]))
# print("freqs "+str(freqs.shape)+" : "+str(freqs))
# rows = np.int_(img.shape[0]*freqs)

# dom_patch = np.zeros(shape=img.shape, dtype=np.uint8)
# for i in range(len(rows) - 1):
#     dom_patch[rows[i]:rows[i + 1], :, :] += np.uint8(palette[indices[i]])

# plt.ion()
# plt.figure()
# plt.imshow(avg_patch)
# plt.figure()
# plt.imshow(dom_patch)
# plt.ioff()
# plt.show()

# print()


"""
pytorch ver.
"""
import time
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import torch.nn.functional as F
from kmeans_pytorch import kmeans

tf = transforms.ToTensor()
img_tensor = tf(img)
# print("img size : "+str(img_tensor.size()))
# img_numpy = np.array(img_tensor).transpose((1, 2, 0))
# print("img_numpy size : "+str(img_numpy.shape))
# plt.imshow(img_numpy)
# plt.show()

# average = torch.mean(torch.mean(img_tensor, dim=1), dim=1)
# print("average : "+str(average))

# ones = torch.ones(size=img_tensor.size()).float()
# ones = ones.permute(1,2,0)
# avg_patch = ones * average
# print("avg_patch "+str(avg_patch.size()))

# ####################   How to use   ###########################
# from split_color import split_color
# palette, counts = split_color(img_tensor, n_colors=5)
# ###############################################################

# # to numpy (display)
# counts = np.array(counts)
# avg_patch = np.array(avg_patch)
# palette = np.array(palette) * 255.0
# print("*********counts "+str(counts.shape))
# print("*********avg_patch "+str(avg_patch.shape))
# print("*********palette "+str(palette.shape))

# freqs = np.cumsum(np.hstack([[0], counts/counts.sum()]))
# print("freqs "+str(freqs.shape)+" : "+str(freqs))
# rows = np.int_(img.shape[0]*freqs)

# print("palette:\n"+str(palette))

# dom_patch = np.zeros(shape=img.shape, dtype=np.uint8)
# for i in range(len(rows) - 1):
#     dom_patch[rows[i]:rows[i + 1], :, :] += np.uint8(palette[i])

# plt.ion()
# plt.figure()
# plt.imshow(avg_patch)
# plt.figure()
# plt.imshow(dom_patch)
# plt.ioff()
# plt.show()


### create (b,f,3,w,h) ###
tf = transforms.ToTensor()
img_tensor_ = tf(img)
img2_tensor = F.interpolate(tf(img3).unsqueeze(0), size=416)     ## torch.Size([1, 3, 416, 416])
img_tensor  = F.interpolate(tf(img ).unsqueeze(0), size=416)     ## torch.Size([1, 3, 416, 416])
test_data = torch.cat((img2_tensor, img_tensor), 0).unsqueeze(0) ## torch.Size([1, 2, 3, 416, 416])
# test_data = img_tensor_.unsqueeze(0).unsqueeze(0)
###############  (b,f,3,w,h)  How to use   ####################
from split_color import split_color
bf_palette, bf_counts = split_color(test_data, n_colors=5)
###############################################################
# ##############    (3,w,h)  How to use   #######################
# from split_color import split_color
# bf_palette, bf_counts = split_color(img_tensor_, n_colors=5)
# bf_palette = bf_palette.unsqueeze(0).unsqueeze(0)
# bf_counts  = bf_counts.unsqueeze(0).unsqueeze(0)
# ###############################################################
plt.ion()
b,f = bf_palette.size()[0:2]
for bi in range(b):
    for fi in range(f):
        counts  = bf_counts[bi][fi]
        palette = bf_palette[bi][fi]
        ## to numpy (display)
        counts = np.array(counts)
        palette = np.array(palette) * 255.0
        #
        freqs = np.cumsum(np.hstack([[0], counts/counts.sum()]))
        rows = np.int_(img.shape[0]*freqs)
        #
        dom_patch = np.zeros(shape=img.shape, dtype=np.uint8)
        for i in range(len(rows) - 1):
            dom_patch[rows[i]:rows[i + 1], :, :] += np.uint8(palette[i])
        #
        plt.figure()
        plt.imshow(dom_patch)
plt.ioff()
plt.show()