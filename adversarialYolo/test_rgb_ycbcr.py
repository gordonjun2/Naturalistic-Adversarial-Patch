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
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage, Compose
from imageaug.transforms import Colorspace, RandomAdjustment, RandomRotatedCrop

def yCbCr2rgb(input_im):
    im_flat = input_im.contiguous().view(-1, 3).float().cuda()
    mat = torch.tensor([[1.164, 1.164, 1.164],
                       [0, -0.392, 2.017],
                       [1.596, -0.813, 0]]).cuda()
    bias = torch.tensor([-222.921/255.0, 135.576/255.0, -276.836/255.0]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
    out = temp.view(3, list(input_im.size())[1], list(input_im.size())[2])
    return out

def rgb2yCbCr(input_im):
    # print("input_im size: "+str(input_im.size()))
    # print("input_im : "+str(input_im.data))
    # input_im.shape: (3, 300,300)   range: [0,1]
    im_flat = input_im.contiguous().view(-1, 3).float().cuda()
    mat = torch.tensor([[0.257, -0.148, 0.439],
                        [0.504, -0.291, -0.368],
                        [0.098, 0.439, -0.071]]).cuda()
    bias = torch.tensor([16.0/255.0, 128.0/255.0, 128.0/255.0]).cuda()
    temp = im_flat.mm(mat).cuda() + bias
    out = temp.view(3, input_im.shape[1], input_im.shape[2])
    return out


# patch_size = 300

# patchfile = "saved_patches/patch11.jpg"
# test_img  = Image.open(patchfile)
# img_rgb   = test_img.convert('RGB')
# img_ycbcr = test_img.convert('YCbCr')

# tf_resize   = transforms.Resize((patch_size,patch_size))
# tf_totensor = transforms.ToTensor()

# _img_rgb = tf_resize(img_rgb)
# rgb_cpu  = tf_totensor(_img_rgb)
# print("rgb_cpu size : "+str(rgb_cpu.size()))
# rgb_gpu  = rgb_cpu.cuda()
# print("rgb_gpu size : "+str(rgb_gpu.size()))

# _img_ycbcr = tf_resize(img_ycbcr)
# ycbcr_cpu  = tf_totensor(_img_ycbcr)
# ycbcr_gpu  = ycbcr_cpu.cuda()


# #
# img = rgb_gpu.detach().cpu()
# print(img.data[:,0:2,0:2])
# img = transforms.ToPILImage()(img)
# plt.imshow(img)
# plt.title("rgb")
# # plt.show()  ## --------------------------------- ori
# print('rgb_gpu szie : '+str(img.size))
# rgb_np_test = np.ndarray((img.size[0], img.size[1], 3), 'u1', img.tobytes())
# print("rgb_np_test size : "+str(rgb_np_test.shape))
# # Image.fromarray(rgb_np_test, "RGB").show()
# #
# img = ycbcr_gpu.detach().cpu()
# # print(img.data[:,0:2,0:2])
# img = transforms.ToPILImage()(img)
# # plt.imshow(img)
# # plt.title("ycbcr")
# # plt.show()
# ycbcr_np_test = np.ndarray((img.size[0], img.size[1], 3), 'u1', img.tobytes())
# # Image.fromarray(ycbcr_np_test, "YCbCr").show()


# print("-------------------------------------------------------------")

# f = Colorspace("rgb", "yuv")
# f2 = Colorspace("yuv", "rgb")
# b = f(rgb_gpu).detach().cpu()
# # print(b.data[:,0:2,0:2])
# c = f2(b)
# print(c.data[:,0:2,0:2])
# print("b size : "+str(b.size()))
# img = transforms.ToPILImage()(b)
# # print("img size : "+str(img.size))
# ycbcr_np_test2 = np.ndarray((img.size[0], img.size[1], 3), 'u1', img.tobytes())
# # Image.fromarray(ycbcr_np_test2, "YCbCr").show()
# img = transforms.ToPILImage()(c)
# rgb_np_test2 = np.ndarray((img.size[0], img.size[1], 3), 'u1', img.tobytes())
# # Image.fromarray(rgb_np_test2, "RGB").show()

# # print(rgb_cpu[0,:,:].unsqueeze(0).size())
# y = ycbcr_cpu[0,:,:].unsqueeze(0)
# # Image.fromarray(np.array(y[0,:,:]), "L").show()  # error
# u = torch.zeros(y.size()) + 0.2
# v = torch.zeros(y.size())
# result_yuv = torch.cat((y,u,v),0)
# print(result_yuv.size())
# result_rgb = Colorspace("yuv", "rgb")(result_yuv)
# img = transforms.ToPILImage()(result_rgb)
# result_rgb_np = np.ndarray((img.size[0], img.size[1], 3), 'u1', img.tobytes())
# Image.fromarray(np.array(result_rgb_np), "RGB").show()



### TODO color ###
# y = torch.clamp(torch.randn(1,300,300), 0.000001, 0.99999)
# print("y      : "+str(y.data))
# print("y size : "+str(y.size()))
# u = torch.zeros(y.size()) + 0.3
# v = torch.zeros(y.size()) + 0.3
# img_yuv = torch.cat((y,u,v),0)
# result_rgb = Colorspace("yuv", "rgb")(img_yuv)
# # show
# img = transforms.ToPILImage()(result_rgb)
# result_rgb_np = np.ndarray((img.size[0], img.size[1], 3), 'u1', img.tobytes())
# Image.fromarray(np.array(result_rgb_np), "RGB").show()

# rgb
r = torch.zeros(1,300,300) + 0.0
g = torch.zeros(r.size())  + 0.0
b = torch.zeros(r.size())  + 0.0
img_rgb = torch.cat((r,g,b),0)
# show
img = transforms.ToPILImage()(img_rgb)
print("img size : "+str(img.size))
result_rgb_np = np.ndarray((img.size[0], img.size[1], 3), 'u1', img.tobytes())
print("result_rgb_np size : "+str(result_rgb_np.shape))
Image.fromarray(np.array(result_rgb_np), "RGB").show()
#
result_yuv = Colorspace("rgb", "yuv")(img_rgb)
print("result_yuv size : "+str(result_yuv.size()))
print("result_yuv      : "+str(result_yuv.data))
print("y : "+str(result_yuv.data[0]))
print("u : "+str(result_yuv.data[1]))
print("v : "+str(result_yuv.data[2]))




