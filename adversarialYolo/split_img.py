from PIL import Image, ImageDraw
import numpy as np
import os

savedir = "saved_patches/20200821_21_1g_4l_paper_obj/"
patch_name = "patch_30"
patchfile = savedir + patch_name + ".jpg"
patch_img = Image.open(patchfile).convert('RGB')
w, h = patch_img.size
if(w == h):
    crop_rectangle = (0, 0, int(w/2), int(w/2))
    pacth_1 = patch_img.crop(crop_rectangle)
    crop_rectangle = (int(w/2), 0, int(w), int(w/2))
    pacth_2 = patch_img.crop(crop_rectangle)
    crop_rectangle = (0, int(w/2), int(w/2), int(w))
    pacth_3 = patch_img.crop(crop_rectangle)
    crop_rectangle = (int(w/2), int(w/2), int(w), int(w))
    pacth_4 = patch_img.crop(crop_rectangle)
# pacth_1.show()
# pacth_2.show()
# pacth_3.show()
# pacth_4.show()
pacth_1.save(os.path.join(savedir, patch_name + "_1.jpg"))
pacth_2.save(os.path.join(savedir, patch_name + "_2.jpg"))
pacth_3.save(os.path.join(savedir, patch_name + "_3.jpg"))
pacth_4.save(os.path.join(savedir, patch_name + "_4.jpg"))
