import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import glob
import matplotlib.image as mpimg
import imageio
from ipdb import set_trace as st 
import tqdm
import matplotlib
from PIL import Image, ImageDraw, ImageFont

matplotlib.use('Agg')

exp = 'exp73'
img_list = glob.glob(f'../adversarial-attack-ensemble/exp/{exp}/generated/generated-images-*.png')

# font = ImageFont.truetype("sans-serif.ttf", 16)
fig = plt.figure()
ims = []
Myfont  = ImageFont.truetype('./cmb10.ttf',size=20)
(x, y) = (5, 5) # Text position
color  = 'rgb(0, 0, 0)' # Text color
for ii,i in enumerate(img_list[::4]):
    Srcimg  = Image.open(i)
    Drawimg = ImageDraw.Draw(Srcimg)
    Mytext = str(ii)
    Drawimg.text((x, y), Mytext, fill=color,font=Myfont)
    Srcimg = np.array(Srcimg)
    ims.append(Srcimg)
# ani = animation.ArtistAnimation(fig, ims, interval=200, repeat_delay=1000)
# ani.save("test.gif",writer='pillow')
imageio.mimsave(f'./gif/{exp}.gif', ims, duration = 0.01)
print(f"finished: {exp}")
