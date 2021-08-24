#%%
from PIL import Image


fake_images_path           = "../adversarial-attack-ensemble/exp/exp52/generated/generated-images-0600.png"
img = Image.open(fake_images_path)
(w, h) = img.size
print('w=%d, h=%d', w, h)
img.show()

#%%
new_img = img.resize((w, int(h*1.5)),resample=Image.BILINEAR)
new_img.show()
new_img.save("ttt.png")