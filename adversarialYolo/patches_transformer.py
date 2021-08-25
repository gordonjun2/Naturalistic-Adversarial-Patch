import cv2
import numpy as np

img_path = "saved_patches/patch11.jpg"
img_saved = "saved_patches/patch11_test.jpg"
img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
print('Original Dimensions : ',img.shape)
 
scale_percent = 25 # percent of original size
x_offset=0
y_offset=0

width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
# new image
img_new = np.zeros([img.shape[1], img.shape[0], 3], dtype=np.uint8)
img_new[y_offset:y_offset+resized.shape[0], x_offset:x_offset+resized.shape[1]] = resized


print('resized Dimensions : ',resized.shape)
print('img_new Dimensions : ',img_new.shape)
 
cv2.imshow("img_new image", img_new)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite(img_saved, img_new)
print("saved at "+str(img_saved))