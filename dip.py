import cv2
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
import numpy as np
img = cv2.imread("/home/umerfarooq/WS-DMF/0.jpg",cv2.IMREAD_GRAYSCALE)
fig,axs = plt.subplots(1,5,figsize=(15,15))
axs[0].imshow(img,cmap='gray')
axs[0].set_title("Original")
axs[0].axis('off')
_,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
thinned = cv2.ximgproc.thinning(img)
axs[1].imshow(thinned,cmap='gray')
axs[1].set_title("Thinned")
axs[1].axis('off')
# img = img/255.0
skeleton = skeletonize(img)
skeleton = (skeleton * 255).astype(np.uint8)
axs[2].imshow(skeleton,cmap='gray')
axs[2].set_title("Skeletonized")
axs[2].axis('off')

dif = thinned-skeleton
axs[3].imshow(dif,cmap='gray')
axs[3].set_title("thinned-skeleton")
axs[3].axis('off')

dif = skeleton-thinned
axs[4].imshow(dif,cmap='gray')
axs[4].set_title("skeleton-thinned")
axs[4].axis('off')
plt.show()

img2 = cv2.imread("man.png",cv2.IMREAD_GRAYSCALE)
_,binary = cv2.threshold(img2,100,255,cv2.THRESH_BINARY)

plt.imshow(binary,cmap='gray')
plt.show()
erode = cv2.erode(binary,(3,3))
boundary = binary-erode
_,binary2 = cv2.threshold(boundary,100,255,cv2.THRESH_BINARY)
plt.imshow(binary2,cmap='gray')
plt.show()