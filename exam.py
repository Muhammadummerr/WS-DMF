import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure


def components(binary_mask,number):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=number)


    num_objects = num_labels - 1  
    total_object_pixels = 0
    # Display the pixel count for each object
    print(f"Total number of objects: {num_objects}")
    for i in range(1, num_labels):  # Skip the background (label 0)
        object_area = stats[i, cv2.CC_STAT_AREA]  # Pixel count for object i
        print(f"Object {i}: {object_area} pixels")
        total_object_pixels+=object_area


    print(f"Total number of {number}-connected objects: {num_objects}")
    print(f"Total number of object pixels: {total_object_pixels}")


def convert_8_to_4_connected(binary_image):
    # Ensure binary input
    binary_image = (binary_image > 0).astype(np.uint8)
    
    # Create a copy of the image to modify
    converted_image = binary_image.copy()
    
    # Iterate through the image to detect and fix diagonal connections
    for y in range(1, binary_image.shape[0] - 1):
        for x in range(1, binary_image.shape[1] - 1):
            if binary_image[y, x] == 1:  # Only check for object pixels
                # Check for diagonal connections and add bridging pixels
                if binary_image[y - 1, x - 1] == 1 and binary_image[y - 1, x] == 0 and binary_image[y, x - 1] == 0:
                    converted_image[y - 1, x] = 1
                    converted_image[y, x - 1] = 1
                if binary_image[y - 1, x + 1] == 1 and binary_image[y - 1, x] == 0 and binary_image[y, x + 1] == 0:
                    converted_image[y - 1, x] = 1
                    converted_image[y, x + 1] = 1
                if binary_image[y + 1, x - 1] == 1 and binary_image[y + 1, x] == 0 and binary_image[y, x - 1] == 0:
                    converted_image[y + 1, x] = 1
                    converted_image[y, x - 1] = 1
                if binary_image[y + 1, x + 1] == 1 and binary_image[y + 1, x] == 0 and binary_image[y, x + 1] == 0:
                    converted_image[y + 1, x] = 1
                    converted_image[y, x + 1] = 1
    
    # Label the connected components in the converted image
    labeled_image, object_count = measure.label(converted_image, connectivity=1, return_num=True)
    
    # Calculate the total number of object pixels
    total_object_pixels = np.sum(converted_image)
    
    return converted_image, object_count, total_object_pixels



img = cv2.imread('/home/umerfarooq/WS-DMF/PracticalExamImage.bmp', cv2.IMREAD_ANYCOLOR)


# Task1
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

task1img = np.where(gray_image != 255, 1, 0).astype(np.uint8)

plt.figure(figsize=(6,6))
plt.imshow(task1img, cmap='gray')
plt.title('Grayscale image(TASK 1)')
plt.axis('off')
plt.show()

components(task1img,8)

#TASK 2
bianrizedimg = task1img.copy()
components(bianrizedimg,8)


#TASK 3
binary_for_thinning = (task1img * 255).astype(np.uint8)

thinned = cv2.ximgproc.thinning(binary_for_thinning)
plt.imshow(thinned,cmap='gray')
plt.title("Thinned image")
plt.axis('off')
plt.show()


#TASK 4

print("THINNED IMAGE (8-CONNECTIVITY)")
components(thinned,8)

#TASK 5
print("THINNED IMAGE (4-CONNECTIVITY)")
components(thinned,4)

fig,ax = plt.subplots(1,2)
#TASK 6
converted, count, pixels = convert_8_to_4_connected(task1img)
print("Original Image:")
ax[0].imshow(task1img,cmap='gray')
ax[0].set_title("Original")
print("\nConverted Image:")
ax[1].imshow(converted,cmap='gray')
ax[1].set_title("TASK 6 4 connectivity same objects")
plt.show()
print(f"Object Count: {count}")
print(f"Total Object Pixels for 4-connectivity: {pixels}\n")
