# packages
import numpy as np
import cv2
cv2.saliency
from matplotlib import pyplot as plt
import os
import segmentImage
import SaliencyRC
import random
from sklearn.cluster import MiniBatchKMeans
import time
import random
import glob
import matplotlib.pyplot as plt
from random import sample

# change directory to the file where images are outputted
os.getcwd()
os.chdir('/Users/William/Documents/Python Projects/ML')
os.chdir('/Users/William/Documents/Science Research 3/Cam Images')

# webcam
cam = cv2.VideoCapture(0)
cv2.namedWindow("Camera of Video Processing Unit")


img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("Camera of Video Processing Unit", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed (exits the video capture UI)
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed (captures image)
        img_name = "VPU_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

DATA_DIR = '/Users/William/Documents/Science Research 3/Cam Images' #original images
DATA_DIR_Final = '/Users/William/Documents/Science Research 3/Cam Images/Processed Cam Images/' #processed images

im_type ='VPU'

# looping through all images in the folder Cam Images
input_images = []
for img in glob.glob("/Users/William/Documents/Science Research 3/Cam Images/*.png"):
    n= cv2.imread(img)
    input_images.append(n)

num_images = len(input_images)

# salient object detection
for i in range(0, num_images):
    def test_segmentation():
        img3f = input_images[i]
        img3f = img3f.astype(np.float32)
        img3f *= 1. / 255
        imgLab3f = cv2.cvtColor(img3f,cv2.COLOR_BGR2Lab)
        num,imgInd = segmentImage.SegmentImage(imgLab3f,None,0.5,200,50)

        colors = [[random.randint(0,255),random.randint(0,255),random.randint(0,255)] for _ in range(num)]
        showImg = np.zeros(img3f.shape,dtype=np.int8)
        height = imgInd.shape[0]
        width = imgInd.shape[1]
        for y in range(height):
            for x in range(width):
                if imgInd[y,x].all() > 0:
                    showImg[y,x] = colors[imgInd[y,x] % num]
    def test_rc_map():
        img3i = input_images[i]
        img3f = img3i.astype(np.float32)
        img3f *= 1. / 255
        start = cv2.getTickCount()
        sal = SaliencyRC.GetHC(img3f)
        end = cv2.getTickCount()
        #print((end - start)/cv2.getTickFrequency())
        np.save("sal.npy",sal)
        idxs = np.where(sal < (sal.max()+sal.min()) / 1.8)
        img3i[idxs] = 0
        sal = sal * 255
        sal = sal.astype(np.int16)
        cv2.namedWindow("sb")
        cv2.moveWindow("sb",20,20)
        #cv2.imwrite(os.path.splitext(input_images[i])[0]+'_saliencyMap.jpg', sal)
        cv2.imwrite('saliencyMap.jpg', sal)

    test_rc_map()

    #change the saliency map to color
    background = cv2.cvtColor(cv2.imread('saliencyMap.jpg'), cv2.COLOR_BGR2RGB)
    overlay = cv2.cvtColor(input_images[i], cv2.COLOR_BGR2RGB)

    # masking the image background by overlaying the saliency map onto the original image
    lower_white = np.array([220, 220, 220], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(cv2.bitwise_not(background), lower_white, upper_white)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (
    3, 3)))  # "erase" the small white points in the resulting mask
    mask2 = cv2.bitwise_not(mask)
    bk = np.full(background.shape, 255, dtype=np.uint8)  # white bk
    bk_masked = cv2.bitwise_and(bk, bk, mask=mask) # get masked background, mask must be inverted
    fg_masked = cv2.bitwise_and(overlay, overlay, mask=mask2)
    merged = cv2.bitwise_or(fg_masked, bk_masked)

    # edge detection
    edges = cv2.Canny(overlay, 100, 200)
    cv2.imwrite('edges.jpg', edges)
    cv2.imwrite('edges.jpg', cv2.bitwise_not(edges))
    edges = cv2.cvtColor(cv2.imread('edges.jpg'), cv2.COLOR_BGR2RGB)

    img_final = cv2.addWeighted(merged, 0.9, edges, 0.1, 0)
    img_final = cv2.cvtColor(img_final, cv2.COLOR_RGB2BGR)
    cv2.imwrite('final.jpg', img_final)

    # color clustering using kmeans
    img_kmeans = cv2.imread('final.jpg')
    (h, w) = img_kmeans.shape[:2]
    img_kmeans = cv2.cvtColor(img_kmeans, cv2.COLOR_BGR2YCR_CB)
    img_kmeans = img.reshape((img_kmeans.shape[0] * img_kmeans.shape[1], 3))
    clt = MiniBatchKMeans(n_clusters = 8)
    labels = clt.fit_predict(img_kmeans)
    quant = clt.cluster_centers_.astype("uint8")[labels]
    quant = quant.reshape((h, w, 3))
    img_kmeans = img_kmeans.reshape((h, w, 3))
    img_kmeans = cv2.cvtColor(img_kmeans, cv2.COLOR_YCR_CB2BGR)
    cv2.imwrite("kmeansYCBCR.jpg", quant)

    # image downsampling
    img = cv2.imread("kmeansYCBCR.jpg")
    bicubic_img = cv2.resize(img,dsize=(100, 100), interpolation = cv2.INTER_CUBIC)

    cv2.imwrite(DATA_DIR_Final+im_type+str(i)+'_final_100.jpg', bicubic_img)