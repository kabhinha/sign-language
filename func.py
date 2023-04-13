from math import ceil
import numpy as np
import classifier as cf
import cv2
import time


def crop_hand(img, hands, member, winname):
    imgsize = 300
    hand_member = hands[member]
    x_, y_, w_, h_ = hand_member["bbox"]
    x,y, _ = img.shape
    if (x_+1<x and y_+1<y) and (x_+1+w_<x and y_+h_+1<y) and ((x_+1>0 and y_+1>0) and (x_+1+w_>0 and y_+h_+1>0)):
        imgcrop = img[y_:y_+h_,x_:x_+w_]
        imgwhite = np.ones((imgsize, imgsize, 3), np.uint8)*255
        ar = h_/w_
        if ar>1:
            k = imgsize/h_
            wc = ceil(k*w_)
            wgap = ceil((imgsize-wc)/2)
            imgresize = cv2.resize(imgcrop, (wc, imgsize))
            imgwhite[:, wgap:wc+wgap] = imgresize
            
        else:
            k = imgsize/w_
            hc = ceil(k*h_)
            hgap = ceil((imgsize-hc)/2)
            imgresize = cv2.resize(imgcrop, (imgsize, hc))
            imgwhite[hgap:hc+hgap, :] = imgresize
        cv2.imshow(winname, imgwhite)
        key = cv2.waitKey(1)
        if key==ord("s"):
            cv2.imwrite(fr"DATA/ASL/F/IMG{time.time()}.jpg", imgwhite)
        return imgwhite

def img_text(img):
    modelPath= r"Model\keras_Model.h5"
    labelsPath = r"Model\labels.txt"
    model, labels = cf.load_modell(modelPath, labelsPath)
    results = cf.getPrediction(img, model, labels)
 #   if results:
#        time.sleep(0.001)
    print(results)