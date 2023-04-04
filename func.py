from math import ceil
import numpy as np
import cv2


def crop_hand(img, hands, member):
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
            imgresizeshape = imgresize.shape
            imgwhite[:, wgap:imgresizeshape[1]+wgap] = imgresize
            
        else:
            k = imgsize/w_
            hc = ceil(k*h_)
            hgap = ceil((imgsize-hc)/2)
            imgresize = cv2.resize(imgcrop, (imgsize, hc))
            imgresizeshape = imgresize.shape
            imgwhite[hgap:imgresizeshape[0]+hgap, :] = imgresize
        return imgwhite