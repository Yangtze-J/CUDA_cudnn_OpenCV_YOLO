import cv2
import glob
import os
import pandas as pd
import numpy as np

def scaleRadius(img,scale):
    x=img[int(img.shape[0]/2),:,:].sum(1)
    rx = (x > x.mean()/10).sum()/2
    my = img[:,img.shape[1] // 2,:].sum(1)
    ry = (my > my.mean() / 10).sum() / 2
    s=scale*1.0/max(rx,ry)
    return cv2.resize(img,(0,0),fx=s,fy=s)

scale = 300

for f in glob.glob("../train/*.jpeg"):
    a=cv2.imread(f)
    a=scaleRadius(a,scale)
    b=np.zeros(a.shape)
    cv2.circle(b,(int(a.shape[1]/2),int(a.shape[0]/2)),int(scale*0.95),(1,1,1),-1,8,0)
    aa=cv2.addWeighted(a,4,cv2.GaussianBlur(a,(0,0),6),-4,128)*b+128*(1-b)
    w= aa.shape[1]
    w_margin = int(w/10)
    aa=aa[:, w_margin:w-w_margin]
    image_resize=cv2.resize(aa,(512,512))
    cv2.imwrite("./train/"+f, image_resize)



