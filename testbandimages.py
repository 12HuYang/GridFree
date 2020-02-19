import numpy as np
import cv2

colortable=np.array([[255,0,0],[0,255,0],[0,0,255]],'uint8')  #red,green,blue
imagefile=''
filesrc=cv2.imread(imagefile,flags=cv2.IMREAD_ANYCOLOR)
h,w,c=np.shape(filesrc)
rgbfile=cv2.cvtColor(filesrc,cv2.COLOR_BGR2RGB)
grayfile=cv2.cvtColor(filesrc,cv2.COLOR_BGR2GRAY)
filesize=(h,w)
rband=np.zeros(filesrc.shape)
gband=np.zeros(filesrc.shape)
bband=np.zeros(filesrc.shape)
grayband=np.zeros(filesize)






