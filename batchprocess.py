#from tkinter import *
#from tkinter import ttk
import tkinter.filedialog as filedialog
from tkinter import messagebox

from PIL import Image,ImageDraw,ImageFont
from PIL import ImageTk,ImageGrab
import cv2
from skimage import filters
import matplotlib.pyplot as pyplt
import numpy as np

from sklearn.cluster import KMeans
import tkintercorestat
import tkintercore
import cal_kernelsize

import os
import csv
import scipy.linalg as la
import multiprocessing
import time
#from multiprocessing import Process

batch_colorbandtable=np.array([[255,0,0],[255,127,0],[255,255,0],[127,255,0],[0,255,255],[0,127,255],[0,0,255],[127,0,255],[75,0,130],[255,0,255]],'uint8')


class batch_img():
    def __init__(self,size,bands):
        self.size=size
        self.bands=bands

class batch_ser_func():
    def __init__(self,filename):
        self.file=filename
        self.folder=FOLDER
        self.exportpath=exportpath
        self.batch_Multiimage={}
        self.batch_Multigray={}
        self.batch_Multitype={}
        self.batch_Multiimagebands={}
        self.batch_Multigraybands={}
        self.batch_displaybandarray={}
        self.batch_originbandarray={}
        self.batch_originpcabands={}
        self.batch_colordicesband={}
        self.batch_results={}
        self.kernersizes={}
        self.reseglabels=None
        self.displayfea_l=0
        self.displayfea_w=0
        self.RGB_vector=None
        self.colorindex_vector=None
        self.displaypclagels=None
        self.needswitchkmeanssel=False
        self.partialpca=False
        self.drawpolygon=False
        self.filtercoord=[]
        self.filterbackground=[]
        self.nonzero_vector=[]


    def Open_batchimage(self):
        try:
            Filersc=cv2.imread(self.folder+'/'+self.file,flags=cv2.IMREAD_ANYCOLOR)
            height,width,channel=np.shape(Filersc)
            Filesize=(height,width)
            print('filesize:',height,width)
            RGBfile=cv2.cvtColor(Filersc,cv2.COLOR_BGR2RGB)
            Grayfile=cv2.cvtColor(Filersc,cv2.COLOR_BGR2Lab)
            Grayfile=cv2.cvtColor(Grayfile,cv2.COLOR_BGR2GRAY)
            Grayimg=batch_img(Filesize,Grayfile)
            RGBbands=np.zeros((channel,height,width))
            for j in range(channel):
                band=RGBfile[:,:,j]
                band=np.where(band==0,1e-6,band)
                RGBbands[j,:,:]=band
            RGBimg=batch_img(Filesize,RGBbands)
            tempdict={self.file:RGBimg}
            self.batch_Multiimagebands.update(tempdict)
            tempdict={self.file:Grayfile}
            self.batch_Multigray.update(tempdict)
            tempdict={self.file:0}
            self.batch_Multitype.update(tempdict)
            tempdict={self.file:Grayimg}
            self.batch_Multigraybands.update(tempdict)
            if len(filtercoord)>0:
                self.partialpca=True
                self.drawpolygon=drawpolygon
                self.filtercoord=filtercoord.copy()
                self.filterbackground=filterbackground.copy()

        except:
            # messagebox.showerror('Invalid Image Format','Cannot open '+filename)
            return False
        return True

    def fillbands(self,originbands,displaybands,vector,vectorindex,name,band,filter=0):
        tempdict={name:band}
        if isinstance(filter, int):
            if name not in originbands:
                originbands.update(tempdict)
                image = cv2.resize(band, (self.displayfea_w, self.displayfea_l), interpolation=cv2.INTER_LINEAR)
                displaydict = {name: image}
                displaybands.update(displaydict)
                fea_bands = image.reshape((self.displayfea_l * self.displayfea_w), 1)[:, 0]
                vector[:, vectorindex] = vector[:, vectorindex] + fea_bands
        else:
            if name not in originbands:
                originbands.update(tempdict)
                image = cv2.resize(band, (self.displayfea_w, self.displayfea_l), interpolation=cv2.INTER_LINEAR)
                image = np.multiply(image, filter)
                displaydict = {name: image}
                displaybands.update(displaydict)
                fea_bands = image.reshape((self.displayfea_l * self.displayfea_w), 1)[:, 0]
                vector[:, vectorindex] = vector[:, vectorindex] + fea_bands
        return

    def singleband(self):
        try:
            bands=self.batch_Multiimagebands[self.file].bands
        except:
            return
        if self.partialpca==True:
            npfilter=np.zeros((self.filterbackground[0],self.filterbackground[1]))
            filter=Image.fromarray(npfilter)
            draw=ImageDraw.Draw(filter)
            if self.drawpolygon == False:
                draw.ellipse(self.filtercoord, fill='red')
            else:
                draw.polygon(self.filtercoord, fill='red')
            filter = np.array(filter)
            filter=np.divide(filter,np.max(filter))
            self.partialsingleband(filter)
            return
        channel,fea_l,fea_w=bands.shape
        print('bandsize',fea_l,fea_w)
        if fea_l*fea_w>2000*2000:
            ratio=batch_findratio([fea_l,fea_w],[2000,2000])
        else:
            ratio=1
        print('ratio',ratio)


        originbands={}
        displays={}
        displaybands=cv2.resize(bands[0,:,:],(int(fea_w/ratio),int(fea_l/ratio)),interpolation=cv2.INTER_LINEAR)
        displayfea_l,displayfea_w=displaybands.shape
        print(displayfea_l, displayfea_w)
        self.RGB_vector=np.zeros((displayfea_l*displayfea_w,3))
        self.colorindex_vector=np.zeros((displayfea_l*displayfea_w,12))
        self.displayfea_l,self.displayfea_w=displaybands.shape
        Red=bands[0,:,:]
        Green=bands[1,:,:]
        Blue=bands[2,:,:]

        PAT_R=Red/(Red+Green)
        PAT_G=Green/(Green+Blue)
        PAT_B=Blue/(Blue+Red)

        DIF_R=2*Red-Green-Blue
        DIF_G=2*Green-Blue-Red
        DIF_B=2*Blue-Red-Green

        GLD_R=Red/(np.multiply(np.power(Blue,0.618),np.power(Green,0.382))+1e-6)
        GLD_G=Green/(np.multiply(np.power(Blue,0.618),np.power(Red,0.382))+1e-6)
        GLD_B=Blue/(np.multiply(np.power(Green,0.618),np.power(Red,0.382))+1e-6)

        self.fillbands(originbands,displays,self.colorindex_vector,0,'PAT_R',PAT_R)
        self.fillbands(originbands,displays,self.colorindex_vector,1,'PAT_G',PAT_G)
        self.fillbands(originbands,displays,self.colorindex_vector,2,'PAT_B',PAT_B)
        self.fillbands(originbands,displays,self.colorindex_vector,3,'DIF_R',DIF_R)
        self.fillbands(originbands,displays,self.colorindex_vector,4,'DIF_G',DIF_G)
        self.fillbands(originbands,displays,self.colorindex_vector,5,'DIF_B',DIF_B)
        self.fillbands(originbands,displays,self.colorindex_vector,6,'GLD_R',GLD_R)
        self.fillbands(originbands,displays,self.colorindex_vector,7,'GLD_G',GLD_G)
        self.fillbands(originbands,displays,self.colorindex_vector,8,'GLD_B',GLD_B)
        self.fillbands(originbands,displays,self.colorindex_vector,9,'Band1',Red)
        self.fillbands(originbands,displays,self.colorindex_vector,10,'Band2',Green)
        self.fillbands(originbands,displays,self.colorindex_vector,11,'Band3',Blue)

        NDI=128*((Green-Red)/(Green+Red)+1)
        VEG=Green/(np.power(Red,0.667)*np.power(Blue,(1-0.667)))
        Greenness=Green/(Green+Red+Blue)
        CIVE=0.44*Red+0.811*Green+0.385*Blue+18.7845
        MExG=1.262*Green-0.844*Red-0.311*Blue
        NDRB=(Red-Blue)/(Red+Blue)
        NGRDI=(Green-Red)/(Green+Red)

        colorindex_vector=np.zeros((displayfea_l*displayfea_w,7))

        self.fillbands(originbands,displays,colorindex_vector,0,'NDI',NDI)
        self.fillbands(originbands,displays,colorindex_vector,1,'VEG',VEG)
        self.fillbands(originbands,displays,colorindex_vector,2,'Greenness',Greenness)
        self.fillbands(originbands,displays,colorindex_vector,3,'CIVE',CIVE)
        self.fillbands(originbands,displays,colorindex_vector,4,'MExG',MExG)
        self.fillbands(originbands,displays,colorindex_vector,5,'NDRB',NDRB)
        self.fillbands(originbands,displays,colorindex_vector,6,'NGRDI',NGRDI)

        for i in range(12):
            perc = np.percentile(self.colorindex_vector[:, i], 1)
            print('perc', perc)
            self.colorindex_vector[:, i] = np.where(self.colorindex_vector[:, i] < perc, perc, self.colorindex_vector[:, i])
            perc = np.percentile(self.colorindex_vector[:, i], 99)
            print('perc', perc)
            self.colorindex_vector[:, i] = np.where(self.colorindex_vector[:, i] > perc, perc, self.colorindex_vector[:, i])

        rgb_M=np.mean(self.RGB_vector.T,axis=1)
        colorindex_M=np.mean(self.colorindex_vector.T,axis=1)
        print('rgb_M',rgb_M,'colorindex_M',colorindex_M)
        rgb_C=self.RGB_vector-rgb_M
        colorindex_C=self.colorindex_vector-colorindex_M
        rgb_V=np.corrcoef(rgb_C.T)
        color_V=np.corrcoef(colorindex_C.T)
        nans = np.isnan(color_V)
        color_V[nans] = 1e-6
        # try:
        #     rgb_std = rgb_C / np.std(RGB_vector.T, axis=1)
        # except:
        #     pass
        # rgb_std=rgb_C/np.std(self.RGB_vector.T,axis=1)
        color_std=colorindex_C/np.std(self.colorindex_vector.T,axis=1)
        nans = np.isnan(color_std)
        color_std[nans] = 1e-6
        try:
            rgb_eigval, rgb_eigvec = np.linalg.eig(rgb_V)
            print('rgb_eigvec', rgb_eigvec)
        except:
            pass
        # rgb_eigval,rgb_eigvec=np.linalg.eig(rgb_V)
        # color_V[~np.isfinite(color_V)]=1e-6
        color_eigval,color_eigvec=np.linalg.eig(color_V)
        # print('rgb_eigvec',rgb_eigvec)
        print('color_eigvec',color_eigvec)
        featurechannel=12
        pcabands=np.zeros((self.colorindex_vector.shape[0],featurechannel))

        for i in range(12):
            pcn = color_eigvec[:, i]
            pcnbands = np.dot(color_std, pcn)
            pcvar = np.var(pcnbands)
            print('color index pc', i + 1, 'var=', pcvar)
            pcabands[:, i] = pcabands[:, i] + pcnbands

        for i in range(12):
            perc = np.percentile(pcabands[:, i], 1)
            print('perc', perc)
            pcabands[:, i] = np.where(pcabands[:, i] < perc, perc, pcabands[:, i])
            perc = np.percentile(pcabands[:, i], 99)
            print('perc', perc)
            pcabands[:, i] = np.where(pcabands[:, i] > perc, perc, pcabands[:, i])

        # displayfea_vector=np.concatenate((self.RGB_vector,self.colorindex_vector),axis=1)
        displayfea_vector=np.copy(self.colorindex_vector)
        self.batch_originpcabands.update({self.file:displayfea_vector})
        pcabandsdisplay=pcabands.reshape(displayfea_l,displayfea_w,featurechannel)
        tempdictdisplay={'LabOstu':pcabandsdisplay}
        self.batch_displaybandarray.update({self.file:tempdictdisplay})
        self.batch_originbandarray.update({self.file:originbands})

    def fillpartialbands(self,vector, vectorindex, band, filter_vector):
        nonzero = np.where(filter_vector != 0)
        vector[nonzero, vectorindex] = vector[nonzero, vectorindex] + band

    def partialsingleband(self,filter):
        try:
            bands=self.batch_Multiimagebands[self.file].bands
        except:
            return
        # self.partialpca=True

        channel, fea_l, fea_w = bands.shape
        print('bandsize', fea_l, fea_w)
        if fea_l*fea_w>2000*2000:
            ratio=batch_findratio([fea_l,fea_w],[2000,2000])
        else:
            ratio=1
        print('ratio',ratio)

        originbands = {}
        displays = {}
        displaybands = cv2.resize(bands[0, :, :], (int(fea_w / ratio), int(fea_l / ratio)),
                                  interpolation=cv2.INTER_LINEAR)
        displayfea_l, displayfea_w = displaybands.shape
        print(displayfea_l, displayfea_w)
        self.RGB_vector = np.zeros((displayfea_l * displayfea_w, 3))
        self.colorindex_vector = np.zeros((displayfea_l * displayfea_w, 12))
        self.displayfea_l, self.displayfea_w = displaybands.shape
        filter = cv2.resize(filter, (self.displayfea_w, self.displayfea_l), interpolation=cv2.INTER_LINEAR)
        nonzero = np.where(filter != 0)
        filter_vector = filter.reshape((self.displayfea_l * self.displayfea_w), 1)[:, 0]
        Red = cv2.resize(bands[0, :, :], (self.displayfea_w, self.displayfea_l), interpolation=cv2.INTER_LINEAR)[nonzero]
        Green = cv2.resize(bands[1, :, :], (self.displayfea_w, self.displayfea_l), interpolation=cv2.INTER_LINEAR)[nonzero]
        Blue = cv2.resize(bands[2, :, :], (self.displayfea_w, self.displayfea_l), interpolation=cv2.INTER_LINEAR)[nonzero]
        self.fillpartialbands(self.RGB_vector, 0, Red, filter_vector)
        self.fillpartialbands(self.RGB_vector, 1, Green, filter_vector)
        self.fillpartialbands(self.RGB_vector, 2, Blue, filter_vector)

        PAT_R = Red / (Red + Green)
        PAT_G = Green / (Green + Blue)
        PAT_B = Blue / (Blue + Red)

        DIF_R = 2 * Red - Green - Blue
        DIF_G = 2 * Green - Blue - Red
        DIF_B = 2 * Blue - Red - Green

        GLD_R = Red / (np.multiply(np.power(Blue, 0.618), np.power(Green, 0.382)))
        GLD_G = Green / (np.multiply(np.power(Blue, 0.618), np.power(Red, 0.382)))
        GLD_B = Blue / (np.multiply(np.power(Green, 0.618), np.power(Red, 0.382)))

        self.fillpartialbands(self.colorindex_vector, 0, PAT_R, filter_vector)
        self.fillpartialbands(self.colorindex_vector, 1, PAT_G, filter_vector)
        self.fillpartialbands(self.colorindex_vector, 2, PAT_B, filter_vector)
        self.fillpartialbands(self.colorindex_vector, 3, DIF_R, filter_vector)
        self.fillpartialbands(self.colorindex_vector, 4, DIF_G, filter_vector)
        self.fillpartialbands(self.colorindex_vector, 5, DIF_B, filter_vector)
        self.fillpartialbands(self.colorindex_vector, 6, GLD_R, filter_vector)
        self.fillpartialbands(self.colorindex_vector, 7, GLD_G, filter_vector)
        self.fillpartialbands(self.colorindex_vector, 8, GLD_B, filter_vector)
        self.fillpartialbands(self.colorindex_vector, 9, Red, filter_vector)
        self.fillpartialbands(self.colorindex_vector, 10, Green, filter_vector)
        self.fillpartialbands(self.colorindex_vector, 11, Blue, filter_vector)

        Green=self.RGB_vector[:,1]
        Red=self.RGB_vector[:,0]
        Blue=self.RGB_vector[:,2]
        minvector = np.ones((displayfea_l * displayfea_w, 2)) * 1e-6
        minvector=minvector[:,0]
        NDI = 128 * ((Green - Red) / (Green + Red + minvector) + 1)
        NDI=NDI.reshape((self.displayfea_l , self.displayfea_w))
        VEG = Green / (np.power(Red, 0.667) * np.power(Blue, (1 - 0.667))+minvector)
        VEG=VEG.reshape((self.displayfea_l , self.displayfea_w))
        Greenness = Green / (Green + Red + Blue + minvector)
        Greenness=Greenness.reshape((self.displayfea_l , self.displayfea_w))
        CIVE = 0.44 * Red + 0.811 * Green + 0.385 * Blue + 18.7845
        CIVE=CIVE.reshape((self.displayfea_l , self.displayfea_w))
        MExG = 1.262 * Green - 0.844 * Red - 0.311 * Blue
        MExG=MExG.reshape((self.displayfea_l , self.displayfea_w))
        NDRB = (Red - Blue) / (Red + Blue + minvector)
        NDRB=NDRB.reshape((self.displayfea_l , self.displayfea_w))
        NGRDI = (Green - Red) / (Green + Red + minvector)
        NGRDI=NGRDI.reshape((self.displayfea_l , self.displayfea_w))

        colorindex_vector = np.zeros((displayfea_l * displayfea_w, 7))

        self.fillbands(originbands, displays, colorindex_vector, 0, 'NDI', NDI)
        self.fillbands(originbands, displays, colorindex_vector, 1, 'VEG', VEG)
        self.fillbands(originbands, displays, colorindex_vector, 2, 'Greenness', Greenness)
        self.fillbands(originbands, displays, colorindex_vector, 3, 'CIVE', CIVE)
        self.fillbands(originbands, displays, colorindex_vector, 4, 'MExG', MExG)
        self.fillbands(originbands, displays, colorindex_vector, 5, 'NDRB', NDRB)
        self.fillbands(originbands, displays, colorindex_vector, 6, 'NGRDI', NGRDI)


        for i in range(12):
            perc = np.percentile(self.colorindex_vector[:, i], 1)
            print('partial perc', perc)
            self.colorindex_vector[:, i] = np.where(self.colorindex_vector[:, i] < perc, perc, self.colorindex_vector[:, i])
            perc = np.percentile(self.colorindex_vector[:, i], 99)
            print('partial perc', perc)
            self.colorindex_vector[:, i] = np.where(self.colorindex_vector[:, i] > perc, perc, self.colorindex_vector[:, i])

        self.nonzero_vector = np.where(filter_vector != 0)
        rgb_M = np.mean(self.RGB_vector[self.nonzero_vector,:].T, axis=1)
        colorindex_M = np.mean(self.colorindex_vector[self.nonzero_vector,:].T, axis=1)
        print('rgb_M', rgb_M, 'colorindex_M', colorindex_M)
        rgb_C = self.RGB_vector[self.nonzero_vector,:][0] - rgb_M.T
        colorindex_C = self.colorindex_vector[self.nonzero_vector,:][0] - colorindex_M.T
        rgb_V = np.corrcoef(rgb_C.T)
        color_V = np.corrcoef(colorindex_C.T)
        nans = np.isnan(color_V)
        color_V[nans] = 1e-6
        rgb_std = rgb_C / (np.std(self.RGB_vector[self.nonzero_vector, :].T, axis=1)).T
        color_std = colorindex_C / (np.std(self.colorindex_vector[self.nonzero_vector, :].T, axis=1)).T
        nans = np.isnan(color_std)
        color_std[nans] = 1e-6
        rgb_eigval, rgb_eigvec = np.linalg.eig(rgb_V)
        color_eigval, color_eigvec = np.linalg.eig(color_V)
        print('rgb_eigvec', rgb_eigvec)
        print('color_eigvec', color_eigvec)
        featurechannel = 12
        pcabands = np.zeros((self.colorindex_vector.shape[0], featurechannel))
        rgbbands = np.zeros((self.colorindex_vector.shape[0], 3))
        for i in range(0, featurechannel):
            pcn = color_eigvec[:, i]
            pcnbands = np.dot(color_std, pcn)
            pcvar = np.var(pcnbands)
            print('color index pc', i + 1, 'var=', pcvar)
            pcabands[self.nonzero_vector, i] = pcabands[self.nonzero_vector, i] + pcnbands

        for i in range(12):
            perc = np.percentile(pcabands[:, i], 1)
            print('perc', perc)
            pcabands[:, i] = np.where(pcabands[:, i] < perc, perc, pcabands[:, i])
            perc = np.percentile(pcabands[:, i], 99)
            print('perc', perc)
            pcabands[:, i] = np.where(pcabands[:, i] > perc, perc, pcabands[:, i])

        displayfea_vector = np.copy(self.colorindex_vector)

        self.batch_originpcabands.update({self.file: displayfea_vector})
        pcabandsdisplay = pcabands.reshape(displayfea_l, displayfea_w, featurechannel)
        tempdictdisplay = {'LabOstu': pcabandsdisplay}
        self.batch_displaybandarray.update({self.file: tempdictdisplay})
        self.batch_originbandarray.update({self.file:originbands})


    def kmeansclassify(self):
        if kmeans==0:
            messagebox.showerror('Kmeans error','Kmeans should greater than 0')
            return None
        file=self.file
        originpcabands=self.batch_displaybandarray[file]['LabOstu']
        pcah,pcaw,pcac=originpcabands.shape
        tempband=np.zeros((pcah,pcaw,1))
        if pcweight==0.0:
            tempband[:,:,0]=tempband[:,:,0]+originpcabands[:,:,pcs]
        else:
            if pcweight<0.0:
                rgbpc=originpcabands[:,:,9]
            else:
                rgbpc=originpcabands[:,:,10]
            rgbpc=(rgbpc-rgbpc.min())*255/(rgbpc.max()-rgbpc.min())
            firstterm=abs(pcweight)*2*rgbpc
            colorpc=originpcabands[:,:,pcs]
            colorpc=(colorpc-colorpc.min())*255/(colorpc.max()-colorpc.min())
            secondterm=(1-abs(pcweight)*2)*colorpc
            tempband[:,:,0]=tempband[:,:,0]+firstterm+secondterm
        self.displaypclagels=np.copy(tempband[:,:,0])
        print('origin pc range',tempband.max(),tempband.min())
        if kmeans==1:
            print('kmeans=1')
            displaylabels=np.mean(tempband,axis=2)
            pyplt.imsave(file+'_k=1.png',displaylabels)
        else:
            if kmeans>1:
                h,w,c=tempband.shape
                print('shape',tempband.shape)
                reshapedtif=tempband.reshape(tempband.shape[0]*tempband.shape[1],c)
                if self.partialpca==True:
                    partialshape = reshapedtif[self.nonzero_vector]
                    print('partial reshape', partialshape.shape)
                    clf = KMeans(n_clusters=kmeans, init='k-means++', n_init=10, random_state=0)
                    tempdisplayimg = clf.fit(partialshape)
                    reshapedtif[self.nonzero_vector, 0] = np.add(tempdisplayimg.labels_, 1)
                    print(reshapedtif[self.nonzero_vector])
                    displaylabels = reshapedtif.reshape((self.batch_displaybandarray[self.file]['LabOstu'].shape[0],
                                                         self.batch_displaybandarray[self.file]['LabOstu'].shape[1]))
                    # reshapedtif=cv2.resize(reshapedtif,(c,resizeshape[0]*resizeshape[1]),cv2.INTER_LINEAR)
                    clusterdict = {}
                    displaylabels = displaylabels + 10
                    for i in range(kmeans):
                        locs = np.where(tempdisplayimg.labels_ == i)
                        try:
                            maxval = partialshape[locs].max()
                        except:
                            print('kmeans', i)
                            messagebox.showerror('Cluster maximum value is ', i)
                            return displaylabels
                        print(maxval)
                        clusterdict.update({maxval: i + 11})
                    print(clusterdict)
                    sortcluster = list(sorted(clusterdict))
                    print(sortcluster)
                    for i in range(len(sortcluster)):
                        cluster_num = clusterdict[sortcluster[i]]
                        displaylabels = np.where(displaylabels == cluster_num, i, displaylabels)
                    return displaylabels
                else:
                    print('reshape',reshapedtif.shape)
                    clf=KMeans(n_clusters=kmeans,init='k-means++',n_init=10,random_state=0)
                    tempdisplayimg=clf.fit(reshapedtif)
                    # print('label=0',np.any(tempdisplayimg==0))
                    displaylabels=tempdisplayimg.labels_.reshape((self.batch_displaybandarray[self.file]['LabOstu'].shape[0],
                                                          self.batch_displaybandarray[self.file]['LabOstu'].shape[1]))

            clusterdict={}
            displaylabels=displaylabels+10
            for i in range(kmeans):
                locs=np.where(tempdisplayimg.labels_==i)
                maxval=reshapedtif[locs].max()
                print(maxval)
                clusterdict.update({maxval:i+10})
            print(clusterdict)
            sortcluster=list(sorted(clusterdict))
            print(sortcluster)
            for i in range(len(sortcluster)):
                cluster_num=clusterdict[sortcluster[i]]
                displaylabels=np.where(displaylabels==cluster_num,i,displaylabels)


        return displaylabels



    def kmeansclassify_oldversion(self):
        if kmeans==0:
            messagebox.showerror('Kmeans error','Kmeans should greater than 0')
            return None
        file=self.file
        originpcabands=self.batch_displaybandarray[self.file]['LabOstu']
        pcah,pcaw,pcac=originpcabands.shape
        print(self.file,'originpcabands',pcah,pcaw,pcac)
        pcakeys=pcs
        tempband=np.zeros((pcah,pcaw,len(pcakeys)))
        for i in range(len(pcakeys)):
            channel=int(pcakeys[i])-1
            tempband[:,:,i]=tempband[:,:,i]+originpcabands[:,:,channel]
        if kmeans==1:
            print('kmeans=1')
            displaylabels=np.mean(tempband,axis=2)
            pyplt.imsave(file+'_k=1.png',displaylabels)
        else:

        #tempband=displaybandarray[currentfilename]['LabOstu']
            if kmeans>1:
                h,w,c=tempband.shape
                print('shape',tempband.shape)
                reshapedtif=tempband.reshape(tempband.shape[0]*tempband.shape[1],c)
                print('reshape',reshapedtif.shape)
                clf=KMeans(n_clusters=kmeans,init='k-means++',n_init=10,random_state=0)
                tempdisplayimg=clf.fit(reshapedtif)
                # print('label=0',np.any(tempdisplayimg==0))
                displaylabels=tempdisplayimg.labels_.reshape((self.batch_displaybandarray[self.file]['LabOstu'].shape[0],
                                                      self.batch_displaybandarray[self.file]['LabOstu'].shape[1]))

            clusterdict={}
            displaylabels=displaylabels+10
            for i in range(kmeans):
                locs=np.where(tempdisplayimg.labels_==i)
                maxval=reshapedtif[locs].max()
                print(maxval)
                clusterdict.update({maxval:i+10})
            print(clusterdict)
            sortcluster=list(sorted(clusterdict))
            print(sortcluster)
            for i in range(len(sortcluster)):
                cluster_num=clusterdict[sortcluster[i]]
                displaylabels=np.where(displaylabels==cluster_num,i,displaylabels)


        return displaylabels

    def generateimgplant(self,displaylabels):
        colordicesband=np.copy(displaylabels)
        tempdisplayimg=np.zeros((self.batch_displaybandarray[self.file]['LabOstu'].shape[0],
                                 self.batch_displaybandarray[self.file]['LabOstu'].shape[1]))
        colordivimg=np.zeros((self.batch_displaybandarray[self.file]['LabOstu'].shape[0],
                              self.batch_displaybandarray[self.file]['LabOstu'].shape[1]))
        for i in range(len(kmeans_sel)):
            sk=kmeans_sel[i]-1
            tempdisplayimg=np.where(displaylabels==sk,1,tempdisplayimg)
        currentlabels=np.copy(tempdisplayimg)
        originbinaryimg=np.copy(tempdisplayimg)

        tempcolorimg=np.copy(displaylabels).astype('float32')
        ratio=batch_findratio([tempdisplayimg.shape[0],tempdisplayimg.shape[1]],[850,850])

        if tempdisplayimg.shape[0]*tempdisplayimg.shape[1]<850*850:
            tempdisplayimg=cv2.resize(tempdisplayimg,(int(tempdisplayimg.shape[1]*ratio),int(tempdisplayimg.shape[0]*ratio)))
            colordivimg=cv2.resize(tempcolorimg,(int(colordivimg.shape[1]*ratio),int(colordivimg.shape[0]*ratio)))
        else:
            tempdisplayimg=cv2.resize(tempdisplayimg,(int(tempdisplayimg.shape[1]/ratio),int(tempdisplayimg.shape[0]/ratio)))
            colordivimg=cv2.resize(tempcolorimg,(int(colordivimg.shape[1]/ratio),int(colordivimg.shape[0]/ratio)))
        binaryimg=np.zeros((tempdisplayimg.shape[0],tempdisplayimg.shape[1],3))
        colordeimg=np.zeros((colordivimg.shape[0],colordivimg.shape[1],3))
        locs=np.where(tempdisplayimg==1)
        binaryimg[locs]=[240,228,66]
        for i in range(kmeans):
            locs=np.where(colordivimg==i)
            colordeimg[locs]=batch_colorbandtable[i]
        Image.fromarray(colordeimg.astype('uint8')).save(self.file+'-allcolorindex.png',"PNG")
        Image.fromarray((binaryimg.astype('uint8'))).save(self.file+'-binaryimg.png',"PNG")
        return currentlabels,originbinaryimg

    def resegment(self):
        if type(self.reseglabels) == type(None):
            return False
        labels=np.copy(self.reseglabels)
        reseglabels,border,colortable,labeldict=tkintercorestat.resegmentinput(labels,minthres,maxthres,minlw,maxlw)
        self.batch_results.update({self.file:(labeldict,{})})
        return True


    def extraction(self,currentlabels):
        #return -1: FAIL, return 0: needs to change something, return 1: pass
        if kmeans==1:
            messagebox.showerror('Invalid Class #',message='#Class = 1, try change it to 2 or more, and refresh Color-Index.')
            return -1
        nonzeros=np.count_nonzero(currentlabels)
        print('nonzero counts',nonzeros)
        nonzeroloc=np.where(currentlabels!=0)
        try:
            ulx,uly=min(nonzeroloc[1]),min(nonzeroloc[0])
        except:
            messagebox.showerror('Invalid Colorindices',message='Need to process colorindicies')
            return -1
        rlx,rly=max(nonzeroloc[1]),max(nonzeroloc[0])
        nonzeroratio=float(nonzeros)/((rlx-ulx)*(rly-uly))
        print(nonzeroratio)
        if nonzeroratio>std_nonzeroratio*2:
            if round(nonzeroratio+std_nonzeroratio,1) <=1.1:
                self.needswitchkmeanssel=True
                return 0

        dealpixel=nonzeroratio*currentlabels.shape[0]*currentlabels.shape[1]
        ratio=1
        if nonzeroratio<=0.2:# and nonzeroratio>=0.1:
            ratio=batch_findratio([currentlabels.shape[0],currentlabels.shape[1]],[1600,1600])
            if currentlabels.shape[0]*currentlabels.shape[1]>1600*1600:
                workingimg=cv2.resize(currentlabels,(int(currentlabels.shape[1]/ratio),int(currentlabels.shape[0]/ratio)),interpolation=cv2.INTER_LINEAR)
            else:
                #ratio=1
                #print('nonzeroratio',ratio)
                workingimg=np.copy(currentlabels)
            segmentratio=0
        else:
            print('deal pixel',dealpixel)
            if dealpixel>512000:
                if currentlabels.shape[0]*currentlabels.shape[1]>850*850:
                    segmentratio=batch_findratio([currentlabels.shape[0],currentlabels.shape[1]],[850,850])
                    if segmentratio<2:
                        segmentratio=2
                    workingimg=cv2.resize(currentlabels,(int(currentlabels.shape[1]/segmentratio),int(currentlabels.shape[0]/segmentratio)),interpolation=cv2.INTER_LINEAR)
            else:
                segmentratio=1
                #print('ratio',ratio)
                workingimg=np.copy(currentlabels)
        pixelmmratio=1.0
        coin=False
        print('nonzeroratio:',ratio,'segmentation ratio',segmentratio)
        print('workingimgsize:',workingimg.shape)
        pyplt.imsave('workingimg.png',workingimg)
        originlabels=None
        if originlabels is None:
            originlabels,border,colortable,originlabeldict=tkintercorestat.init(workingimg,workingimg,'',workingimg,10,coin)
        self.reseglabels=originlabels
        self.batch_results.update({self.file:(originlabeldict,{})})
        return 1

    def savePCAimg(self,originfile):
        file=self.file
        path=self.exportpath
        originpcabands=self.batch_displaybandarray[file]['LabOstu']
        pcah,pcaw,pcac=originpcabands.shape
        tempband=np.zeros((pcah,pcaw))
        if pcweight==0.0:
            tempband=tempband+originpcabands[:,:,pcs]
        else:
            if pcweight<0.0:
                rgbpc=originpcabands[:,:,0]
            else:
                rgbpc=originpcabands[:,:,1]
            rgbpc=(rgbpc-rgbpc.min())*255/(rgbpc.max()-rgbpc.min())
            firstterm=abs(pcweight)*2*rgbpc
            colorpc=originpcabands[:,:,pcs]
            colorpc=(colorpc-colorpc.min())*255/(colorpc.max()-colorpc.min())
            secondterm=(1-abs(pcweight)*2)*colorpc
            tempband=tempband+firstterm+secondterm
        displaylabels=np.copy(tempband)

        if displaylabels.min()<0:
            displaylabels=displaylabels-displaylabels.min()
        colorrange=displaylabels.max()-displaylabels.min()
        displaylabels=displaylabels*255/colorrange
        grayimg=Image.fromarray(displaylabels.astype('uint8'),'L')
        originheight,originwidth=self.batch_Multigraybands[file].size
        origingray=grayimg.resize([originwidth,originheight],resample=Image.BILINEAR)
        origingray.save(path+'/'+originfile+'-PCAimg.png',"PNG")
        # addcolorstrip()
        return

    def savePCAimg_oldversion(self,originfile):
        file=self.file
        path=self.exportpath
        originpcabands=self.batch_displaybandarray[file]['LabOstu']
        pcah,pcaw,pcac=originpcabands.shape
        pcakeys=pcs
        tempband=np.zeros((pcah,pcaw,len(pcakeys)))
        for i in range(len(pcakeys)):
            channel=int(pcakeys[i])-1
            tempband[:,:,i]=tempband[:,:,i]+originpcabands[:,:,channel]
        displaylabels=np.mean(tempband,axis=2)
        # generateimgplant(displaylabels)
        # grayimg=(((displaylabels-displaylabels.min())/(displaylabels.max()-displaylabels.min()))*255.9).astype(np.uint8)
        # pyplt.imsave('k=1.png',displaylabels.astype('uint8'))
        # pyplt.imsave('k=1.png',grayimg)
        if displaylabels.min()<0:
            displaylabels=displaylabels-displaylabels.min()
        colorrange=displaylabels.max()-displaylabels.min()
        displaylabels=displaylabels*255/colorrange
        grayimg=Image.fromarray(displaylabels.astype('uint8'),'L')
        originheight,originwidth=self.batch_Multigraybands[file].size
        origingray=grayimg.resize([originwidth,originheight],resample=Image.BILINEAR)
        origingray.save(path+'/'+originfile+'-PCAimg.png',"PNG")
        # addcolorstrip()
        return

    def showcounting(self,tup,number=True,frame=True,header=True,whext=False,blkext=False):
        labels=tup[0]
        colortable=tup[2]
        coinparts=tup[3]
        filename=tup[4]

        uniquelabels=list(colortable.keys())

        imgrsc=cv2.imread(FOLDER+'/'+filename,flags=cv2.IMREAD_ANYCOLOR)
        imgrsc=cv2.cvtColor(imgrsc,cv2.COLOR_BGR2RGB)
        imgrsc=cv2.resize(imgrsc,(labels.shape[1],labels.shape[0]),interpolation=cv2.INTER_LINEAR)
        image=Image.fromarray(imgrsc)
        if whext==True:
            # blkbkg=np.zeros((labels.shape[0],labels.shape[1],3),dtype='float')
            whbkg=np.zeros((labels.shape[0],labels.shape[1],3),dtype='float')
            whbkg[:,:,:]=[255,255,255]
            itemlocs=np.where(labels!=0)
            # blkbkg[itemlocs]=imgrsc[itemlocs]
            whbkg[itemlocs]=imgrsc[itemlocs]
            image=Image.fromarray(whbkg.astype('uint8'))
        if blkext==True:
            blkbkg=np.zeros((labels.shape[0],labels.shape[1],3),dtype='float')
            itemlocs=np.where(labels!=0)
            blkbkg[itemlocs]=imgrsc[itemlocs]
            blkbkg[itemlocs]=imgrsc[itemlocs]
            image=Image.fromarray(blkbkg.astype('uint8'))

        print('showcounting_resize',image.size)
        image.save('beforlabel.gif',append_images=[image])
        draw=ImageDraw.Draw(image)
        sizeuniq,sizecounts=np.unique(labels,return_counts=True)
        minsize=min(sizecounts)
        suggsize=int(minsize**0.5)
        if suggsize>22:
            suggsize=22
        if suggsize<14:
            suggsize=14
        font=ImageFont.truetype('cmb10.ttf',size=suggsize)

        for uni in uniquelabels:
            if uni!=0:
                pixelloc = np.where(labels == uni)
                try:
                    ulx = min(pixelloc[1])
                except:
                    continue
                uly = min(pixelloc[0])
                rlx = max(pixelloc[1])
                rly = max(pixelloc[0])
                midx = ulx + int((rlx - ulx) / 2)
                midy = uly + int((rly - uly) / 2)
                print(ulx, uly, rlx, rly)
                if frame==True:
                    draw.polygon([(ulx,uly),(rlx,uly),(rlx,rly),(ulx,rly)],outline='red')
                if number==True:
                    if uni in colortable:
                        canvastext = str(colortable[uni])
                    else:
                        canvastext = 'No label'
                    # if imgtypevar.get()=='0':
                    draw.text((midx-1, midy+1), text=canvastext, font=font, fill='white')
                    draw.text((midx+1, midy+1), text=canvastext, font=font, fill='white')
                    draw.text((midx-1, midy-1), text=canvastext, font=font, fill='white')
                    draw.text((midx+1, midy-1), text=canvastext, font=font, fill='white')
                    #draw.text((midx,midy),text=canvastext,font=font,fill=(141,2,31,0))
                    draw.text((midx,midy),text=canvastext,font=font,fill='black')


        if header==True:
            content='item count:'+str(len(uniquelabels))+'\n File: '+filename
            contentlength=len(content)+50
            #rectext=canvas.create_text(10,10,fill='black',font='Times 16',text=content,anchor=NW)
            draw.text((10-1, 10+1), text=content, font=font, fill='white')
            draw.text((10+1, 10+1), text=content, font=font, fill='white')
            draw.text((10-1, 10-1), text=content, font=font, fill='white')
            draw.text((10+1, 10-1), text=content, font=font, fill='white')
            #draw.text((10,10),text=content,font=font,fill=(141,2,31,0))
            draw.text((10,10),text=content,font=font,fill='black')
        #image.save(originfile+'-countresult'+extension,"JPEG")
        #firstimg=Multigraybands[currentfilename]
        #height,width=firstimg.size
        height,width,channel=self.batch_displaybandarray[filename]['LabOstu'].shape
        ratio=batch_findratio([height,width],[850,850])
        #if labels.shape[0]*labels.shape[1]<850*850:
        #    disimage=image.resize([int(labels.shape[1]*ratio),int(labels.shape[0]*ratio)],resample=Image.BILINEAR)
        #else:
        #    disimage=image.resize([int(labels.shape[1]/ratio),int(labels.shape[0]/ratio)],resample=Image.BILINEAR)
        print('show counting ratio',ratio)
        if height*width<850*850:
            print('showcounting small')
            disimage=image.resize([int(width*ratio),int(height*ratio)],resample=Image.BILINEAR)
        else:
            print('showcounting big')
            disimage=image.resize([int(width/ratio),int(height/ratio)],resample=Image.BILINEAR)
        print('showcounting shape',disimage.size)
        displayoutput=ImageTk.PhotoImage(disimage)
        disimage.save('output.gif',append_images=[disimage])
        #image.save('originoutput.gif',append_images=[image])
        return displayoutput,image,disimage

    def export_ext(self,whext=False,blkext=False):
        if len(batch_filenames)==0:
            messagebox.showerror('No files','Please load images to process')
            return
        file=self.file
        path=self.exportpath
        suggsize=8
        smallfont=ImageFont.truetype('cmb10.ttf',size=suggsize)
        # kernersizes={}
        # for file in batch_filenames:
        labeldict=self.batch_results[file][0]
        itervalue='iter0'
        labels=labeldict[itervalue]['labels']
        counts=labeldict[itervalue]['counts']
        colortable=labeldict[itervalue]['colortable']
        head_tail=os.path.split(file)
        originfile,extension=os.path.splitext(head_tail[1])
        if len(path)>0:
            tup=(labels,counts,colortable,[],file)
            _band,segimg,small_segimg=self.showcounting(tup,False,True,True,whext,blkext)
            imageband=segimg
            draw=ImageDraw.Draw(imageband)
            uniquelabels=list(colortable.keys())
            tempdict={}
            pixelmmratio=1.0
            print('pixelmmratio',pixelmmratio)
            if file not in self.kernersizes:
                for uni in uniquelabels:
                    if uni !=0:
                        pixelloc = np.where(labels == float(uni))
                        try:
                            ulx = min(pixelloc[1])
                        except:
                            continue
                        uly = min(pixelloc[0])
                        rlx = max(pixelloc[1])
                        rly = max(pixelloc[0])
                        print(ulx, uly, rlx, rly)
                        midx = ulx + int((rlx - ulx) / 2)
                        midy = uly + int((rly - uly) / 2)
                        length={}
                        currborder=tkintercore.get_boundaryloc(labels,uni)
                        # print('currborder',currborder)
                        print('currborder length',len(currborder[0])*len(currborder[1]))
                        pixperc=float(len(pixelloc[0])/(labels.shape[0]*labels.shape[1]))
                        print('pix length percentage',pixperc)
                        if pixperc>0.06:
                            x0=ulx
                            y0=uly
                            x1=rlx
                            y1=rly
                            kernellength=float(((x0-x1)**2+(y0-y1)**2)**0.5)
                        else:
                            for i in range(len(currborder[0])):
                                for j in range(i+1,len(currborder[0])):
                                    templength=float(((currborder[0][i]-currborder[0][j])**2+(currborder[1][i]-currborder[1][j])**2)**0.5)
                                    length.update({(i,j):templength})
                            sortedlength=sorted(length,key=length.get,reverse=True)
                            try:
                                topcouple=sortedlength[0]
                            except:
                                continue
                            kernellength=length[topcouple]
                            i=topcouple[0]
                            j=topcouple[1]
                            x0=currborder[1][i]
                            y0=currborder[0][i]
                            x1=currborder[1][j]
                            y1=currborder[0][j]
                            #slope=float((y0-y1)/(x0-x1))
                            linepoints=[(currborder[1][i],currborder[0][i]),(currborder[1][j],currborder[0][j])]
                            #draw.line(linepoints,fill='yellow')
                            #points=linepixels(currborder[1][i],currborder[0][i],currborder[1][j],currborder[0][j])

                        lengthpoints=cal_kernelsize.bresenhamline(x0,y0,x1,y1)  #x0,y0,x1,y1
                        for point in lengthpoints:
                            # if imgtypevar.get()=='0':
                            draw.point([int(point[0]),int(point[1])],fill='yellow')
                        tengentaddpoints=cal_kernelsize.tengentadd(x0,y0,x1,y1,rlx,rly,labels,uni) #find tangent line above
                        #for point in tengentaddpoints:
                            #if int(point[0])>=ulx and int(point[0])<=rlx and int(point[1])>=uly and int(point[1])<=rly:
                        #    draw.point([int(point[0]),int(point[1])],fill='green')
                        tengentsubpoints=cal_kernelsize.tengentsub(x0,y0,x1,y1,ulx,uly,labels,uni) #find tangent line below
                        #for point in tengentsubpoints:
                        #    draw.point([int(point[0]),int(point[1])],fill='green')
                        pointmatchdict={}
                        for i in range(len(tengentaddpoints)):  #find the pixel pair with shortest distance
                            width=kernellength
                            pointmatch=[]
                            point=tengentaddpoints[i]
                            try:
                                templabel=labels[int(point[1]),int(point[0])]
                            except:
                                continue
                            if templabel==uni:
                                for j in range(len(tengentsubpoints)):
                                    subpoint=tengentsubpoints[j]
                                    tempwidth=float(((point[0]-subpoint[0])**2+(point[1]-subpoint[1])**2)**0.5)
                                    if tempwidth<width:
                                        pointmatch[:]=[]
                                        pointmatch.append(point)
                                        pointmatch.append(subpoint)
                                        #print('tempwidth',width)
                                        width=tempwidth
                            if len(pointmatch)>0:
                                #print('pointmatch',pointmatch)
                                pointmatchdict.update({(pointmatch[0],pointmatch[1]):width})
                        widthsort=sorted(pointmatchdict,key=pointmatchdict.get,reverse=True)
                        try:
                            pointmatch=widthsort[0]
                            print('final pointmatch',pointmatch)
                        except:
                            continue
                        if len(pointmatch)>0:
                            x0=int(pointmatch[0][0])
                            y0=int(pointmatch[0][1])
                            x1=int(pointmatch[1][0])
                            y1=int(pointmatch[1][1])
                            # if imgtypevar.get()=='0':
                            draw.line([(x0,y0),(x1,y1)],fill='yellow')
                            width=float(((x0-x1)**2+(y0-y1)**2)**0.5)
                            print('width',width,'length',kernellength)
                            print('kernelwidth='+str(width*pixelmmratio))
                            print('kernellength='+str(kernellength*pixelmmratio))
                            #print('kernelwidth='+str(kernelwidth*pixelmmratio))
                            tempdict.update({uni:[kernellength,width,pixelmmratio**2*len(pixelloc[0]),kernellength*pixelmmratio,width*pixelmmratio]})
                        if uni in colortable:
                            canvastext = str(colortable[uni])
                        else:
                            canvastext = 'No label'
                        # if imgtypevar.get()=='0':
                        draw.text((midx-1, midy+1), text=canvastext, font=smallfont, fill='white')
                        draw.text((midx+1, midy+1), text=canvastext, font=smallfont, fill='white')
                        draw.text((midx-1, midy-1), text=canvastext, font=smallfont, fill='white')
                        draw.text((midx+1, midy-1), text=canvastext, font=smallfont, fill='white')
                        #draw.text((midx,midy),text=canvastext,font=font,fill=(141,2,31,0))
                        draw.text((midx,midy),text=canvastext,font=smallfont,fill='black')

                        #print(event.x, event.y, labels[event.x, event.y], ulx, uly, rlx, rly)

                        #recborder = canvas.create_rectangle(ulx, uly, rlx, rly, outline='red')
                        #drawcontents.append(recborder)
                self.kernersizes.update({file:tempdict})
            originheight,originwidth=self.batch_Multigraybands[file].size
            image=imageband.resize([originwidth,originheight],resample=Image.BILINEAR)
            extcolor=""
            if whext==True:
                extcolor= "-extwht"
            if blkext==True:
                extcolor="-extblk"
            image.save(path+'/'+originfile+extcolor+'-sizeresult'+'.png',"PNG")
            tup=(labels,counts,colortable,[],file)
            _band,segimg,small_segimg=self.showcounting(tup,False,True,True,whext,blkext)
            segimage=segimg.resize([originwidth,originheight],resample=Image.BILINEAR)
            segimage.save(path+'/'+originfile+extcolor+'-segmentresult'+'.png',"PNG")
            _band,segimg,small_segimg=self.showcounting(tup,True,True,True,whext,blkext)
            segimage=segimg.resize([originwidth,originheight],resample=Image.BILINEAR)
            segimage.save(path+'/'+originfile+extcolor+'-labelresult'+'.png',"PNG")

    def export_result(self):
        file=self.file
        if len(batch_filenames)==0:
            messagebox.showerror('No files','Please load images to process')
            return
        suggsize=8
        smallfont=ImageFont.truetype('cmb10.ttf',size=suggsize)
        self.kernersizes={}
        '''background big img'''
        if segmentoutputopt==True:
            self.export_ext(True,False)
            self.export_ext(False,True)
        '''end background big img'''
        labeldict=self.batch_results[self.file][0]
        itervalue='iter0'
        labels=labeldict[itervalue]['labels']
        counts=labeldict[itervalue]['counts']
        colortable=labeldict[itervalue]['colortable']
        head_tail=os.path.split(self.file)
        originfile,extension=os.path.splitext(head_tail[1])

        '''export cropped images'''
        if cropimageopt==True:
            if len(self.exportpath)>0:
                originheight, originwidth = self.batch_Multigraybands[file].size
                uniquelabels = list(colortable.keys())
                cropratio = batch_findratio((originheight, originwidth), (labels.shape[0], labels.shape[1]))
                if cropratio > 1:
                    cache = (np.zeros((originheight, originwidth)),
                             {"f": int(cropratio), "stride": int(cropratio)})
                    originconvband = tkintercorestat.pool_backward(labels, cache)
                else:
                    originconvband = np.copy(labels)
                imgrsc = cv2.imread(os.path.join(FOLDER,file), flags=cv2.IMREAD_ANYCOLOR)
                # cv2.imwrite(os.path.join(self.exportpath, originfile + '_before.png'), originconvband)
                labelsegfile = os.path.join(self.exportpath, originfile + '_cropimage_label.csv')
                with open(labelsegfile, mode='w') as f:
                    csvwriter = csv.writer(f)
                    # rowcontent=['id','locs']
                    rowcontent = ['index', 'i', 'j', 'filename', 'label']
                    csvwriter.writerow(rowcontent)
                    #     result_ref=envi.open(head_tail[0]+'/'+originfile+'/results/REFLECTANCE_'+originfile+'.hdr', head_tail[0]+'/'+originfile+'/results/REFLECTANCE_'+originfile+'.dat')
                    #     result_nparr=np.array(result_ref.load())
                    #     corrected_nparr=np.copy(result_nparr)
                    index = 1
                    for uni in uniquelabels:
                        if uni != 0:
                            tempuni = colortable[uni]
                            if tempuni == 'Ref':
                                # pixelloc = np.where(convband == 65535)
                                originpixelloc = np.where(originconvband == 65535)
                            else:
                                # pixelloc = np.where(convband == float(uni))
                                originpixelloc = np.where(originconvband == float(uni))
                            # kernelval=corrected_nparr[pixelloc]
                            # nirs=np.mean(kernelval,axis=0)
                            #             print('nirs 170',nirs[170])
                            #             if nirs[170]<0.15:
                            #                 lesszeroonefive.append(uni)
                            try:
                                # ulx = min(pixelloc[1])
                                ulx = min(originpixelloc[1])
                            except:
                                print('no pixellloc[1] on uni=', uni)
                                print('pixelloc =', originpixelloc)
                                continue
                            uly = min(originpixelloc[0])
                            rlx = max(originpixelloc[1])
                            rly = max(originpixelloc[0])
                            width = rlx - ulx + 1
                            height = rly - uly + 1
                            originbkgloc = np.where(originconvband == 0)
                            blx = min(originbkgloc[1])
                            bly = min(originbkgloc[0])
                            if max(height / width, width / height) > 1.1:
                                edgelen = max(height, width)
                                zeronp = np.ones((edgelen, edgelen, 3), dtype='float')
                            else:
                                zeronp = np.ones((height, width, 3), dtype='float')
                            zeronp = zeronp * imgrsc[blx, bly, :]
                            temppixelloc = (originpixelloc[0] - uly, originpixelloc[1] - ulx)
                            zeronp[temppixelloc[0], temppixelloc[1], :] = imgrsc[originpixelloc[0], originpixelloc[1],
                                                                          :]
                            # cropimage = imgrsc[uly:rly, ulx:rlx]
                            cropimage = np.copy(zeronp)
                            cv2.imwrite(os.path.join(self.exportpath, originfile + '_crop_' + str(int(uni)) + '.png'), cropimage)
                            print('output to cropimg', self.exportpath, originfile + '_crop_' + str(int(uni)) + '.png')
                            rowcontent = [index, 0, 0, originfile + '_crop_' + str(int(uni)) + '.png', 0]
                            csvwriter.writerow(rowcontent)
                            index += 1
                            # rowcontent=[colortable[uni]]
                            # rowcontent=rowcontent+list(pixelloc[0])
                            # csvwriter.writerow(rowcontent)
                            # rowcontent=[colortable[uni]]
                            # rowcontent=rowcontent+list(pixelloc[1])
                            # csvwriter.writerow(rowcontent)

                    f.close()
                # print(lesszeroonefive)



        if segmentoutputopt==True:
            if len(self.exportpath)>0:
                tup=(labels,counts,colortable,[],self.file)
                _band,segimg,small_segimg=self.showcounting(tup,False)
                #imageband=outputimgbands[file][itervalue]
                imageband=segimg
                # draw=ImageDraw.Draw(imageband)
                uniquelabels=list(colortable.keys())
                # tempdict={}
                pixelmmratio=1.0
                #print('coinsize',coinsize.get(),'pixelmmratio',pixelmmratio)
                print('pixelmmratio',pixelmmratio)
                originheight,originwidth=self.batch_Multigraybands[file].size
                '''big output img'''
                image=imageband.resize([originwidth,originheight],resample=Image.BILINEAR)
                image.save(self.exportpath+'/'+originfile+'-sizeresult'+'.png',"PNG")
                tup=(labels,counts,colortable,[],file)
                _band,segimg,small_segimg=self.showcounting(tup,False)
                segimage=segimg.resize([originwidth,originheight],resample=Image.BILINEAR)
                segimage.save(self.exportpath+'/'+originfile+'-segmentresult'+'.png',"PNG")
                _band,segimg,small_segimg=self.showcounting(tup,True)
                segimage=segimg.resize([originwidth,originheight],resample=Image.BILINEAR)
                segimage.save(self.exportpath+'/'+originfile+'-labelresult'+'.png',"PNG")
                '''end big output img'''
                originrestoredband=np.copy(labels)
                restoredband=originrestoredband.astype('uint8')
                colordicesband=self.batch_colordicesband[file]
                colordiv=np.zeros((colordicesband.shape[0],colordicesband.shape[1],3))
                self.savePCAimg(originfile)

                # kvar=int(kmeans.get())
                # print('kvar',kvar)
                # for i in range(kvar):
                #     locs=np.where(colordicesband==i)
                #     colordiv[locs]=colorbandtable[i]
                # colordivimg=Image.fromarray(colordiv.astype('uint8'))
                # colordivimg.save(path+'/'+originfile+'-colordevice'+'.jpeg',"JPEG")
                colordivimg=Image.open(file+'-allcolorindex.png')
                copycolordiv=colordivimg.resize([originwidth,originheight],resample=Image.BILINEAR)
                copycolordiv.save(self.exportpath+'/'+originfile+'-colordevice'+'.png',"PNG")
                # pyplt.imsave(path+'/'+originfile+'-colordevice'+'.png',colordiv.astype('uint8'))
                # copybinary=np.zeros((originbinaryimg.shape[0],originbinaryimg.shape[1],3),dtype='float')
                # nonzeros=np.where(originbinaryimg==1)
                # copybinary[nonzeros]=[255,255,0]
                # binaryimg=Image.fromarray(copybinary.astype('uint8'))
                binaryimg=Image.open(file+'-binaryimg.png')
                copybinaryimg=binaryimg.resize([originwidth,originheight],resample=Image.BILINEAR)
                copybinaryimg.save(self.exportpath+'/'+originfile+'-binaryimg'+'.png',"PNG")
                # pyplt.imsave(path+'/'+originfile+'-binaryimg'+'.png',originbinaryimg.astype('uint8'))

                #restoredband=cv2.resize(src=restoredband,dsize=(originwidth,originheight),interpolation=cv2.INTER_LINEAR)
                '''calculate output csv content'''
                print(restoredband.shape)
                currentsizes=self.kernersizes[self.file]
                indicekeys=list(self.batch_originbandarray[self.file].keys())
                indeclist=[ 0 for i in range(len(indicekeys)*3)]
                pcalist=[0 for i in range(3)]
                '''end'''
                # temppcabands=np.zeros((self.batch_originpcabands[self.file].shape[0],len(pcs)))
                # for i in range(len(pcs)):
                #     temppcabands[:,i]=temppcabands[:,i]+self.batch_originpcabands[self.file][:,pcs[i]-1]
                # pcabands=np.mean(temppcabands,axis=1)
                # # pcabands=pcabands.reshape((originheight,originwidth))
                # pcabands=pcabands.reshape((self.displayfea_l,self.displayfea_w))
                '''calculate output csv'''
                pcabands=np.copy(self.displaypclagels)
                datatable={}
                origindata={}
                for key in indicekeys:
                    data=self.batch_originbandarray[self.file][key]
                    data=data.tolist()
                    tempdict={key:data}
                    origindata.update(tempdict)
                    print(key)
                # for uni in colortable:
                print(uniquelabels)
                print('len uniquelabels',len(uniquelabels))

                for uni in uniquelabels:
                    print(uni,colortable[uni])
                    uniloc=np.where(labels==float(uni))
                    if len(uniloc)==0 or len(uniloc[1])==0:
                        print('no uniloc\n')
                        print(uniloc[0],uniloc[1])
                        continue
                    smalluniloc=np.where(originrestoredband==uni)
                    ulx,uly=min(smalluniloc[1]),min(smalluniloc[0])
                    rlx,rly=max(smalluniloc[1]),max(smalluniloc[0])
                    width=rlx-ulx
                    length=rly-uly
                    print(width,length)
                    subarea=restoredband[uly:rly+1,ulx:rlx+1]
                    subarea=subarea.tolist()
                    amount=len(uniloc[0])
                    print(amount)
                    try:
                        sizes=currentsizes[uni]
                    except:
                        print('no sizes\n')
                        continue
                    #templist=[amount,length,width]
                    templist=[amount,sizes[0],sizes[1],sizes[2],sizes[3],sizes[4]]
                    tempdict={colortable[uni]:templist+indeclist+pcalist}  #NIR,Redeyes,R,G,B,NDVI,area
                    print(tempdict)
                    for ki in range(len(indicekeys)):
                        originNDVI=origindata[indicekeys[ki]]
                        print(len(originNDVI),len(originNDVI[0]))
                        pixellist=[]
                        for k in range(len(uniloc[0])):
                            #print(uniloc[0][k],uniloc[1][k])
                            try:
                                tempdict[colortable[uni]][6+ki*3]+=originNDVI[uniloc[0][k]][uniloc[1][k]]
                            except IndexError:
                                print(uniloc[0][k],uniloc[1][k])
                            tempdict[colortable[uni]][7+ki*3]+=originNDVI[uniloc[0][k]][uniloc[1][k]]
                            pixellist.append(originNDVI[uniloc[0][k]][uniloc[1][k]])
                        tempdict[colortable[uni]][ki*3+6]=tempdict[colortable[uni]][ki*3+6]/amount
                        tempdict[colortable[uni]][ki*3+8]=np.std(pixellist)
                    pixellist=[]
                    for k in range(len(uniloc[0])):
                        try:
                            tempdict[colortable[uni]][-2]+=pcabands[uniloc[0][k]][uniloc[1][k]]
                        except IndexError:
                            print(uniloc[0][k],uniloc[1][k])
                        tempdict[colortable[uni]][-3]+=pcabands[uniloc[0][k]][uniloc[1][k]]
                        pixellist.append(pcabands[uniloc[0][k]][uniloc[1][k]])
                        tempdict[colortable[uni]][-3]=tempdict[colortable[uni]][-3]/amount
                        tempdict[colortable[uni]][-1]=np.std(pixellist)
                    datatable.update(tempdict)
                filename=self.exportpath+'/'+originfile+'-outputdata.csv'
                with open(filename,mode='w') as f:
                    csvwriter=csv.writer(f)
                    rowcontent=['Index','Plot','Area(#pixel)','Length(#pixel)','Width(#pixel)','Area(mm2)','Length(mm)','Width(mm)']
                    for key in indicekeys:
                        rowcontent.append('avg-'+str(key))
                        rowcontent.append('sum-'+str(key))
                        rowcontent.append('std-'+str(key))
                    rowcontent.append('avg-PCA')
                    rowcontent.append('sum-PCA')
                    rowcontent.append('std-PCA')
                    #csvwriter.writerow(['ID','NIR','Red Edge','Red','Green','Blue','NIRv.s.Green','LabOstu','area(#of pixel)'])
                    #csvwriter.writerow(['Index','Plot','Area(#pixels)','avg-NDVI','sum-NDVI','std-NDVI','Length(#pixel)','Width(#pixel)'])#,'#holes'])
                    csvwriter.writerow(rowcontent)
                    i=1
                    for uni in datatable:
                        row=[i,uni]
                        for j in range(len(datatable[uni])):
                            row.append(datatable[uni][j])
                        #row=[i,uni,datatable[uni][0],datatable[uni][1],datatable[uni][2],datatable[uni][5],datatable[uni][3],datatable[uni][4]]#,
                             #datatable[uni][5]]
                        i+=1
                        print(row)
                        csvwriter.writerow(row)

                print('total data length=',len(datatable))
                '''output csv output end'''

    def process(self):
        if self.Open_batchimage()==False:
            return
        self.singleband()
        colordicesband=self.kmeansclassify()
        if type(colordicesband)==type(None):
            print("colordiceband return none\n")
            return
        self.batch_colordicesband.update({self.file:colordicesband})
        currentlabels,originbinaryimg=self.generateimgplant(colordicesband)
        if self.extraction(currentlabels)==-1:
            print("extraction return false\n")
            return
        if self.extraction(currentlabels)==0:
            print("need to switch pc sel in batch.txt\n")
            return
        if self.resegment()==False:
            print("resegment return false\n")
            return
        self.export_result()









batch_filenames=[]
batch_Multiimage={}
batch_Multigray={}
batch_Multitype={}
batch_Multiimagebands={}
batch_Multigraybands={}
batch_displaybandarray={}
batch_originbandarray={}
batch_originpcabands={}
batch_colordicesband={}

batch_results={}

pcweight=0
pcs=0
kmeans=0
kmeans_sel=[]
maxthres=0
minthres=0
maxlw=0
minlw=0
std_nonzeroratio=0
FOLDER=''
exportpath=''
drawpolygon=False
filtercoord=[]
filterbackground=[]
segmentoutputopt=False
cropimageopt=False

def batch_findratio(originsize,objectsize):
    oria=originsize[0]
    orib=originsize[1]
    obja=objectsize[0]
    objb=objectsize[1]
    if oria>obja or orib>objb:
        ratio=round(max((oria/obja),(orib/objb)))
    else:
        ratio=round(min((obja/oria),(objb/orib)))
    if oria*orib>850 * 850:
        if ratio<2:
            ratio=2
    return ratio

def Open_batchimage(dir,filename):
    global batch_Multiimage,batch_Multigray,batch_Multitype,batch_Multiimagebands,batch_Multigraybands
    try:
        Filersc=cv2.imread(dir+'/'+filename,flags=cv2.IMREAD_ANYCOLOR)
        height,width,channel=np.shape(Filersc)
        Filesize=(height,width)
        print('filesize:',height,width)
        RGBfile=cv2.cvtColor(Filersc,cv2.COLOR_BGR2RGB)
        Grayfile=cv2.cvtColor(Filersc,cv2.COLOR_BGR2Lab)
        Grayfile=cv2.cvtColor(Grayfile,cv2.COLOR_BGR2GRAY)
        Grayimg=batch_img(Filesize,Grayfile)
        RGBbands=np.zeros((channel,height,width))
        for j in range(channel):
            band=RGBfile[:,:,j]
            band=np.where(band==0,1e-6,band)
            RGBbands[j,:,:]=band
        RGBimg=batch_img(Filesize,RGBbands)
        tempdict={filename:RGBimg}
        batch_Multiimagebands.update(tempdict)
        tempdict={filename:Grayfile}
        batch_Multigray.update(tempdict)
        tempdict={filename:0}
        batch_Multitype.update(tempdict)
        tempdict={filename:Grayimg}
        batch_Multigraybands.update(tempdict)
        # batch_filenames.append(filename)
    except:
        # messagebox.showerror('Invalid Image Format','Cannot open '+filename)
        return False
    return True

def Open_batchfile():
    global pcs,pcweight,kmeans,kmeans_sel,maxthres,minthres,maxlw,minlw,std_nonzeroratio
    global drawpolygon,filtercoord,filterbackground,segmentoutputopt,cropimageopt
    btfile=filedialog.askopenfilename()
    if len(btfile)>0:
        if '.txt' in btfile:
            with open(btfile,mode='r') as f:
                setting=f.readlines()
                # print(setting)
                pcweight=float(setting[0].split(',')[1])
                pcs=int(setting[1].split(',')[1])-1
                # print(pcs)
                # for i in range(len(pcs)):
                #     pcs[i]=int(pcs[i])
                kmeans=setting[2].split(',')[1]
                kmeans=int(kmeans)
                kmeans_sel=setting[3].split(',')[1:-1]
                maxthres=setting[4].split(',')[1]
                try:
                    maxthres=float(maxthres)
                except:
                    messagebox.showerror('Load Max area error','No Max area threshold value.')
                    return
                minthres=setting[5].split(',')[1]
                minthres=float(minthres)
                maxlw=setting[6].split(',')[1]
                maxlw=float(maxlw)
                minlw=setting[7].split(',')[1]
                minlw=float(minlw)
                std_nonzeroratio = float(setting[8].split(',')[1])
                drawpolygon=int(setting[9].split(',')[1])
                drawpolygon=bool(drawpolygon)
                filtercoord=setting[10].split(',')[1:-1]
                filterbackground=[]
                if len(filtercoord)>0:
                    filtercoord=[float(ele) for ele in filtercoord]
                    filterbackground = setting[11].split(',')[1:-1]
                    filterbackground=[int(ele) for ele in filterbackground]
                segmentoutputopt=setting[12].split(',')[1]
                segmentoutputopt=bool(int(segmentoutputopt))
                cropimageopt=setting[13].split(',')[1]
                cropimageopt=bool(int(cropimageopt))
                for i in range(len(kmeans_sel)):
                    kmeans_sel[i]=int(kmeans_sel[i])
                print('PCweight',pcweight,'PCsel',pcs+1,'KMeans',kmeans,'KMeans-Selection',kmeans_sel)
                print('maxthres',maxthres,'minthres',minthres,'maxlw',maxlw,'minlw',minlw)
                messagebox.showinfo('Batch settings','PCweight='+str(pcweight)+'\nPCsel='+str(pcs+1)+'\nKMeans='+str(kmeans)+
                                    '\nCluster selection'+str(kmeans_sel)+'\nMax area='+str(maxthres)+
                                    '\nMin area='+str(minthres)+'\nMax diagonal='+str(maxlw)+'\nMin diagonal='+
                                    str(minlw)+'\nDrawpolygon='+str(drawpolygon)+'\nFilter='+''.join(str(e) for e in filtercoord)+
                                    '\nOutputBigimg='+str(segmentoutputopt)+'\nOutputCropimg='+str(cropimageopt))

def Open_batchfolder():
    # global batch_filenames,batch_Multiimage,batch_Multigray,batch_Multitype,batch_Multiimagebands,batch_Multigraybands
    # global batch_displaybandarray,batch_originbandarray,batch_originpcabands
    # global pcs,kmeans,kmeans_sel
    # global batch_results
    global batch_filenames
    global FOLDER

    batch_filenames=[]
    # batch_Multiimage={}
    # batch_Multigray={}
    # batch_Multitype={}
    # batch_Multiimagebands={}
    # batch_Multigraybands={}
    #
    # batch_displaybandarray={}
    # batch_originbandarray={}
    # batch_originpcabands={}
    #
    # batch_results={}

    # pcs=0
    # kmeans=0
    # kmeans_sel=0

    FOLDER=filedialog.askdirectory()
    if len(FOLDER)>0:
        print(FOLDER)
        # for root, dirs,files in os.walk(FOLDER):
        files=os.listdir(FOLDER)
        for filename in files:
            # print('root',root)
            # print('dirs',dirs)
            # print("filename",filename)
            batch_filenames.append(filename)
            # batch_filenames.append(filename)
        #     Open_batchimage(FOLDER,filename)
        #     batch_singleband(filename)
        # messagebox.showinfo('Finish loading','Loading Image finished')
        # if len(batch_filenames)==0:
        #     messagebox.showerror('No file','No file under current folder')
        #     return
    batch_filenames.sort()
    print('filenames',batch_filenames)
def batch_kmeansclassify(file):
    if kmeans==0:
        messagebox.showerror('Kmeans error','Kmeans should greater than 0')
        return

    originpcabands=batch_displaybandarray[file]['LabOstu']
    pcah,pcaw,pcac=originpcabands.shape
    print(file,'originpcabands',pcah,pcaw,pcac)
    pcakeys=pcs
    tempband=np.zeros((pcah,pcaw,len(pcakeys)))
    for i in range(len(pcakeys)):
        channel=int(pcakeys[i])-1
        tempband[:,:,i]=tempband[:,:,i]+originpcabands[:,:,channel]
    if kmeans==1:
        print('kmeans=1')
        displaylabels=np.mean(tempband,axis=2)
        pyplt.imsave(file+'_k=1.png',displaylabels)
    else:

    #tempband=displaybandarray[currentfilename]['LabOstu']
        if kmeans>1:
            h,w,c=tempband.shape
            print('shape',tempband.shape)
            reshapedtif=tempband.reshape(tempband.shape[0]*tempband.shape[1],c)
            print('reshape',reshapedtif.shape)
            clf=KMeans(n_clusters=kmeans,init='k-means++',n_init=10,random_state=0)
            tempdisplayimg=clf.fit(reshapedtif)
            # print('label=0',np.any(tempdisplayimg==0))
            displaylabels=tempdisplayimg.labels_.reshape((batch_displaybandarray[file]['LabOstu'].shape[0],
                                                  batch_displaybandarray[file]['LabOstu'].shape[1]))
    return displaylabels

def batch_generateimgplant(displaylabels,file):

    colordicesband=np.copy(displaylabels)
    tempdisplayimg=np.zeros((batch_displaybandarray[file]['LabOstu'].shape[0],
                             batch_displaybandarray[file]['LabOstu'].shape[1]))
    colordivimg=np.zeros((batch_displaybandarray[file]['LabOstu'].shape[0],
                          batch_displaybandarray[file]['LabOstu'].shape[1]))
    for i in range(len(kmeans_sel)):
        sk=kmeans_sel[i]-1
        tempdisplayimg=np.where(displaylabels==sk,1,tempdisplayimg)
    currentlabels=np.copy(tempdisplayimg)
    originbinaryimg=np.copy(tempdisplayimg)

    tempcolorimg=np.copy(displaylabels).astype('float32')
    ratio=batch_findratio([tempdisplayimg.shape[0],tempdisplayimg.shape[1]],[850,850])

    if tempdisplayimg.shape[0]*tempdisplayimg.shape[1]<850*850:
        tempdisplayimg=cv2.resize(tempdisplayimg,(int(tempdisplayimg.shape[1]*ratio),int(tempdisplayimg.shape[0]*ratio)))
        colordivimg=cv2.resize(tempcolorimg,(int(colordivimg.shape[1]*ratio),int(colordivimg.shape[0]*ratio)))
    else:
        tempdisplayimg=cv2.resize(tempdisplayimg,(int(tempdisplayimg.shape[1]/ratio),int(tempdisplayimg.shape[0]/ratio)))
        colordivimg=cv2.resize(tempcolorimg,(int(colordivimg.shape[1]/ratio),int(colordivimg.shape[0]/ratio)))
    binaryimg=np.zeros((tempdisplayimg.shape[0],tempdisplayimg.shape[1],3))
    colordeimg=np.zeros((colordivimg.shape[0],colordivimg.shape[1],3))
    locs=np.where(tempdisplayimg==1)
    binaryimg[locs]=[240,228,66]
    for i in range(kmeans):
        locs=np.where(colordivimg==i)
        colordeimg[locs]=batch_colorbandtable[i]
    Image.fromarray(colordeimg.astype('uint8')).save(file+'-allcolorindex.png',"PNG")
    Image.fromarray((binaryimg.astype('uint8'))).save(file+'-binaryimg.png',"PNG")

    return currentlabels,originbinaryimg

def batch_extraction(currentlabels,file):
    global batch_results
    if kmeans==1:
        messagebox.showerror('Invalid Class #',message='#Class = 1, try change it to 2 or more, and refresh Color-Index.')
        return
    nonzeros=np.count_nonzero(currentlabels)
    print('nonzero counts',nonzeros)
    nonzeroloc=np.where(currentlabels!=0)
    try:
        ulx,uly=min(nonzeroloc[1]),min(nonzeroloc[0])
    except:
        messagebox.showerror('Invalid Colorindices',message='Need to process colorindicies')
        return
    rlx,rly=max(nonzeroloc[1]),max(nonzeroloc[0])
    nonzeroratio=float(nonzeros)/((rlx-ulx)*(rly-uly))
    print(nonzeroratio)

    dealpixel=nonzeroratio*currentlabels.shape[0]*currentlabels.shape[1]
    ratio=1
    if nonzeroratio<=0.2:# and nonzeroratio>=0.1:
        ratio=batch_findratio([currentlabels.shape[0],currentlabels.shape[1]],[1600,1600])
        if currentlabels.shape[0]*currentlabels.shape[1]>1600*1600:
            workingimg=cv2.resize(currentlabels,(int(currentlabels.shape[1]/ratio),int(currentlabels.shape[0]/ratio)),interpolation=cv2.INTER_LINEAR)
        else:
            #ratio=1
            #print('nonzeroratio',ratio)
            workingimg=np.copy(currentlabels)
        segmentratio=0
    else:
        print('deal pixel',dealpixel)
        if dealpixel>512000:
            if currentlabels.shape[0]*currentlabels.shape[1]>850*850:
                segmentratio=batch_findratio([currentlabels.shape[0],currentlabels.shape[1]],[850,850])
                if segmentratio<2:
                    segmentratio=2
                workingimg=cv2.resize(currentlabels,(int(currentlabels.shape[1]/segmentratio),int(currentlabels.shape[0]/segmentratio)),interpolation=cv2.INTER_LINEAR)
        else:
            segmentratio=1
            #print('ratio',ratio)
            workingimg=np.copy(currentlabels)
    pixelmmratio=1.0
    coin=False
    print('nonzeroratio:',ratio,'segmentation ratio',segmentratio)
    print('workingimgsize:',workingimg.shape)
    pyplt.imsave('workingimg.png',workingimg)
    originlabels=None
    if originlabels is None:
        originlabels,border,colortable,originlabeldict=tkintercorestat.init(workingimg,workingimg,'',workingimg,10,coin)

    batch_results.update({file:(originlabeldict,{})})

def batch_proc_func(file):
    if len(FOLDER)==0:
        messagebox.showerror('No image folder','Need to assign image folder')
        return
    if len(exportpath)==0:
        messagebox.showerror('No output folder','Need to assign output folder')
        return
    if len(pcs)==0:
        messagebox.showerror('No batch file','Need to load batch file')
        return
    procobj=batch_ser_func(file)
    procobj.process()
    del procobj

def batch_process():
    global batch_colordicesband
    global batch_Multiimage,batch_Multigray,batch_Multitype,batch_Multiimagebands,batch_Multigraybands
    global batch_displaybandarray,batch_originbandarray,batch_originpcabands
    global batch_results

    if len(batch_filenames)==0:
        messagebox.showerror('No files','Please load images to process')
        return
    cpunum=multiprocessing.cpu_count()
    print('# of CPUs',cpunum)
    starttime=time.time()
    print('start time',starttime)
    docneedskmeansadj=[]
    for file in batch_filenames:
        # batch_Multiimage={}
        # batch_Multigray={}
        # batch_Multitype={}
        # batch_Multiimagebands={}
        # batch_Multigraybands={}
        #
        # batch_displaybandarray={}
        # batch_originbandarray={}
        # batch_originpcabands={}
        # batch_colordicesband={}
        #
        # batch_results={}
        # if Open_batchimage(FOLDER,file)==False:
        #     continue
        # batch_singleband(file)
        # colordicesband=batch_kmeansclassify(file)
        # batch_colordicesband.update({file:colordicesband})
        # currentlabels,originbinaryimg=batch_generateimgplant(colordicesband,file)
        # batch_extraction(currentlabels,file)
        # batch_export_result(exportpath,file)
        procobj=batch_ser_func(file)
        procobj.process()
        if procobj.needswitchkmeanssel==True:
            docneedskmeansadj.append(file)
        del procobj
    # multi_pool=multiprocessing.Pool(int(cpunum/4))
    # multi_pool.map(batch_proc_func,batch_filenames)
    if len(docneedskmeansadj)>0:
        messagebox.showinfo('Image process area error',
                            'The image process area density are much greater than the batch document,\n '
                            'please adjust your cluster selection or use anoter batch file: \n'
                            +'\n'.join(docneedskmeansadj))
    print('used time',time.time()-starttime)


    messagebox.showinfo('Done','Batch process ends!')

def batch_showcounting(tup,number=True,frame=True,header=True,whext=False,blkext=False):
    labels=tup[0]
    colortable=tup[2]
    coinparts=tup[3]
    filename=tup[4]

    uniquelabels=list(colortable.keys())

    imgrsc=cv2.imread(FOLDER+'/'+filename,flags=cv2.IMREAD_ANYCOLOR)
    imgrsc=cv2.cvtColor(imgrsc,cv2.COLOR_BGR2RGB)
    imgrsc=cv2.resize(imgrsc,(labels.shape[1],labels.shape[0]),interpolation=cv2.INTER_LINEAR)
    image=Image.fromarray(imgrsc)
    if whext==True:
        # blkbkg=np.zeros((labels.shape[0],labels.shape[1],3),dtype='float')
        whbkg=np.zeros((labels.shape[0],labels.shape[1],3),dtype='float')
        whbkg[:,:,:]=[255,255,255]
        itemlocs=np.where(labels!=0)
        # blkbkg[itemlocs]=imgrsc[itemlocs]
        whbkg[itemlocs]=imgrsc[itemlocs]
        image=Image.fromarray(whbkg.astype('uint8'))
    if blkext==True:
        blkbkg=np.zeros((labels.shape[0],labels.shape[1],3),dtype='float')
        itemlocs=np.where(labels!=0)
        blkbkg[itemlocs]=imgrsc[itemlocs]
        blkbkg[itemlocs]=imgrsc[itemlocs]
        image=Image.fromarray(blkbkg.astype('uint8'))

    print('showcounting_resize',image.size)
    image.save('beforlabel.gif',append_images=[image])
    draw=ImageDraw.Draw(image)
    sizeuniq,sizecounts=np.unique(labels,return_counts=True)
    minsize=min(sizecounts)
    suggsize=int(minsize**0.5)
    if suggsize>22:
        suggsize=22
    if suggsize<14:
        suggsize=14
    font=ImageFont.truetype('cmb10.ttf',size=suggsize)

    for uni in uniquelabels:
        if uni!=0:
            pixelloc = np.where(labels == uni)
            try:
                ulx = min(pixelloc[1])
            except:
                continue
            uly = min(pixelloc[0])
            rlx = max(pixelloc[1])
            rly = max(pixelloc[0])
            midx = ulx + int((rlx - ulx) / 2)
            midy = uly + int((rly - uly) / 2)
            print(ulx, uly, rlx, rly)
            if frame==True:
                draw.polygon([(ulx,uly),(rlx,uly),(rlx,rly),(ulx,rly)],outline='red')
            if number==True:
                if uni in colortable:
                    canvastext = str(colortable[uni])
                else:
                    canvastext = 'No label'
                # if imgtypevar.get()=='0':
                draw.text((midx-1, midy+1), text=canvastext, font=font, fill='white')
                draw.text((midx+1, midy+1), text=canvastext, font=font, fill='white')
                draw.text((midx-1, midy-1), text=canvastext, font=font, fill='white')
                draw.text((midx+1, midy-1), text=canvastext, font=font, fill='white')
                #draw.text((midx,midy),text=canvastext,font=font,fill=(141,2,31,0))
                draw.text((midx,midy),text=canvastext,font=font,fill='black')


    if header==True:
        content='item count:'+str(len(uniquelabels))+'\n File: '+filename
        contentlength=len(content)+50
        #rectext=canvas.create_text(10,10,fill='black',font='Times 16',text=content,anchor=NW)
        draw.text((10-1, 10+1), text=content, font=font, fill='white')
        draw.text((10+1, 10+1), text=content, font=font, fill='white')
        draw.text((10-1, 10-1), text=content, font=font, fill='white')
        draw.text((10+1, 10-1), text=content, font=font, fill='white')
        #draw.text((10,10),text=content,font=font,fill=(141,2,31,0))
        draw.text((10,10),text=content,font=font,fill='black')
    #image.save(originfile+'-countresult'+extension,"JPEG")
    #firstimg=Multigraybands[currentfilename]
    #height,width=firstimg.size
    height,width,channel=batch_displaybandarray[filename]['LabOstu'].shape
    ratio=batch_findratio([height,width],[850,850])
    #if labels.shape[0]*labels.shape[1]<850*850:
    #    disimage=image.resize([int(labels.shape[1]*ratio),int(labels.shape[0]*ratio)],resample=Image.BILINEAR)
    #else:
    #    disimage=image.resize([int(labels.shape[1]/ratio),int(labels.shape[0]/ratio)],resample=Image.BILINEAR)
    print('show counting ratio',ratio)
    if height*width<850*850:
        print('showcounting small')
        disimage=image.resize([int(width*ratio),int(height*ratio)],resample=Image.BILINEAR)
    else:
        print('showcounting big')
        disimage=image.resize([int(width/ratio),int(height/ratio)],resample=Image.BILINEAR)
    print('showcounting shape',disimage.size)
    displayoutput=ImageTk.PhotoImage(disimage)
    disimage.save('output.gif',append_images=[disimage])
    #image.save('originoutput.gif',append_images=[image])
    return displayoutput,image,disimage

def batch_savePCAimg(path,originfile,file):
    originpcabands=batch_displaybandarray[file]['LabOstu']
    pcah,pcaw,pcac=originpcabands.shape
    pcakeys=pcs
    tempband=np.zeros((pcah,pcaw,len(pcakeys)))
    for i in range(len(pcakeys)):
        channel=int(pcakeys[i])-1
        tempband[:,:,i]=tempband[:,:,i]+originpcabands[:,:,channel]
    displaylabels=np.mean(tempband,axis=2)
    # generateimgplant(displaylabels)
    # grayimg=(((displaylabels-displaylabels.min())/(displaylabels.max()-displaylabels.min()))*255.9).astype(np.uint8)
    # pyplt.imsave('k=1.png',displaylabels.astype('uint8'))
    # pyplt.imsave('k=1.png',grayimg)
    grayimg=Image.fromarray(displaylabels.astype('uint8'),'L')
    originheight,originwidth=batch_Multigraybands[file].size
    origingray=grayimg.resize([originwidth,originheight],resample=Image.BILINEAR)
    origingray.save(path+'/'+originfile+'-PCAimg.png',"PNG")
    # addcolorstrip()
    return

def batch_export_ext(path,file,whext=False,blkext=False):
    global kernersizes
    if len(batch_filenames)==0:
        messagebox.showerror('No files','Please load images to process')
        return
    suggsize=8
    smallfont=ImageFont.truetype('cmb10.ttf',size=suggsize)
    # kernersizes={}
    # for file in batch_filenames:
    labeldict=batch_results[file][0]
    itervalue='iter0'
    labels=labeldict[itervalue]['labels']
    counts=labeldict[itervalue]['counts']
    colortable=labeldict[itervalue]['colortable']
    head_tail=os.path.split(file)
    originfile,extension=os.path.splitext(head_tail[1])
    if len(path)>0:
        tup=(labels,counts,colortable,[],file)
        _band,segimg,small_segimg=batch_showcounting(tup,False,True,True,whext,blkext)
        imageband=segimg
        draw=ImageDraw.Draw(imageband)
        uniquelabels=list(colortable.keys())
        tempdict={}
        pixelmmratio=1.0
        print('pixelmmratio',pixelmmratio)
        if file not in kernersizes:
            for uni in uniquelabels:
                if uni !=0:
                    pixelloc = np.where(labels == float(uni))
                    try:
                        ulx = min(pixelloc[1])
                    except:
                        continue
                    uly = min(pixelloc[0])
                    rlx = max(pixelloc[1])
                    rly = max(pixelloc[0])
                    print(ulx, uly, rlx, rly)
                    midx = ulx + int((rlx - ulx) / 2)
                    midy = uly + int((rly - uly) / 2)
                    length={}
                    currborder=tkintercore.get_boundaryloc(labels,uni)
                    # print('currborder',currborder)
                    print('currborder length',len(currborder[0])*len(currborder[1]))
                    pixperc=float(len(pixelloc[0])/(labels.shape[0]*labels.shape[1]))
                    print('pix length percentage',pixperc)
                    if pixperc>0.06:
                        x0=ulx
                        y0=uly
                        x1=rlx
                        y1=rly
                        kernellength=float(((x0-x1)**2+(y0-y1)**2)**0.5)
                    else:
                        for i in range(len(currborder[0])):
                            for j in range(i+1,len(currborder[0])):
                                templength=float(((currborder[0][i]-currborder[0][j])**2+(currborder[1][i]-currborder[1][j])**2)**0.5)
                                length.update({(i,j):templength})
                        sortedlength=sorted(length,key=length.get,reverse=True)
                        try:
                            topcouple=sortedlength[0]
                        except:
                            continue
                        kernellength=length[topcouple]
                        i=topcouple[0]
                        j=topcouple[1]
                        x0=currborder[1][i]
                        y0=currborder[0][i]
                        x1=currborder[1][j]
                        y1=currborder[0][j]
                        #slope=float((y0-y1)/(x0-x1))
                        linepoints=[(currborder[1][i],currborder[0][i]),(currborder[1][j],currborder[0][j])]
                        #draw.line(linepoints,fill='yellow')
                        #points=linepixels(currborder[1][i],currborder[0][i],currborder[1][j],currborder[0][j])

                    lengthpoints=cal_kernelsize.bresenhamline(x0,y0,x1,y1)  #x0,y0,x1,y1
                    for point in lengthpoints:
                        # if imgtypevar.get()=='0':
                        draw.point([int(point[0]),int(point[1])],fill='yellow')
                    tengentaddpoints=cal_kernelsize.tengentadd(x0,y0,x1,y1,rlx,rly,labels,uni) #find tangent line above
                    #for point in tengentaddpoints:
                        #if int(point[0])>=ulx and int(point[0])<=rlx and int(point[1])>=uly and int(point[1])<=rly:
                    #    draw.point([int(point[0]),int(point[1])],fill='green')
                    tengentsubpoints=cal_kernelsize.tengentsub(x0,y0,x1,y1,ulx,uly,labels,uni) #find tangent line below
                    #for point in tengentsubpoints:
                    #    draw.point([int(point[0]),int(point[1])],fill='green')
                    pointmatchdict={}
                    for i in range(len(tengentaddpoints)):  #find the pixel pair with shortest distance
                        width=kernellength
                        pointmatch=[]
                        point=tengentaddpoints[i]
                        try:
                            templabel=labels[int(point[1]),int(point[0])]
                        except:
                            continue
                        if templabel==uni:
                            for j in range(len(tengentsubpoints)):
                                subpoint=tengentsubpoints[j]
                                tempwidth=float(((point[0]-subpoint[0])**2+(point[1]-subpoint[1])**2)**0.5)
                                if tempwidth<width:
                                    pointmatch[:]=[]
                                    pointmatch.append(point)
                                    pointmatch.append(subpoint)
                                    #print('tempwidth',width)
                                    width=tempwidth
                        if len(pointmatch)>0:
                            #print('pointmatch',pointmatch)
                            pointmatchdict.update({(pointmatch[0],pointmatch[1]):width})
                    widthsort=sorted(pointmatchdict,key=pointmatchdict.get,reverse=True)
                    try:
                        pointmatch=widthsort[0]
                        print('final pointmatch',pointmatch)
                    except:
                        continue
                    if len(pointmatch)>0:
                        x0=int(pointmatch[0][0])
                        y0=int(pointmatch[0][1])
                        x1=int(pointmatch[1][0])
                        y1=int(pointmatch[1][1])
                        # if imgtypevar.get()=='0':
                        draw.line([(x0,y0),(x1,y1)],fill='yellow')
                        width=float(((x0-x1)**2+(y0-y1)**2)**0.5)
                        print('width',width,'length',kernellength)
                        print('kernelwidth='+str(width*pixelmmratio))
                        print('kernellength='+str(kernellength*pixelmmratio))
                        #print('kernelwidth='+str(kernelwidth*pixelmmratio))
                        tempdict.update({uni:[kernellength,width,pixelmmratio**2*len(pixelloc[0]),kernellength*pixelmmratio,width*pixelmmratio]})
                    if uni in colortable:
                        canvastext = str(colortable[uni])
                    else:
                        canvastext = 'No label'
                    # if imgtypevar.get()=='0':
                    draw.text((midx-1, midy+1), text=canvastext, font=smallfont, fill='white')
                    draw.text((midx+1, midy+1), text=canvastext, font=smallfont, fill='white')
                    draw.text((midx-1, midy-1), text=canvastext, font=smallfont, fill='white')
                    draw.text((midx+1, midy-1), text=canvastext, font=smallfont, fill='white')
                    #draw.text((midx,midy),text=canvastext,font=font,fill=(141,2,31,0))
                    draw.text((midx,midy),text=canvastext,font=smallfont,fill='black')

                    #print(event.x, event.y, labels[event.x, event.y], ulx, uly, rlx, rly)

                    #recborder = canvas.create_rectangle(ulx, uly, rlx, rly, outline='red')
                    #drawcontents.append(recborder)
            kernersizes.update({file:tempdict})
        originheight,originwidth=batch_Multigraybands[file].size
        image=imageband.resize([originwidth,originheight],resample=Image.BILINEAR)
        extcolor=""
        if whext==True:
            extcolor= "-extwht"
        if blkext==True:
            extcolor="-extblk"
        image.save(path+'/'+originfile+extcolor+'-sizeresult'+'.png',"PNG")
        tup=(labels,counts,colortable,[],file)
        _band,segimg,small_segimg=batch_showcounting(tup,False,True,True,whext,blkext)
        segimage=segimg.resize([originwidth,originheight],resample=Image.BILINEAR)
        segimage.save(path+'/'+originfile+extcolor+'-segmentresult'+'.png',"PNG")
        _band,segimg,small_segimg=batch_showcounting(tup,True,True,True,whext,blkext)
        segimage=segimg.resize([originwidth,originheight],resample=Image.BILINEAR)
        segimage.save(path+'/'+originfile+extcolor+'-labelresult'+'.png',"PNG")





def batch_export_result(path,file):
    global kernersizes
    if len(batch_filenames)==0:
        messagebox.showerror('No files','Please load images to process')
        return
    suggsize=8
    smallfont=ImageFont.truetype('cmb10.ttf',size=suggsize)
    kernersizes={}
    # path=filedialog.askdirectory()
    batch_export_ext(path,file,True,False)
    batch_export_ext(path,file,False,True)
    # for file in batch_filenames:
    labeldict=batch_results[file][0]
    itervalue='iter0'
    labels=labeldict[itervalue]['labels']
    counts=labeldict[itervalue]['counts']
    colortable=labeldict[itervalue]['colortable']
    head_tail=os.path.split(file)
    originfile,extension=os.path.splitext(head_tail[1])
    if len(path)>0:
        tup=(labels,counts,colortable,[],file)
        _band,segimg,small_segimg=batch_showcounting(tup,False)
        #imageband=outputimgbands[file][itervalue]
        imageband=segimg
        draw=ImageDraw.Draw(imageband)
        uniquelabels=list(colortable.keys())
        # tempdict={}
        pixelmmratio=1.0
        #print('coinsize',coinsize.get(),'pixelmmratio',pixelmmratio)
        print('pixelmmratio',pixelmmratio)
        # for uni in uniquelabels:
        #     if uni !=0:
        #         pixelloc = np.where(labels == float(uni))
        #         try:
        #             ulx = min(pixelloc[1])
        #         except:
        #             continue
        #         uly = min(pixelloc[0])
        #         rlx = max(pixelloc[1])
        #         rly = max(pixelloc[0])
        #         print(ulx, uly, rlx, rly)
        #         midx = ulx + int((rlx - ulx) / 2)
        #         midy = uly + int((rly - uly) / 2)
        #         length={}
        #         currborder=tkintercore.get_boundaryloc(labels,uni)
        #         for i in range(len(currborder[0])):
        #             for j in range(i+1,len(currborder[0])):
        #                 templength=float(((currborder[0][i]-currborder[0][j])**2+(currborder[1][i]-currborder[1][j])**2)**0.5)
        #                 length.update({(i,j):templength})
        #         sortedlength=sorted(length,key=length.get,reverse=True)
        #         try:
        #             topcouple=sortedlength[0]
        #         except:
        #             continue
        #         kernellength=length[topcouple]
        #         i=topcouple[0]
        #         j=topcouple[1]
        #         x0=currborder[1][i]
        #         y0=currborder[0][i]
        #         x1=currborder[1][j]
        #         y1=currborder[0][j]
        #         #slope=float((y0-y1)/(x0-x1))
        #         linepoints=[(currborder[1][i],currborder[0][i]),(currborder[1][j],currborder[0][j])]
        #         #draw.line(linepoints,fill='yellow')
        #         #points=linepixels(currborder[1][i],currborder[0][i],currborder[1][j],currborder[0][j])
        #
        #         lengthpoints=cal_kernelsize.bresenhamline(x0,y0,x1,y1)  #x0,y0,x1,y1
        #         for point in lengthpoints:
        #             # if imgtypevar.get()=='0':
        #             draw.point([int(point[0]),int(point[1])],fill='yellow')
        #         tengentaddpoints=cal_kernelsize.tengentadd(x0,y0,x1,y1,rlx,rly,labels,uni) #find tangent line above
        #         #for point in tengentaddpoints:
        #             #if int(point[0])>=ulx and int(point[0])<=rlx and int(point[1])>=uly and int(point[1])<=rly:
        #         #    draw.point([int(point[0]),int(point[1])],fill='green')
        #         tengentsubpoints=cal_kernelsize.tengentsub(x0,y0,x1,y1,ulx,uly,labels,uni) #find tangent line below
        #         #for point in tengentsubpoints:
        #         #    draw.point([int(point[0]),int(point[1])],fill='green')
        #         pointmatchdict={}
        #         for i in range(len(tengentaddpoints)):  #find the pixel pair with shortest distance
        #             width=kernellength
        #             pointmatch=[]
        #             point=tengentaddpoints[i]
        #             try:
        #                 templabel=labels[int(point[1]),int(point[0])]
        #             except:
        #                 continue
        #             if templabel==uni:
        #                 for j in range(len(tengentsubpoints)):
        #                     subpoint=tengentsubpoints[j]
        #                     tempwidth=float(((point[0]-subpoint[0])**2+(point[1]-subpoint[1])**2)**0.5)
        #                     if tempwidth<width:
        #                         pointmatch[:]=[]
        #                         pointmatch.append(point)
        #                         pointmatch.append(subpoint)
        #                         #print('tempwidth',width)
        #                         width=tempwidth
        #             if len(pointmatch)>0:
        #                 #print('pointmatch',pointmatch)
        #                 pointmatchdict.update({(pointmatch[0],pointmatch[1]):width})
        #         widthsort=sorted(pointmatchdict,key=pointmatchdict.get,reverse=True)
        #         try:
        #             pointmatch=widthsort[0]
        #             print('final pointmatch',pointmatch)
        #         except:
        #             continue
        #         if len(pointmatch)>0:
        #             x0=int(pointmatch[0][0])
        #             y0=int(pointmatch[0][1])
        #             x1=int(pointmatch[1][0])
        #             y1=int(pointmatch[1][1])
        #             # if imgtypevar.get()=='0':
        #             draw.line([(x0,y0),(x1,y1)],fill='yellow')
        #             width=float(((x0-x1)**2+(y0-y1)**2)**0.5)
        #             print('width',width,'length',kernellength)
        #             print('kernelwidth='+str(width*pixelmmratio))
        #             print('kernellength='+str(kernellength*pixelmmratio))
        #             #print('kernelwidth='+str(kernelwidth*pixelmmratio))
        #             tempdict.update({uni:[kernellength,width,pixelmmratio**2*len(pixelloc[0]),kernellength*pixelmmratio,width*pixelmmratio]})
        #         if uni in colortable:
        #             canvastext = str(colortable[uni])
        #         else:
        #             canvastext = 'No label'
        #         # if imgtypevar.get()=='0':
        #         draw.text((midx-1, midy+1), text=canvastext, font=smallfont, fill='white')
        #         draw.text((midx+1, midy+1), text=canvastext, font=smallfont, fill='white')
        #         draw.text((midx-1, midy-1), text=canvastext, font=smallfont, fill='white')
        #         draw.text((midx+1, midy-1), text=canvastext, font=smallfont, fill='white')
        #         #draw.text((midx,midy),text=canvastext,font=font,fill=(141,2,31,0))
        #         draw.text((midx,midy),text=canvastext,font=smallfont,fill='black')

                #print(event.x, event.y, labels[event.x, event.y], ulx, uly, rlx, rly)

                #recborder = canvas.create_rectangle(ulx, uly, rlx, rly, outline='red')
                #drawcontents.append(recborder)

        # kernersizes.update({file:tempdict})
        originheight,originwidth=batch_Multigraybands[file].size
        image=imageband.resize([originwidth,originheight],resample=Image.BILINEAR)
        image.save(path+'/'+originfile+'-sizeresult'+'.png',"PNG")
        tup=(labels,counts,colortable,[],file)
        _band,segimg,small_segimg=batch_showcounting(tup,False)
        segimage=segimg.resize([originwidth,originheight],resample=Image.BILINEAR)
        segimage.save(path+'/'+originfile+'-segmentresult'+'.png',"PNG")
        _band,segimg,small_segimg=batch_showcounting(tup,True)
        segimage=segimg.resize([originwidth,originheight],resample=Image.BILINEAR)
        segimage.save(path+'/'+originfile+'-labelresult'+'.png',"PNG")
        originrestoredband=np.copy(labels)
        restoredband=originrestoredband.astype('uint8')
        colordicesband=batch_colordicesband[file]
        colordiv=np.zeros((colordicesband.shape[0],colordicesband.shape[1],3))
        batch_savePCAimg(path,originfile,file)

        # kvar=int(kmeans.get())
        # print('kvar',kvar)
        # for i in range(kvar):
        #     locs=np.where(colordicesband==i)
        #     colordiv[locs]=colorbandtable[i]
        # colordivimg=Image.fromarray(colordiv.astype('uint8'))
        # colordivimg.save(path+'/'+originfile+'-colordevice'+'.jpeg',"JPEG")
        colordivimg=Image.open(file+'-allcolorindex.png')
        copycolordiv=colordivimg.resize([originwidth,originheight],resample=Image.BILINEAR)
        copycolordiv.save(path+'/'+originfile+'-colordevice'+'.png',"PNG")
        #pyplt.imsave(path+'/'+originfile+'-colordevice'+'.png',colordiv.astype('uint8'))
        # copybinary=np.zeros((originbinaryimg.shape[0],originbinaryimg.shape[1],3),dtype='float')
        # nonzeros=np.where(originbinaryimg==1)
        # copybinary[nonzeros]=[255,255,0]
        # binaryimg=Image.fromarray(copybinary.astype('uint8'))
        binaryimg=Image.open(file+'-binaryimg.png')
        copybinaryimg=binaryimg.resize([originwidth,originheight],resample=Image.BILINEAR)
        copybinaryimg.save(path+'/'+originfile+'-binaryimg'+'.png',"PNG")
        # pyplt.imsave(path+'/'+originfile+'-binaryimg'+'.png',originbinaryimg.astype('uint8'))

        #restoredband=cv2.resize(src=restoredband,dsize=(originwidth,originheight),interpolation=cv2.INTER_LINEAR)
        print(restoredband.shape)
        currentsizes=kernersizes[file]
        indicekeys=list(batch_originbandarray[file].keys())
        indeclist=[ 0 for i in range(len(indicekeys)*3)]
        pcalist=[0 for i in range(3)]
        temppcabands=np.zeros((batch_originpcabands[file].shape[0],len(pcs)))
        for i in range(len(pcs)):
            temppcabands[:,i]=temppcabands[:,i]+batch_originpcabands[file][:,pcs[i]-1]
        pcabands=np.mean(temppcabands,axis=1)
        pcabands=pcabands.reshape((originheight,originwidth))
        datatable={}
        origindata={}
        for key in indicekeys:
            data=batch_originbandarray[file][key]
            data=data.tolist()
            tempdict={key:data}
            origindata.update(tempdict)
            print(key)
        # for uni in colortable:
        print(uniquelabels)
        print('len uniquelabels',len(uniquelabels))
        for uni in uniquelabels:
            print(uni,colortable[uni])
            uniloc=np.where(labels==float(uni))
            if len(uniloc)==0 or len(uniloc[1])==0:
                print('no uniloc\n')
                print(uniloc[0],uniloc[1])
                continue
            smalluniloc=np.where(originrestoredband==uni)
            ulx,uly=min(smalluniloc[1]),min(smalluniloc[0])
            rlx,rly=max(smalluniloc[1]),max(smalluniloc[0])
            width=rlx-ulx
            length=rly-uly
            print(width,length)
            subarea=restoredband[uly:rly+1,ulx:rlx+1]
            subarea=subarea.tolist()
            amount=len(uniloc[0])
            print(amount)
            try:
                sizes=currentsizes[uni]
            except:
                print('no sizes\n')
                continue
            #templist=[amount,length,width]
            templist=[amount,sizes[0],sizes[1],sizes[2],sizes[3],sizes[4]]
            tempdict={colortable[uni]:templist+indeclist+pcalist}  #NIR,Redeyes,R,G,B,NDVI,area
            print(tempdict)
            for ki in range(len(indicekeys)):
                originNDVI=origindata[indicekeys[ki]]
                print(len(originNDVI),len(originNDVI[0]))
                pixellist=[]
                for k in range(len(uniloc[0])):
                    #print(uniloc[0][k],uniloc[1][k])
                    try:
                        tempdict[colortable[uni]][6+ki*3]+=originNDVI[uniloc[0][k]][uniloc[1][k]]
                    except IndexError:
                        print(uniloc[0][k],uniloc[1][k])
                    tempdict[colortable[uni]][7+ki*3]+=originNDVI[uniloc[0][k]][uniloc[1][k]]
                    pixellist.append(originNDVI[uniloc[0][k]][uniloc[1][k]])
                tempdict[colortable[uni]][ki*3+6]=tempdict[colortable[uni]][ki*3+6]/amount
                tempdict[colortable[uni]][ki*3+8]=np.std(pixellist)
            pixellist=[]
            for k in range(len(uniloc[0])):
                try:
                    tempdict[colortable[uni]][-2]+=pcabands[uniloc[0][k]][uniloc[1][k]]
                except IndexError:
                    print(uniloc[0][k],uniloc[1][k])
                tempdict[colortable[uni]][-3]+=pcabands[uniloc[0][k]][uniloc[1][k]]
                pixellist.append(pcabands[uniloc[0][k]][uniloc[1][k]])
                tempdict[colortable[uni]][-3]=tempdict[colortable[uni]][-3]/amount
                tempdict[colortable[uni]][-1]=np.std(pixellist)
            datatable.update(tempdict)
        filename=path+'/'+originfile+'-outputdata.csv'
        with open(filename,mode='w') as f:
            csvwriter=csv.writer(f)
            rowcontent=['Index','Plot','Area(#pixel)','Length(#pixel)','Width(#pixel)','Area(mm2)','Length(mm)','Width(mm)']
            for key in indicekeys:
                rowcontent.append('avg-'+str(key))
                rowcontent.append('sum-'+str(key))
                rowcontent.append('std-'+str(key))
            rowcontent.append('avg-PCA')
            rowcontent.append('sum-PCA')
            rowcontent.append('std-PCA')
            #csvwriter.writerow(['ID','NIR','Red Edge','Red','Green','Blue','NIRv.s.Green','LabOstu','area(#of pixel)'])
            #csvwriter.writerow(['Index','Plot','Area(#pixels)','avg-NDVI','sum-NDVI','std-NDVI','Length(#pixel)','Width(#pixel)'])#,'#holes'])
            csvwriter.writerow(rowcontent)
            i=1
            for uni in datatable:
                row=[i,uni]
                for j in range(len(datatable[uni])):
                    row.append(datatable[uni][j])
                #row=[i,uni,datatable[uni][0],datatable[uni][1],datatable[uni][2],datatable[uni][5],datatable[uni][3],datatable[uni][4]]#,
                     #datatable[uni][5]]
                i+=1
                print(row)
                csvwriter.writerow(row)
        print('total data length=',len(datatable))
    # messagebox.showinfo('Saved',message='Results are saved to '+path)
    # tx=root.winfo_x()
    # ty=root.winfo_y()
    # top=Toplevel()
    # top.attributes("-topmost",True)
    # w = 300
    # h = 150
    # dx=100
    # dy=100
    # top.geometry("%dx%d+%d+%d" % (w, h, tx + dx, ty + dy))
    # top.title('Saved')
    # Message(top,text='Results are saved to '+path,padx=20,pady=20).pack()
    # okbut=Button(top,text='Okay',command=top.destroy)
    # okbut.pack(side=BOTTOM)
    # top.after(10000,top.destroy)
    # batchfile=path+'/'+originfile+'-batch'+'.txt'
    # with open(batchfile,'w') as f:
    #     for key in batch.keys():
    #         f.write(key)
    #         f.write(',')
    #         for i in range(len(batch[key])):
    #             f.write(str(batch[key][i]))
    #             f.write(',')
    #         f.write('\n')
    #     f.close()

def batch_exportpath():
    global exportpath
    exportpath=filedialog.askdirectory()
    while len(exportpath)==0:
        exportpath=filedialog.askdirectory()





