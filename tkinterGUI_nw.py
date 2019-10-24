from tkinter import *
from tkinter import ttk
import tkinter.filedialog as filedialog
from tkinter import messagebox

from PIL import Image,ImageDraw,ImageFont
from PIL import ImageTk
import cv2
from skimage import filters
import rasterio
import matplotlib.pyplot as pyplt

import numpy as np
import os
import time
import csv

from functools import partial
import sys

import kplus
from sklearn.cluster import KMeans
import tkintercorestat
import tkintercore
import cal_kernelsize

class img():
    def __init__(self,size,bands):
        self.size=size
        self.bands=bands

displayimg={'Origin':None,
            'Gray/NIR':None,
            'ColorIndices':None,
            'Output':None}
cluster=['LabOstu','NDI'] #,'Greenness','VEG','CIVE','MExG','NDVI','NGRDI','HEIGHT']
filenames=[]

Multiimage={}
Multigray={}
Multitype={}
Multiimagebands={}
Multigraybands={}
workbandarray={}
displaybandarray={}
originbandarray={}
clusterdisplay={}
kernersizes={}
multi_results={}
outputimgdict={}
outputimgbands={}

root=Tk()
root.title('GridFree')
root.geometry("")
root.option_add('*tearoff',False)
emptymenu=Menu(root)
root.config(menu=emptymenu)

refvar=StringVar()
edge=StringVar()
kmeans=IntVar()
filedropvar=StringVar()
displaybut_var=StringVar()
bandchoice={}
checkboxdict={}

minipixelareaclass=0

currentfilename='seedsample.JPG'
currentlabels=None
workingimg=None
## Funcitons
def distance(p1,p2):
    return np.sum((p1-p2)**2)

def kplusinit(data,k):
    centroid=[]
    centroid.append(data[np.random.randint(data.shape[0]),:])
    for c_id in range(k-1):
        dist=[]
        for i in range(data.shape[0]):
            point=data[i,:]
            d=sys.maxsize

            for j in range(len(centroid)):
                temp_dist=distance(point,centroid[j])
                d=min(d,temp_dist)
            dist.append(d)

        dist=np.array(dist)
        next_centroid=data[np.argmax(dist),:]
        centroid.append(next_centroid)
    return centroid

def kmeansplus(data,k):
    means=kplusinit(data,k)
    clusters=kplus.FindClusters(means,data)
    print(cluster)



def findratio(originsize,objectsize):
    if originsize[0]>objectsize[0] or originsize[1]>objectsize[1]:
        ratio=round(max(originsize[0]/objectsize[0],originsize[1]/objectsize[1]))
    else:
        ratio=round(min(objectsize[0]/originsize[0],objectsize[1]/originsize[1]))
    return ratio


def changedisplayimg(frame,text):
    global displaybut_var
    displaybut_var.set(disbuttonoption[text])
    for widget in frame.winfo_children():
        widget.pack_forget()
    widget.configure(image=displayimg[text])
    widget.image=displayimg[text]
    widget.pack()
    #print('change to '+text)
    #time.sleep(1)

def generatedisplayimg(filename):
    firstimg=Multiimagebands[filename]
    height,width=firstimg.size
    ratio=findratio([height,width],[620,620])
    resize=cv2.resize(Multiimage[filename],(int(width/ratio),int(height/ratio)),interpolation=cv2.INTER_LINEAR)
    rgbimg=ImageTk.PhotoImage(Image.fromarray(resize.astype('uint8')))
    displayimg['Origin']=rgbimg
    resize=cv2.resize(Multigray[filename],(int(width/ratio),int(height/ratio)),interpolation=cv2.INTER_LINEAR)
    grayimg=ImageTk.PhotoImage(Image.fromarray(resize.astype('uint8')))
    displayimg['Gray/NIR']=grayimg
    pyplt.imsave('displayimg.png',displaybandarray['LabOstu'])
    indimg=cv2.imread('displayimg.png')
    displayimg['ColorIndices']=ImageTk.PhotoImage(Image.fromarray(indimg))
    displayimg['Output']=ImageTk.PhotoImage(Image.fromarray(np.zeros((int(height/ratio),int(width/ratio))).astype('uint8')))


def Open_File(filename):   #add to multi-image,multi-gray  #call band calculation
    global Multiimage,Multigray,Multitype,Multiimagebands,Multigraybands
    try:
        Filersc=cv2.imread(filename)
        height,width,channel=np.shape(Filersc)
        Filesize=(height,width)
        RGBfile=cv2.cvtColor(Filersc,cv2.COLOR_BGR2RGB)
        Multiimage.update({filename:RGBfile})
        Grayfile=cv2.cvtColor(Filersc,cv2.COLOR_BGR2Lab)
        Grayfile=cv2.cvtColor(Grayfile,cv2.COLOR_BGR2GRAY)
        #Grayfile=cv2.GaussianBlur(Grayfile,(3,3),cv2.BORDER_DEFAULT)
        #ostu=filters.threshold_otsu(Grayfile)
        #Grayfile=Grayfile.astype('float32')
        #Grayfile=Grayfile/ostu
        Grayimg=img(Filesize,Grayfile)
        RGBbands=np.zeros((channel,height,width))
        for j in range(channel):
            band=RGBfile[:,:,j]
            band=np.where(band==0,1e-6,band)
            #ostu=filters.threshold_otsu(band)
            #band=band/ostu
            RGBbands[j,:,:]=band
        RGBimg=img(Filesize,RGBbands)
        tempdict={filename:RGBimg}
        Multiimagebands.update(tempdict)
        tempdict={filename:Grayfile}
        Multigray.update(tempdict)
        tempdict={filename:0}
        Multitype.update(tempdict)
        tempdict={filename:Grayimg}
        Multigraybands.update(tempdict)
    except:
        try:
            Filersc=rasterio.open(filename)
            height=Filersc.height
            width=Filersc.width
            channel=Filersc.count
            print(Filersc)
            Filebands=np.zeros((3,height,width))
            imagebands=np.zeros((height,width,3))
            grayimg=np.zeros((height,width,3))
            graybands=np.zeros((3,height,width))
            Filesize=(height,width)
            for j in range(channel):
                band=Filersc.read(j+1)
                if j<3:
                    imagebands[:,:,j]=band
                    band = np.where((band == 0)|(band==-10000) | (band==-9999), 1e-6, band)
                    Filebands[j,:,:]=band
                else:
                    grayimg[:,:,j-3]=band
                    band = np.where((band == 0)|(band==-10000) | (band==-9999), 1e-6, band)
                    graybands[j-3,:,:]=band

            Fileimg=img(Filesize,Filebands)
            tempdict={filename:imagebands}
            Multiimage.update(tempdict)
            tempdict={filename:Fileimg}
            Multiimagebands.update(tempdict)
            tempdict={filename:2}
            Multitype.update(tempdict)
            Grayim=img(Filesize,graybands)
            tempdict={filename:grayimg}
            Multigray.update(tempdict)
            tempdict={filename:Grayim}
            Multigraybands.update(tempdict)
            print(Filebands)
        except:
            messagebox.showerror('Invalid Filename','Cannot open '+filename)
            return
    filenames.append(filename)



def Open_Multifile():
    global Multiimage,Multigray,Multitype,Multiimagebands,changefileframe,imageframe,Multigraybands,filenames
    global changefiledrop,filedropvar,originbandarray,displaybandarray,clusterdisplay,currentfilename,resviewframe

    MULTIFILES=filedialog.askopenfilenames()
    if len(MULTIFILES)>0:
        Multiimage={}
        Multigray={}
        Multitype={}
        Multiimagebands={}
        Multigraybands={}
        filenames=[]
        originbandarray={}
        displaybandarray={}
        clusterdisplay={}
        for i in range(len(MULTIFILES)):
            Open_File(MULTIFILES[i])
        for widget in changefileframe.winfo_children():
            widget.pack_forget()
        filedropvar.set(filenames[0])
        changefiledrop=OptionMenu(changefileframe,filedropvar,*filenames,command=partial(changeimage,imageframe))
        changefiledrop.pack()
        singleband(filenames[0])
        generatedisplayimg(filenames[0])
        currentfilename=filenames[0]
        for i in range(len(cluster)):
            bandchoice[cluster[i]].set('')
        changedisplayimg(imageframe,'Origin')



def workbandsize(item):
    pass

def singleband(file):
    global displaybandarray,originbandarray
    bands=Multigraybands[file].bands
    bandsize=Multigraybands[file].size
    try:
        channel,height,width=bands.shape
    except:
        channel=0
    if channel>1:
        bands=bands[0,:,:]
    bands=cv2.GaussianBlur(bands,(3,3),cv2.BORDER_DEFAULT)
    ostu=filters.threshold_otsu(bands)
    bands=bands.astype('float32')
    bands=bands/ostu
    #display purpose
    ratio=findratio([bandsize[0],bandsize[1]],[620,620])
    if 'LabOstu' not in originbandarray:
        originbandarray.update({'LabOstu':bands})
        displaybands=cv2.resize(bands,(int(bandsize[1]/ratio),int(bandsize[0]/ratio)),interpolation=cv2.INTER_LINEAR)
        #displaybands=displaybands.reshape((int(bandsize[1]/ratio),int(bandsize[0]/ratio),3))
        #kernel=np.ones((2,2),np.float32)/4
        displaybandarray.update({'LabOstu':displaybands})
        #displaybandarray.update({'LabOstu':cv2.filter2D(displaybands,-1,kernel)})
    bands=Multiimagebands[file].bands
    for i in range(3):
        bands[i,:,:]=cv2.GaussianBlur(bands[i,:,:],(3,3),cv2.BORDER_DEFAULT)
    NDI=128*((bands[1,:,:]-bands[0,:,:])/(bands[1,:,:]+bands[0,:,:])+1)
    tempdict={'NDI':NDI}
    if 'NDI' not in originbandarray:
        originbandarray.update(tempdict)
        displaybands=cv2.resize(NDI,(int(bandsize[1]/ratio),int(bandsize[0]/ratio)),interpolation=cv2.INTER_LINEAR)
        kernel=np.ones((2,2),np.float32)/4
        displaydict={'NDI':cv2.filter2D(displaybands,-1,kernel)}
        #displaydict=displaydict.reshape((int(bandsize[1]/ratio),int(bandsize[0]/ratio),3))
        displaybandarray.update(displaydict)
    '''
    Greenness = bands[1, :, :] / (bands[0, :, :] + bands[1, :, :] + bands[2, :, :])
    tempdict = {'Greenness': Greenness}
    if 'Greenness' not in originbandarray:
        originbandarray.update(tempdict)
        image=cv2.resize(Greenness,(int(bandsize[1]/ratio),int(bandsize[0]/ratio)),interpolation=cv2.INTER_LINEAR)
        #image=image.reshape((int(bandsize[1]/ratio),int(bandsize[0]/ratio),3))
        worktempdict={'Greenness':image}
        displaybandarray.update(worktempdict)
    VEG=bands[1,:,:]/(np.power(bands[0,:,:],0.667)*np.power(bands[2,:,:],(1-0.667)))
    tempdict={'VEG':VEG}
    if 'VEG' not in originbandarray:
        originbandarray.update(tempdict)
        image=cv2.resize(VEG,(int(bandsize[1]/ratio),int(bandsize[0]/ratio)),interpolation=cv2.INTER_LINEAR)
        #image=image.reshape((int(bandsize[1]/ratio),int(bandsize[0]/ratio),3))
        worktempdict={'VEG':image}
        displaybandarray.update(worktempdict)
    CIVE=0.441*bands[0,:,:]-0.811*bands[1,:,:]+0.385*bands[2,:,:]+18.78745
    tempdict={'CIVE':CIVE}
    if 'CIVE' not in originbandarray:
        originbandarray.update(tempdict)
        image=cv2.resize(CIVE,(int(bandsize[1]/ratio),int(bandsize[0]/ratio)),interpolation=cv2.INTER_LINEAR)
        #image=image.reshape((int(bandsize[1]/ratio),int(bandsize[0]/ratio),3))
        worktempdict={'CIVE':image}
        displaybandarray.update(worktempdict)
    MExG=1.262*bands[1,:,:]-0.884*bands[0,:,:]-0.311*bands[2,:,:]
    tempdict={'MExG':MExG}
    if 'MExG' not in originbandarray:
        originbandarray.update(tempdict)
        image=cv2.resize(MExG,(int(bandsize[1]/ratio),int(bandsize[0]/ratio)),interpolation=cv2.INTER_LINEAR)
        #image=image.reshape((int(bandsize[1]/ratio),int(bandsize[0]/ratio),3))
        worktempdict={'MExG':image}
        displaybandarray.update(worktempdict)
    NDVI=(bands[0,:,:]-bands[2,:,:])/(bands[0,:,:]+bands[2,:,:])
    tempdict={'NDVI':NDVI}
    if 'NDVI' not in originbandarray:
        originbandarray.update(tempdict)
        image=cv2.resize(NDVI,(int(bandsize[1]/ratio),int(bandsize[0]/ratio)),interpolation=cv2.INTER_LINEAR)
        #image=image.reshape((int(bandsize[1]/ratio),int(bandsize[0]/ratio),3))
        worktempdict={'NDVI':image}
        displaybandarray.update(worktempdict)
    NGRDI=(bands[1,:,:]-bands[0,:,:])/(bands[1,:,:]+bands[0,:,:])
    tempdict={'NGRDI':NGRDI}
    if 'NGRDI' not in originbandarray:
        originbandarray.update(tempdict)
        image=cv2.resize(NGRDI,(int(bandsize[1]/ratio),int(bandsize[0]/ratio)),interpolation=cv2.INTER_LINEAR)
        #image=image.reshape((int(bandsize[1]/ratio),int(bandsize[0]/ratio),3))
        worktempdict={'NGRDI':image}
        displaybandarray.update(worktempdict)
    if channel>=1:
        nirbands=Multigraybands[file].bands
        NDVI=(nirbands[0,:,:]-bands[1,:,:])/(nirbands[0,:,:]+bands[1,:,:])
        tempdict={'NDVI':NDVI}
        #if 'NDVI' not in originbandarray:
        originbandarray.update(tempdict)
        image=cv2.resize(NDVI,(int(bandsize[1]/ratio),int(bandsize[0]/ratio)),interpolation=cv2.INTER_LINEAR)
        #image=image.reshape((int(bandsize[1]/ratio),int(bandsize[0]/ratio),3))
        worktempdict={'NDVI':image}
        displaybandarray.update(worktempdict)
    if channel==3:
        bands=Multigraybands[file].bands
        Height=bands[2,:,:]
        tempdict={'HEIGHT':Height}
        if 'HEIGHT' not in originbandarray:
            originbandarray.update(tempdict)
            image=cv2.resize(Height,(int(bandsize[1]/ratio),int(bandsize[0]/ratio)),interpolation=cv2.INTER_LINEAR)
            worktempdict={'HEIGHT':image}
            displaybandarray.update(worktempdict)
    else:
        originbandarray.update({'HEIGHT':np.zeros(bandsize)})
        image=np.zeros((int(bandsize[0]/ratio),int(bandsize[0]/ratio)))
        displaybandarray.update({'HEIGHT':image})
    '''









def Band_calculation():
    global originbandarray,workbandarray
    originbandarray={}
    workbandarray={}
    for file in filenames:
        singleband(file)




def changeimage(frame,filename):
    global clusterdisplay,currentfilename
    clusterdisplay={}
    currentfilename=filename
    print(filename)
    generatedisplayimg(filename)
    changedisplayimg(frame,'Origin')

def generateplant(checkbox,bandchoice):
    keys=bandchoice.keys()
    choicelist=[]
    imageband=np.zeros((displaybandarray['LabOstu'].shape))
    for key in keys:
        tup=bandchoice[key].get()
        if '1' in tup:
            choicelist.append(key)
            imageband=imageband+displaybandarray[key]
    if len(choicelist)==0:
        messagebox.showerror('No Indices is selected',message='Please select indicies to do KMeans Classification.')
        return
    if int(kmeans.get())==1:
        pyplt.imsave('displayimg.png',imageband)
        indimg=cv2.imread('displayimg.png')
        displayimg['ColorIndices']=ImageTk.PhotoImage(Image.fromarray(indimg))
        changedisplayimg(imageframe,'ColorIndices')

    else:
        if ''.join(choicelist) in clusterdisplay:
            tempdict=clusterdisplay[''.join(choicelist)]
            if kmeans.get() in tempdict:
                displaylabels=tempdict[kmeans.get()]
            else:
                reshapemodified_tif=np.zeros((displaybandarray['LabOstu'].shape[0]*displaybandarray['LabOstu'].shape[1],len(choicelist)))
                displaylabels=kmeansclassify(choicelist,reshapemodified_tif)
            generateimgplant(displaylabels)
            return
        else:
            reshapemodified_tif=np.zeros((displaybandarray['LabOstu'].shape[0]*displaybandarray['LabOstu'].shape[1],len(choicelist)))
            displaylabels=kmeansclassify(choicelist,reshapemodified_tif)
            generateimgplant(displaylabels)


def generatecheckbox(frame,classnum):
    global checkboxdict
    for widget in frame.winfo_children():
        widget.grid_forget()
    checkboxdict={}
    for i in range(int(classnum)):
        dictkey='class '+str(i+1)
        tempdict={dictkey:Variable()}
        checkboxdict.update(tempdict)
        #ch=ttk.Checkbutton(frame,text=dictkey,command=partial(generateplant,checkboxdict,bandchoice,classnum),variable=checkboxdict[dictkey])
        ch=ttk.Checkbutton(frame,text=dictkey,command=changecluster,variable=checkboxdict[dictkey])
        ch.grid(row=int(i/3),column=int(i%3))
        if i==minipixelareaclass:
            ch.invoke()

def generateimgplant(displaylabels):
    global currentlabels
    keys=checkboxdict.keys()
    plantchoice=[]
    for key in keys:
        plantchoice.append(checkboxdict[key].get())
    tempdisplayimg=np.zeros((displaybandarray['LabOstu'].shape))
    for i in range(len(plantchoice)):
        tup=plantchoice[i]
        if '1' in tup:
            tempdisplayimg=np.where(displaylabels==i,1,tempdisplayimg)
    currentlabels=np.copy(tempdisplayimg)
    pyplt.imsave('displayimg.png',tempdisplayimg)
    indimg=cv2.imread('displayimg.png')
    displayimg['ColorIndices']=ImageTk.PhotoImage(Image.fromarray(indimg))
    changedisplayimg(imageframe,'ColorIndices')


def kmeansclassify(choicelist,reshapedtif):
    global clusterdisplay,minipixelareaclass
    if int(kmeans.get())==0:
        return
    for i in range(len(choicelist)):
            tempband=displaybandarray[choicelist[i]]
            #tempband=cv2.resize(tempband,(450,450),interpolation=cv2.INTER_LINEAR)
            reshapedtif[:,i]=tempband.reshape(tempband.shape[0]*tempband.shape[1],1)[:,0]
    clf=KMeans(n_clusters=int(kmeans.get()),init='k-means++',n_init=10,random_state=0)
    tempdisplayimg=clf.fit_predict(reshapedtif)
    displaylabels=tempdisplayimg.reshape(displaybandarray['LabOstu'].shape)
    pixelarea=1.0
    for i in range(int(kmeans.get())):
        pixelloc=np.where(displaylabels==i)
        pixelnum=len(pixelloc[0])
        temparea=float(pixelnum/(displaylabels.shape[0]*displaylabels.shape[1]))
        if temparea<pixelarea:
            minipixelareaclass=i
            pixelarea=temparea
    if kmeans.get() not in displaylabels:
        tempdict={kmeans.get():displaylabels}
        clusterdisplay.update({''.join(choicelist):tempdict})
    return displaylabels

def changecluster():
    keys=bandchoice.keys()
    choicelist=[]
    imageband=np.zeros((displaybandarray['LabOstu'].shape))
    for key in keys:
        tup=bandchoice[key].get()
        if '1' in tup:
            choicelist.append(key)
            imageband=imageband+displaybandarray[key]
    if len(choicelist)==0:
        messagebox.showerror('No Indices is selected',message='Please select indicies to do KMeans Classification.')
        return
    if int(kmeans.get())==1:
        pyplt.imsave('displayimg.png',imageband)
        indimg=cv2.imread('displayimg.png')
        displayimg['ColorIndices']=ImageTk.PhotoImage(Image.fromarray(indimg))
        changedisplayimg(imageframe,'ColorIndices')
    else:
        if ''.join(choicelist) in clusterdisplay:
            tempdict=clusterdisplay[''.join(choicelist)]
            if kmeans.get() in tempdict:
                displaylabels=tempdict[kmeans.get()]
            else:
                reshapemodified_tif=np.zeros((displaybandarray['LabOstu'].shape[0]*displaybandarray['LabOstu'].shape[1],len(choicelist)))
                displaylabels=kmeansclassify(choicelist,reshapemodified_tif)
            generateimgplant(displaylabels)
            return
        else:
            reshapemodified_tif=np.zeros((displaybandarray['LabOstu'].shape[0]*displaybandarray['LabOstu'].shape[1],len(choicelist)))
            displaylabels=kmeansclassify(choicelist,reshapemodified_tif)
            generateimgplant(displaylabels)




    print(kmeans.get())
    print(refvar.get())
    print(edge.get())
    print(bandchoice)
    print(checkboxdict)

def showcounting(tup):
    global multi_results,pixelmmratio,kernersizes
    labels=tup[0]
    counts=tup[1]
    colortable=tup[2]
    #labeldict=tup[0]
    coinparts=tup[3]
    filename=tup[4]
    #currlabeldict=labeldict['iter'+str(int(itervar)-1)]
    #print(currlabeldict)
    #labels=currlabeldict['labels']
    #counts=currlabeldict['counts']
    #colortable=currlabeldict['colortable']
    uniquelabels=list(colortable.keys())
    originfile,extension=os.path.splitext(filename)
    image=Image.open(filename)
    image=image.resize([labels.shape[1],labels.shape[0]],resample=Image.BILINEAR)
    draw=ImageDraw.Draw(image)
    #font=ImageFont.load_default()
    if labels.shape[1]<1000:
        font=ImageFont.truetype('cmb10.ttf',size=14)
    else:
        font=ImageFont.truetype('cmb10.ttf',size=28)
    if len(coinparts)>0:
        tempband=np.zeros(labels.shape)
        coinkeys=coinparts.keys()
        for coin in coinkeys:
            coinlocs=coinparts[coin]
            tempband[coinlocs]=1
        coinarea=np.where(tempband==1)
        coinulx=min(coinarea[1])
        coinuly=min(coinarea[0])
        coinrlx=max(coinarea[1])
        coinrly=max(coinarea[0])
        coinlength=coinrly-coinuly
        coinwidth=coinrlx-coinulx
        pixelmmratio=19.05**2/(coinlength*coinwidth)
    global recborder
    for uni in uniquelabels:
        if uni !=0:
            pixelloc = np.where(labels == uni)
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
                draw.point([int(point[0]),int(point[1])],fill='yellow')
            tengentaddpoints=cal_kernelsize.tengentadd(x0,y0,x1,y1,rlx,rly,labels,uni)
            #for point in tengentaddpoints:
                #if int(point[0])>=ulx and int(point[0])<=rlx and int(point[1])>=uly and int(point[1])<=rly:
            #    draw.point([int(point[0]),int(point[1])],fill='green')
            tengentsubpoints=cal_kernelsize.tengentsub(x0,y0,x1,y1,ulx,uly,labels,uni)
            #for point in tengentsubpoints:
            #    draw.point([int(point[0]),int(point[1])],fill='green')
            width=1e10
            pointmatch=[]
            for i in range(len(tengentaddpoints)):
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
                            width=tempwidth
            if len(pointmatch)>0:
                x0=int(pointmatch[0][0])
                y0=int(pointmatch[0][1])
                x1=int(pointmatch[1][0])
                y1=int(pointmatch[1][1])
                draw.line([(x0,y0),(x1,y1)],fill='yellow')
                print('kernelwidth='+str(width*pixelmmratio))
                print('kernellength='+str(kernellength*pixelmmratio))
                #print('kernelwidth='+str(kernelwidth*pixelmmratio))
                tempdict.update({uni:[kernellength,width,kernellength*pixelmmratio,width*pixelmmratio]})


            #print(event.x, event.y, labels[event.x, event.y], ulx, uly, rlx, rly)

            #recborder = canvas.create_rectangle(ulx, uly, rlx, rly, outline='red')
            #drawcontents.append(recborder)
                draw.polygon([(ulx,uly),(rlx,uly),(rlx,rly),(ulx,rly)],outline='red')
                if uni in colortable:
                    canvastext = str(colortable[uni])
                else:
                    canvastext = 'No label'
                #rectext = canvas.create_text(midx, midy, fill='black', font='Times 8', text=canvastext)
                #drawcontents.append(rectext)
                draw.text((midx,midy),text=canvastext,font=font,fill='black')
                #trainingdataset.append([originfile+'-training'+extension,'wheat',str(ulx),str(rlx),str(uly),str(rly)])
    kernersizes.update({filename:tempdict})
    content='item count:'+str(len(uniquelabels))+'\n File: '+filename
    contentlength=len(content)+50
    #rectext=canvas.create_text(10,10,fill='black',font='Times 16',text=content,anchor=NW)
    draw.text((10,10),text=content,font=font,fill='black')
    #image.save(originfile+'-countresult'+extension,"JPEG")
    ratio=findratio([labels.shape[0],labels.shape[1]],[620,620])
    if labels.shape[0]*labels.shape[1]<620*620:
        disimage=image.resize([int(labels.shape[1]*ratio),int(labels.shape[0]*ratio)],resample=Image.BILINEAR)
    else:
        disimage=image.resize([int(labels.shape[1]/ratio),int(labels.shape[0]/ratio)],resample=Image.BILINEAR)
    displayoutput=ImageTk.PhotoImage(disimage)
    return displayoutput,image
    #displayimg['Output']=displayoutput
    #changedisplayimg(imageframe,'Output')
    #time.sleep(5)
    #image.show()



def changeoutputimg(itervar):
    outputimg=outputimgdict['iter'+str(int(itervar)-1)]
    displayimg['Output']=outputimg
    changedisplayimg(imageframe,'Output')

def export_result(iterver):
    files=multi_results.keys()
    path=filedialog.askdirectory()
    for file in files:
        labeldict=multi_results[file][0]
        itervalue='iter'+str(int(iterver.get())-1)
        print(itervalue)
        print(labeldict)
        labels=labeldict[itervalue]['labels']
        counts=labeldict[itervalue]['counts']
        colortable=labeldict[itervalue]['colortable']
        head_tail=os.path.split(file)
        originfile,extension=os.path.splitext(head_tail[1])
        if len(path)>0:
            imageband=outputimgbands[itervalue]
            originheight,originwidth=Multigraybands[file].size
            image=imageband.resize([originwidth,originheight],resample=Image.BILINEAR)
            image.save(originfile+'-countresult'+'.png',"PNG")
            originrestoredband=labels
            restoredband=originrestoredband.astype('float32')
            restoredband=cv2.resize(src=restoredband,dsize=(originwidth,originheight),interpolation=cv2.INTER_LINEAR)
            print(restoredband.shape)
            currentsizes=kernersizes[file]
            indicekeys=list(originbandarray.keys())
            indeclist=[ 0 for i in range(len(indicekeys)*3)]
            datatable={}
            origindata={}
            for key in indicekeys:
                data=originbandarray[key]
                data=data.tolist()
                tempdict={key:data}
                origindata.update(tempdict)
                print(key)
            for uni in colortable:
                print(uni,colortable[uni])
                uniloc=np.where(restoredband==float(uni))
                if len(uniloc)==0 or len(uniloc[1])==0:
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
                sizes=currentsizes[uni]
                #templist=[amount,length,width]
                templist=[amount,sizes[0],sizes[1],sizes[2],sizes[3]]
                tempdict={colortable[uni]:templist+indeclist}  #NIR,Redeyes,R,G,B,NDVI,area
                print(tempdict)
                for ki in range(len(indicekeys)):
                    originNDVI=origindata[indicekeys[ki]]
                    print(len(originNDVI),len(originNDVI[0]))
                    pixellist=[]
                    for k in range(len(uniloc[0])):
                        #print(uniloc[0][k],uniloc[1][k])
                        try:
                            tempdict[colortable[uni]][5+ki*3]+=originNDVI[uniloc[0][k]][uniloc[1][k]]
                        except IndexError:
                            print(uniloc[0][k],uniloc[1][k])
                        tempdict[colortable[uni]][6+ki*3]+=originNDVI[uniloc[0][k]][uniloc[1][k]]
                        pixellist.append(originNDVI[uniloc[0][k]][uniloc[1][k]])
                    tempdict[colortable[uni]][ki*3+5]=tempdict[colortable[uni]][ki*3+5]/amount
                    tempdict[colortable[uni]][ki*3+7]=np.std(pixellist)
                datatable.update(tempdict)
            filename=path+'/'+originfile+'-outputdata.csv'
            with open(filename,mode='w') as f:
                csvwriter=csv.writer(f)
                rowcontent=['Index','Plot','Area(#pixel)','Length(#pixel)','Width(#pixel)','Length(mm)','Width(mm)']
                for key in indicekeys:
                    rowcontent.append('avg-'+str(key))
                    rowcontent.append('sum-'+str(key))
                    rowcontent.append('std-'+str(key))
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
    messagebox.showinfo('Saved',message='Results are saved to '+path)








def extraction(frame):
    global kernersizes,multi_results,workingimg,outputimgdict,outputimgbands
    multi_results.clear()
    kernersizes.clear()
    itervar=IntVar()
    outputimgdict.clear()
    for widget in frame.winfo_children():
        widget.pack_forget()
    nonzeros=np.count_nonzero(currentlabels)
    nonzeroloc=np.where(currentlabels!=0)
    ulx,uly=min(nonzeroloc[1]),min(nonzeroloc[0])
    rlx,rly=max(nonzeroloc[1]),max(nonzeroloc[0])
    nonzeroratio=float(nonzeros)/((rlx-ulx)*(rly-uly))
    print(nonzeroratio)
    if nonzeroratio<=0.15:
        ratio=findratio([currentlabels.shape[0],currentlabels.shape[1]],[1000,1000])
        workingimg=cv2.resize(currentlabels,(int(currentlabels.shape[1]*ratio),int(currentlabels.shape[0]*ratio)),interpolation=cv2.INTER_LINEAR)
    else:
        ratio=findratio([currentlabels.shape[0],currentlabels.shape[1]],[450,450])
        workingimg=cv2.resize(currentlabels,(int(currentlabels.shape[1]/ratio),int(currentlabels.shape[0]/ratio)),interpolation=cv2.INTER_LINEAR)
    #cv2.imshow('workingimg',workingimg)
    coin=refvar.get()=='1'
    labels,border,colortable,greatareas,tinyareas,coinparts,labeldict=tkintercorestat.init(workingimg,workingimg,'',workingimg,10,coin)
    multi_results.update({currentfilename:(labeldict,coinparts)})
    iterkeys=list(labeldict.keys())
    iternum=len(iterkeys)
    print(labeldict)
    #iternum=3
    itervar.set(len(iterkeys))
    for key in labeldict:
        tup=(labeldict[key]['labels'],labeldict[key]['counts'],labeldict[key]['colortable'],coinparts,currentfilename)
        outputdisplay,outputimg=showcounting(tup)
        outputimgdict.update({key:outputdisplay})
        outputimgbands.update({key:outputimg})
    time.sleep(5)
    #tup=(labeldict,coinparts,currentfilename)
    resscaler=Scale(frame,from_=1,to=iternum,tickinterval=1,length=220,orient=HORIZONTAL,variable=itervar,command=changeoutputimg)
    resscaler.pack()
    outputbutton=Button(frame,text='Export Results',command=partial(export_result,itervar))
    outputbutton.pack()
    pass

def onFrameConfigure(inputcanvas):
    '''Reset the scroll region to encompass the inner frame'''
    inputcanvas.configure(scrollregion=inputcanvas.bbox(ALL))






def removeedge():
    global pointcontainer,displayorigin
    rgbbands=Multigraybands[currentfilename]
    height,width=rgbbands.size
    ratio=findratio([height,width],[620,620])
    resize=cv2.resize(originbandarray[currentfilename],(int(width/ratio),int(height/ratio)),interpolation=cv2.INTER_LINEAR)
    edged=cv2.Canny(resize,30,200)
    cv2.imshow('edge',edged)


## ----Interface----


## ----Display----
display_fr=Frame(root,width=640,height=640)
control_fr=Frame(root,width=320,height=320)
display_fr.pack(side=LEFT)
control_fr.pack(side=LEFT)
display_label=Text(display_fr,height=1,width=100)
display_label.tag_config("just",justify=CENTER)
display_label.insert(END,'Display Panel',"just")
display_label.configure(state=DISABLED)
display_label.pack(padx=10,pady=10)
Open_File('seedsample.JPG')
singleband('seedsample.JPG')
#cal indices
generatedisplayimg('seedsample.JPG')



imageframe=LabelFrame(display_fr)
imageframe.pack()

panelA=Label(imageframe,text='Display Panel',image=displayimg['Origin'],padx=10,pady=10) #620 x 620
panelA.pack()

buttondisplay=LabelFrame(display_fr)
buttondisplay.pack()

disbuttonoption={'Origin':'1','Gray/NIR':'2','ColorIndices':'3','Output':'4'}
for text in disbuttonoption:
    b=Radiobutton(buttondisplay,text=text,variable=displaybut_var,value=disbuttonoption[text],command=partial(changedisplayimg,imageframe,text))
    b.pack(side=LEFT,padx=20,pady=5)
    if disbuttonoption[text]=='1':
        b.invoke()
## ----Control----
control_label=Text(control_fr,height=1,width=50)
control_label.tag_config("just",justify=CENTER)
control_label.insert(END,'Control Panel',"just")
control_label.configure(state=DISABLED)
control_label.pack()
### ---open file----
openfilebutton=Button(control_fr,text='Open one/multiple images (tif,jpeg,png)',command=Open_Multifile)
openfilebutton.pack()
### ---change file---
changefileframe=LabelFrame(control_fr,text='Change Files')
changefileframe.pack()

filedropvar.set(filenames[0])
changefiledrop=OptionMenu(changefileframe,filedropvar,*filenames,command=partial(changeimage,imageframe))
changefiledrop.pack()
### ---choose color indices---
chframe=LabelFrame(control_fr,text='Select indicies below')
chframe.pack()
chcanvas=Canvas(chframe,width=200,height=100,scrollregion=(0,0,400,400))
chcanvas.pack(side=LEFT)
chscroller=Scrollbar(chframe,orient=VERTICAL)
chscroller.pack(side=RIGHT,fill=Y,expand=True)
chcanvas.config(yscrollcommand=chscroller.set)
chscroller.config(command=chcanvas.yview)
contentframe=LabelFrame(chcanvas)
chcanvas.create_window((4,4),window=contentframe,anchor=NW)
contentframe.bind("<Configure>",lambda event,arg=chcanvas:onFrameConfigure(arg))

for key in cluster:
    tempdict={key:Variable()}
    bandchoice.update(tempdict)
    ch=ttk.Checkbutton(contentframe,text=key,variable=bandchoice[key],command=changecluster)#,command=partial(autosetclassnumber,clusternumberentry,bandchoice))
    if filedropvar.get()=='seedsample.JPG':
        if key=='NDI':
            ch.invoke()
    ch.pack(fill=X)

### ----Class NUM----
kmeanslabel=LabelFrame(control_fr,text='Select # of class')
checkboxframe=LabelFrame(control_fr,text='Select classes')
kmeanslabel.pack()

kmeans.set(2)
kmeansbar=Scale(kmeanslabel,from_=1,to=10,tickinterval=1,length=220,orient=HORIZONTAL,variable=kmeans,command=partial(generatecheckbox,checkboxframe))
kmeansbar.pack()
checkboxframe.pack()
generatecheckbox(checkboxframe,2)

### --- ref and edge settings ---
refframe=LabelFrame(control_fr,text='Reference Setting')
refframe.pack()

refoption=[('Coin as Ref','1'),('No Ref','0')]
refvar.set('1')
for text,mode in refoption:
    b=Radiobutton(refframe,text=text,variable=refvar,value=mode)
    b.pack(side=LEFT,padx=15)
#edgeframe=LabelFrame(control_fr,text='Edge remove setting')
#edgeframe.pack()
#edgeoption=[('Remove edge','1'),('Keep same','0')]

#edge.set('0')
#for text,mode in edgeoption:
#    b=Radiobutton(edgeframe,text=text,variable=edge,value=mode)#,command=removeedge)
#    b.pack(side=LEFT,padx=6)

### ---start extraction---
extractionframe=LabelFrame(control_fr,text='Image extraction')
extractionframe.pack(padx=5,pady=5)
resviewframe=LabelFrame(control_fr)
extractbutton=Button(extractionframe,text='Start Image Process',command=partial(extraction,resviewframe))
extractbutton.pack()
resviewframe.pack()
root.mainloop()

