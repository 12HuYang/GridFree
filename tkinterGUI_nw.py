from tkinter import *
from tkinter import ttk
import tkinter.filedialog as filedialog
from tkinter import messagebox

from PIL import Image,ImageDraw,ImageFont
from PIL import ImageTk,ImageGrab
import cv2
from skimage import filters
#import rasterio
import matplotlib.pyplot as pyplt
from matplotlib.figure import Figure

import numpy as np
import os
import time
import csv

from functools import partial
import sys

import kplus
from sklearn.cluster import KMeans
import tkintercorestat
import tkintercorestat_plot
import tkintercore
import cal_kernelsize
import histograms
import createBins
import axistest

class img():
    def __init__(self,size,bands):
        self.size=size
        self.bands=bands

displayimg={'Origin':None,
            'Color Deviation':None,
            'ColorIndices':None,
            'Output':None}
#cluster=['LabOstu','NDI'] #,'Greenness','VEG','CIVE','MExG','NDVI','NGRDI','HEIGHT']
cluster=['LabOstu','NDI','Greenness','VEG','CIVE','MExG','NDVI','NGRDI','HEIGHT','Band1','Band2','Band3']
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

coinsize=StringVar()
refvar=StringVar()
imgtypevar=StringVar()
edge=StringVar()
kmeans=IntVar()
filedropvar=StringVar()
displaybut_var=StringVar()
bandchoice={}
checkboxdict={}

minipixelareaclass=0

coinbox=None

currentfilename='seedsample.JPG'
currentlabels=None
workingimg=None

boundaryarea=None
outputbutton=None
font=None
reseglabels=None
coindict=None
## Funcitons
refarea=None
originlabels=None
originlabeldict=None
changekmeans=False
convband=None
reflabel=0
minflash=[]
dotflash=[]
labelplotmap={}
mappath=''
elesize=[]
labellist=[]
figdotlist={}


maxx=0
minx=0
bins=None
loccanvas=None
linelocs=[0,0,0,0]
maxy=0
miny=0

def distance(p1,p2):
    return np.sum((p1-p2)**2)




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
    #widget.configure(image=displayimg[text])
    #widget.image=displayimg[text]
    #widget.pack()
    w=displayimg[text]['Size'][1]
    l=displayimg[text]['Size'][0]
    widget.config(width=w,height=l)
    widget.create_image(0,0,image=displayimg[text]['Image'],anchor=NW)
    widget.pack()
    #print('change to '+text)
    #time.sleep(1)

def generatedisplayimg(filename):
    firstimg=Multiimagebands[filename]
    #height,width=firstimg.size
    height,width=displaybandarray[filename]['LabOstu'].shape
    ratio=findratio([height,width],[850,850])
    resizeshape=[]
    if height*width<850*850:
        #resize=cv2.resize(Multiimage[filename],(int(width*ratio),int(height*ratio)),interpolation=cv2.INTER_LINEAR)
        resizeshape.append(width*ratio)
        resizeshape.append(height*ratio)
    else:
        #resize=cv2.resize(Multiimage[filename],(int(width/ratio),int(height/ratio)),interpolation=cv2.INTER_LINEAR)
        resizeshape.append(width/ratio)
        resizeshape.append(height/ratio)
    resize=cv2.resize(Multiimage[filename],(int(resizeshape[0]),int(resizeshape[1])),interpolation=cv2.INTER_LINEAR)
    rgbimg=ImageTk.PhotoImage(Image.fromarray(resize.astype('uint8')))
    tempdict={}
    tempdict.update({'Size':resize.shape})
    tempdict.update({'Image':rgbimg})
    displayimg['Origin']=tempdict
    #if height*width<850*850:
    #    resize=cv2.resize(Multigray[filename],(int(width*ratio),int(height*ratio)),interpolation=cv2.INTER_LINEAR)
    #else:
        #resize=cv2.resize(Multigray[filename],(int(width/ratio),int(height/ratio)),interpolation=cv2.INTER_LINEAR)
    tempdict={}
    tempdict.update({'Size':resize.shape})
    #if height*width<850*850:
    #    tempdict.update({'Image':ImageTk.PhotoImage(Image.fromarray(np.zeros((int(height*ratio),int(width*ratio))).astype('uint8')))})
    #else:
    #    tempdict.update({'Image':ImageTk.PhotoImage(Image.fromarray(np.zeros((int(height/ratio),int(width/ratio))).astype('uint8')))})
    tempdict.update({'Image':ImageTk.PhotoImage(Image.fromarray(np.zeros((int(resizeshape[1]),int(resizeshape[0]))).astype('uint8')))})
    displayimg['Output']=tempdict
    tempband=np.zeros((displaybandarray[filename]['LabOstu'].shape))
    tempband=tempband+displaybandarray[filename]['LabOstu']
    ratio=findratio([tempband.shape[0],tempband.shape[1]],[850,850])
    #if tempband.shape[0]*tempband.shape[1]<850*850:
    #    tempband=cv2.resize(ratio,(int(tempband.shape[1]*ratio),int(tempband.shape[0]*ratio)),interpolation=cv2.INTER_LINEAR)
    #else:
    #    tempband=cv2.resize(ratio,(int(tempband.shape[1]/ratio),int(tempband.shape[0]/ratio)),interpolation=cv2.INTER_LINEAR)
    tempband=cv2.resize(tempband,(int(resizeshape[0]),int(resizeshape[1])),interpolation=cv2.INTER_LINEAR)
    print('resizeshape',resizeshape)
    pyplt.imsave('displayimg.png',tempband)
    indimg=cv2.imread('displayimg.png')
    tempdict={}
    tempdict.update({'Size':tempband.shape})
    tempdict.update({'Image':ImageTk.PhotoImage(Image.fromarray(indimg))})
    displayimg['ColorIndices']=tempdict

    #resize=cv2.resize(Multigray[filename],(int(resizeshape[0]),int(resizeshape[1])),interpolation=cv2.INTER_LINEAR)
    #grayimg=ImageTk.PhotoImage(Image.fromarray(resize.astype('uint8')))
    #tempdict={}
    #tempdict.update({'Size':resize.shape})
    #tempdict.update({'Image':grayimg})
    displayimg['Color Deviation']=tempdict



def Open_File(filename):   #add to multi-image,multi-gray  #call band calculation
    global Multiimage,Multigray,Multitype,Multiimagebands,Multigraybands,filenames
    try:
        Filersc=cv2.imread(filename,flags=cv2.IMREAD_ANYCOLOR)
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
        messagebox.showerror('Invalid Image Format','Cannot open '+filename)
        return False
    filenames.append(filename)
    return True

def commentoutrasterio():
    '''
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
    '''
    pass

def Open_Map():
    global mappath,elesize,labellist
    filepath=filedialog.askopenfilename()
    if len(filepath)>0:
        if 'csv' in filepath:
            mappath=filepath
            elesize=[]
            labellist=[]
            rows=[]
            print('open map at: '+mappath)
            with open(mappath,mode='r',encoding='utf-8-sig') as f:
                csvreader=csv.reader(f)
                for row in csvreader:
                    rows.append(row)
                    temprow=[]
                    for ele in row:
                        if ele is not '':
                            temprow.append(ele)
                    elesize.append(len(temprow))
            for i in range(len(rows)):
                for j in range(len(rows[i])):
                    if rows[i][j]!='':
                        labellist.append(rows[i][j])
        else:
            messagebox.showerror('Invalide File',message='Please open csv formate file as map file.')
        corlortable=tkintercorestat.get_colortable(reseglabels)
        tup=(reseglabels,[],corlortable,{},currentfilename)
        print(elesize)
        mapdict,mapimage=showcounting(tup)
        tempimgbands={}
        tempimgdict={}
        tempimgbands.update({'iter0':mapimage})
        tempimgdict.update({'iter0':mapdict})
        outputimgdict.update({currentfilename:tempimgdict})
        outputimgbands.update({currentfilename:tempimgbands})
        changeoutputimg(currentfilename,'1')

def Open_Multifile():
    global Multiimage,Multigray,Multitype,Multiimagebands,changefileframe,imageframe,Multigraybands,filenames
    global changefiledrop,filedropvar,originbandarray,displaybandarray,clusterdisplay,currentfilename,resviewframe
    global refsubframe,outputbutton,reseglabels,refbutton,figcanvas,loccanvas,originlabels,changekmeans,refarea
    global originlabeldict,convband,panelA

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
        reseglabels=None
        originlabels=None
        originlabeldict=None
        #changekmeans=True
        convband=None
        refvar.set('0')
        kmeans.set('2')
        panelA.delete(ALL)
        panelA.unbind('<Button-1>')
        refarea=None
        if 'NDI' in bandchoice:
            bandchoice['NDI'].set('1')
        if 'NDVI' in bandchoice:
            bandchoice['NDVI'].set('1')
        refbutton.config(state=DISABLED)
        figcanvas.delete(ALL)
        #loccanvas=None
        for widget in refsubframe.winfo_children():
            widget.config(state=DISABLED)
        #for widget in resviewframe.winfo_children():
        #    widget.config(state=DISABLED)
        if outputbutton is not None:
            outputbutton.config(state=DISABLED)
        for i in range(len(MULTIFILES)):
            if Open_File(MULTIFILES[i])==False:
                return
            singleband(MULTIFILES[i])
        for widget in changefileframe.winfo_children():
            widget.pack_forget()
        filedropvar.set(filenames[0])
        changefiledrop=OptionMenu(changefileframe,filedropvar,*filenames,command=partial(changeimage,imageframe))
        changefiledrop.pack()
        #singleband(filenames[0])
        generatedisplayimg(filenames[0])
        currentfilename=filenames[0]
        for i in range(len(cluster)):
            bandchoice[cluster[i]].set('')
        #changedisplayimg(imageframe,'Origin')
        kmeans.set(2)
        reshapemodified_tif=np.zeros((displaybandarray[currentfilename]['LabOstu'].shape[0]*displaybandarray[currentfilename]['LabOstu'].shape[1],1))
        colordicesband=kmeansclassify(['LabOstu'],reshapemodified_tif)
        generateimgplant(colordicesband)
        changedisplayimg(imageframe,'Origin')
        bandchoice['LabOstu'].set('1')




def workbandsize(item):
    pass

def singleband(file):
    global displaybandarray,originbandarray
    try:
        bands=Multigraybands[file].bands
    except:
        return
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
    '''
    if imgtypevar.get()=='0':
        if bandsize[0]*bandsize[1]>2000*2000:
            ratio=findratio([bandsize[0],bandsize[1]],[2000,2000])
        else:
            ratio=1
    if imgtypevar.get()=='1':
        if bandsize[0]*bandsize[1]>1000*1000:
            ratio=findratio([bandsize[0],bandsize[1]],[600,600])
        else:
            #ratio=findratio([bandsize[0],bandsize[1]],[500,500])
            #ratio=float(1/ratio)
            ratio=1
    '''
    if bandsize[0]*bandsize[1]>2000*2000:
        ratio=findratio([bandsize[0],bandsize[1]],[2000,2000])
    else:
        ratio=1
    #if bandsize[0]*bandsize[1]>850*850:
    #    ratio=findratio([bandsize[0],bandsize[1]],[850,850])
    #else:
    #    ratio=1
    #ttestbands=np.copy(bands)
    #testdisplaybands=cv2.resize(ttestbands,(int(bandsize[1]/ratio),int(bandsize[0]/ratio)),interpolation=cv2.INTER_LINEAR)
    #testdisplaybands=cv2.resize(testdisplaybands,(int(resizeshape[0]),int(resizeshape[1])),interpolation=cv2.INTER_LINEAR)
    #print('testdisplaybands size',testdisplaybands.size)
    #if bandsize[0]*bandsize[1]>850*850:
    #    ratio=findratio([bandsize[0],bandsize[1]],[850,850])
    #else:
    #    ratio=1
    originbands={}
    displays={}
    if 'LabOstu' not in originbands:
        originbands.update({'LabOstu':bands})
        displaybands=cv2.resize(bands,(int(bandsize[1]/ratio),int(bandsize[0]/ratio)),interpolation=cv2.INTER_LINEAR)
        #displaybands=displaybands.reshape((int(bandsize[1]/ratio),int(bandsize[0]/ratio),3))
        #kernel=np.ones((2,2),np.float32)/4
        #displaybands=np.copy(bands)
        displays.update({'LabOstu':displaybands})
        #displaybandarray.update({'LabOstu':cv2.filter2D(displaybands,-1,kernel)})
    bands=Multiimagebands[file].bands
    for i in range(3):
        bands[i,:,:]=cv2.GaussianBlur(bands[i,:,:],(3,3),cv2.BORDER_DEFAULT)
    NDI=128*((bands[1,:,:]-bands[0,:,:])/(bands[1,:,:]+bands[0,:,:])+1)
    tempdict={'NDI':NDI}
    if 'NDI' not in originbands:
        originbands.update(tempdict)
        displaybands=cv2.resize(NDI,(int(bandsize[1]/ratio),int(bandsize[0]/ratio)),interpolation=cv2.INTER_LINEAR)
        #displaybands=np.copy(NDI)
        #kernel=np.ones((2,2),np.float32)/4
        #displaydict={'NDI':cv2.filter2D(displaybands,-1,kernel)}
        displaydict={'NDI':displaybands}
        #displaydict=displaydict.reshape((int(bandsize[1]/ratio),int(bandsize[0]/ratio),3))
        displays.update(displaydict)

    Red=bands[0,:,:]
    Green=bands[1,:,:]
    Blue=bands[2,:,:]
    tempdict={'Band1':Red}
    if 'Band1' not in originbands:
        originbands.update(tempdict)
        image=cv2.resize(Red,(int(bandsize[1]/ratio),int(bandsize[0]/ratio)),interpolation=cv2.INTER_LINEAR)
        displaydict={'Band1':image}
        displays.update(displaydict)
    tempdict={'Band2':Green}
    if 'Band2' not in originbands:
        originbands.update(tempdict)
        image=cv2.resize(Red,(int(bandsize[1]/ratio),int(bandsize[0]/ratio)),interpolation=cv2.INTER_LINEAR)
        displaydict={'Band2':image}
        displays.update(displaydict)
    tempdict={'Band3':Blue}
    if 'Band3' not in originbands:
        originbands.update(tempdict)
        image=cv2.resize(Red,(int(bandsize[1]/ratio),int(bandsize[0]/ratio)),interpolation=cv2.INTER_LINEAR)
        displaydict={'Band3':image}
        displays.update(displaydict)
    Greenness = bands[1, :, :] / (bands[0, :, :] + bands[1, :, :] + bands[2, :, :])
    tempdict = {'Greenness': Greenness}
    if 'Greenness' not in originbandarray:
        originbands.update(tempdict)
        image=cv2.resize(Greenness,(int(bandsize[1]/ratio),int(bandsize[0]/ratio)),interpolation=cv2.INTER_LINEAR)
        #image=image.reshape((int(bandsize[1]/ratio),int(bandsize[0]/ratio),3))
        displaydict={'Greenness':image}
        #displaybandarray.update(worktempdict)
        displays.update(displaydict)
    VEG=bands[1,:,:]/(np.power(bands[0,:,:],0.667)*np.power(bands[2,:,:],(1-0.667)))
    tempdict={'VEG':VEG}
    if 'VEG' not in originbandarray:
        originbands.update(tempdict)
        image=cv2.resize(VEG,(int(bandsize[1]/ratio),int(bandsize[0]/ratio)),interpolation=cv2.INTER_LINEAR)
        kernel=np.ones((4,4),np.float32)/16
        #displaybandarray.update({'LabOstu':})
        #image=image.reshape((int(bandsize[1]/ratio),int(bandsize[0]/ratio),3))
        worktempdict={'VEG':cv2.filter2D(image,-1,kernel)}
        displays.update(worktempdict)
    CIVE=0.441*bands[0,:,:]-0.811*bands[1,:,:]+0.385*bands[2,:,:]+18.78745
    tempdict={'CIVE':CIVE}
    if 'CIVE' not in originbandarray:
        originbands.update(tempdict)
        image=cv2.resize(CIVE,(int(bandsize[1]/ratio),int(bandsize[0]/ratio)),interpolation=cv2.INTER_LINEAR)
        #image=image.reshape((int(bandsize[1]/ratio),int(bandsize[0]/ratio),3))
        worktempdict={'CIVE':image}
        displays.update(worktempdict)
    MExG=1.262*bands[1,:,:]-0.884*bands[0,:,:]-0.311*bands[2,:,:]
    tempdict={'MExG':MExG}
    if 'MExG' not in originbandarray:
        originbands.update(tempdict)
        image=cv2.resize(MExG,(int(bandsize[1]/ratio),int(bandsize[0]/ratio)),interpolation=cv2.INTER_LINEAR)
        #image=image.reshape((int(bandsize[1]/ratio),int(bandsize[0]/ratio),3))
        worktempdict={'MExG':image}
        displays.update(worktempdict)
    NDVI=(bands[0,:,:]-bands[2,:,:])/(bands[0,:,:]+bands[2,:,:])
    tempdict={'NDVI':NDVI}
    if 'NDVI' not in originbandarray:
        originbands.update(tempdict)
        image=cv2.resize(NDVI,(int(bandsize[1]/ratio),int(bandsize[0]/ratio)),interpolation=cv2.INTER_LINEAR)
        #image=image.reshape((int(bandsize[1]/ratio),int(bandsize[0]/ratio),3))
        worktempdict={'NDVI':image}
        displays.update(worktempdict)
    NGRDI=(bands[1,:,:]-bands[0,:,:])/(bands[1,:,:]+bands[0,:,:])
    tempdict={'NGRDI':NGRDI}
    if 'NGRDI' not in originbandarray:
        originbands.update(tempdict)
        image=cv2.resize(NGRDI,(int(bandsize[1]/ratio),int(bandsize[0]/ratio)),interpolation=cv2.INTER_LINEAR)
        #image=image.reshape((int(bandsize[1]/ratio),int(bandsize[0]/ratio),3))
        worktempdict={'NGRDI':image}
        displays.update(worktempdict)
    if channel>=1:
        nirbands=Multigraybands[file].bands
        NDVI=(nirbands[0,:,:]-bands[1,:,:])/(nirbands[0,:,:]+bands[1,:,:])
        tempdict={'NDVI':NDVI}
        #if 'NDVI' not in originbandarray:
        originbands.update(tempdict)
        image=cv2.resize(NDVI,(int(bandsize[1]/ratio),int(bandsize[0]/ratio)),interpolation=cv2.INTER_LINEAR)
        #image=image.reshape((int(bandsize[1]/ratio),int(bandsize[0]/ratio),3))
        worktempdict={'NDVI':image}
        displays.update(worktempdict)
    if channel==3:
        bands=Multigraybands[file].bands
        Height=bands[2,:,:]
        tempdict={'HEIGHT':Height}
        if 'HEIGHT' not in originbandarray:
            originbands.update(tempdict)
            image=cv2.resize(Height,(int(bandsize[1]/ratio),int(bandsize[0]/ratio)),interpolation=cv2.INTER_LINEAR)
            worktempdict={'HEIGHT':image}
            displays.update(worktempdict)
    else:
        originbandarray.update({'HEIGHT':np.zeros(bandsize)})
        image=np.zeros((int(bandsize[0]/ratio),int(bandsize[0]/ratio)))
        worktempdict.update({'HEIGHT':image})
        displays.update(worktempdict)


    displaybandarray.update({file:displays})
    originbandarray.update({file:originbands})







def Band_calculation():
    global originbandarray,workbandarray
    originbandarray={}
    workbandarray={}
    for file in filenames:
        singleband(file)




def changeimage(frame,filename):
    global clusterdisplay,currentfilename,resviewframe
    clusterdisplay={}
    currentfilename=filename
    print(filename)
    generatedisplayimg(filename)
    changedisplayimg(frame,'Origin')
    for key in cluster:
        tuplist=[]
        for i in range(len(cluster)):
            tuplist.append('')
        tup=tuple(tuplist)
        bandchoice[key].set(tup)
    #for key in cluster:
    #    ch=ttk.Checkbutton(contentframe,text=key,variable=bandchoice[key],command=changecluster)#,command=partial(autosetclassnumber,clusternumberentry,bandchoice))
    #    ch.pack()

    if filename in multi_results.keys():
        for widget in resviewframe.winfo_children():
            widget.pack_forget()
        iternum=len(list(multi_results[filename][0].keys()))
        itervar=IntVar()
        itervar.set(iternum)
        resscaler=Scale(resviewframe,from_=1,to=iternum,tickinterval=1,length=220,orient=HORIZONTAL,variable=itervar,command=partial(changeoutputimg,filename))
        resscaler.pack()
        outputbutton=Button(resviewframe,text='Export Results',command=partial(export_result,itervar))
        outputbutton.pack()


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
        ratio=findratio([imageband.shape[0],imageband.shape[1]],[850,850])
        imageband=cv2.resize(imageband,(int(imageband.shape[1]/ratio),int(imageband.shape[0]/ratio)),interpolation=cv2.INTER_LINEAR)
        imageband=np.where(imageband==1,2,imageband)
        temprgb=np.zeros((imageband.shape[0],imageband.shape[1],3))
        pyplt.imsave('displayimg.png',imageband)
        indimg=cv2.imread('displayimg.png')
        displayimg['ColorIndices']['Image']=ImageTk.PhotoImage(Image.fromarray(indimg))
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
            pyplt.imsave('allcolorindex.png',displaylabels)
            return
        else:
            reshapemodified_tif=np.zeros((displaybandarray['LabOstu'].shape[0]*displaybandarray['LabOstu'].shape[1],len(choicelist)))
            displaylabels=kmeansclassify(choicelist,reshapemodified_tif)
            generateimgplant(displaylabels)
            pyplt.imsave('allcolorindex.png',displaylabels)




def generatecheckbox(frame,classnum):
    global checkboxdict
    for widget in frame.winfo_children():
        widget.pack_forget()
    checkboxdict={}
    for i in range(10):
        dictkey=str(i+1)
        tempdict={dictkey:Variable()}
        tempdict[dictkey].set('0')
        checkboxdict.update(tempdict)
        ch=Checkbutton(checkboxframe,text=dictkey,variable=checkboxdict[dictkey],command=partial(changecluster,''))
        if i+1>int(kmeans.get()):
            ch.config(state=DISABLED)
        ch.pack(side=LEFT)
        #if i==0:
        #    ch.invoke()
    #for i in range(int(classnum)):
    #    dictkey='class '+str(i+1)
    #    tempdict={dictkey:Variable()}
    #    checkboxdict.update(tempdict)
        #ch=ttk.Checkbutton(frame,text=dictkey,command=partial(generateplant,checkboxdict,bandchoice,classnum),variable=checkboxdict[dictkey])
    #    ch=ttk.Checkbutton(frame,text=dictkey,command=changecluster,variable=checkboxdict[dictkey])
    #    ch.grid(row=int(i/3),column=int(i%3))
    #    if i==minipixelareaclass:
    #        ch.invoke()

def generateimgplant(displaylabels):
    global currentlabels,changekmeans
    keys=checkboxdict.keys()
    plantchoice=[]
    for key in keys:
        plantchoice.append(checkboxdict[key].get())
    tempdisplayimg=np.zeros((displaybandarray[currentfilename]['LabOstu'].shape))
    colordivimg=np.zeros((displaybandarray[currentfilename]['LabOstu'].shape))
    for i in range(len(plantchoice)):
        tup=plantchoice[i]
        if '1' in tup:
            tempdisplayimg=np.where(displaylabels==i,1,tempdisplayimg)
    uniquecolor=np.unique(tempdisplayimg)
    if len(uniquecolor)==1 and uniquecolor[0]==1:
        tempdisplayimg=np.copy(displaylabels).astype('float32')
    currentlabels=np.copy(tempdisplayimg)
    tempcolorimg=np.copy(displaylabels).astype('float32')
    ratio=findratio([tempdisplayimg.shape[0],tempdisplayimg.shape[1]],[850,850])
    if tempdisplayimg.shape[0]*tempdisplayimg.shape[1]<850*850:
        tempdisplayimg=cv2.resize(tempdisplayimg,(int(tempdisplayimg.shape[1]*ratio),int(tempdisplayimg.shape[0]*ratio)))
        colordivimg=cv2.resize(tempcolorimg,(int(colordivimg.shape[1]*ratio),int(colordivimg.shape[0]*ratio)))
    else:
        tempdisplayimg=cv2.resize(tempdisplayimg,(int(tempdisplayimg.shape[1]/ratio),int(tempdisplayimg.shape[0]/ratio)))
        colordivimg=cv2.resize(tempcolorimg,(int(colordivimg.shape[1]/ratio),int(colordivimg.shape[0]/ratio)))
    pyplt.imsave('displayimg.png',tempdisplayimg)
    pyplt.imsave('allcolorindex.png',colordivimg)
    #bands=Image.fromarray(tempdisplayimg)
    #bands=bands.convert('L')
    #bands.save('displayimg.png')
    indimg=cv2.imread('displayimg.png')
    tempdict={}
    tempdict.update({'Size':tempdisplayimg.shape})
    tempdict.update({'Image':ImageTk.PhotoImage(Image.fromarray(indimg))})
    displayimg['ColorIndices']=tempdict

    #indimg=cv2.imread('allcolorindex.png')
    #tempdict.update({'Image':ImageTk.PhotoImage(Image.fromarray(indimg))})
    #
    colorimg=cv2.imread('allcolorindex.png')
    colordivdict={}
    colordivdict.update({'Size':tempdisplayimg.shape})
    colordivdict.update({'Image':ImageTk.PhotoImage(Image.fromarray(colorimg))})
    displayimg['Color Deviation']=colordivdict

    changedisplayimg(imageframe,'ColorIndices')
    changekmeans=True


def kmeansclassify(choicelist,reshapedtif):
    global clusterdisplay,minipixelareaclass
    if int(kmeans.get())==0:
        return
    for i in range(len(choicelist)):
            tempband=displaybandarray[currentfilename][choicelist[i]]
            #tempband=cv2.resize(tempband,(450,450),interpolation=cv2.INTER_LINEAR)
            reshapedtif[:,i]=tempband.reshape(tempband.shape[0]*tempband.shape[1],1)[:,0]
    clf=KMeans(n_clusters=int(kmeans.get()),init='k-means++',n_init=10,random_state=0)
    tempdisplayimg=clf.fit_predict(reshapedtif)
    displaylabels=tempdisplayimg.reshape(displaybandarray[currentfilename]['LabOstu'].shape)
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

def changecluster(event):
    #global kmeanscanvas
    keys=bandchoice.keys()
    choicelist=[]
    imageband=np.zeros((displaybandarray[currentfilename]['LabOstu'].shape))
    for key in keys:
        tup=bandchoice[key].get()
        if '1' in tup:
            choicelist.append(key)
            imageband=imageband+displaybandarray[currentfilename][key]
    colornum=int(kmeans.get())
    colorstrip=np.zeros((10,35*colornum),'float32')
    for i in range(colornum):
        for j in range(0,35):
            colorstrip[:,i*35+j]=i+1
    pyplt.imsave('colorstrip.png',colorstrip)
    #kmeanscanvas.delete(ALL)
    #colorimg=cv2.imread('colorstrip.png')
    #colorimg=ImageTk.PhotoImage(Image.fromarray(colorimg))
    #kmeanscanvas.create_image(0,0,image=colorimg,anchor=NW)
    #kmeanscanvas.pack()
    if len(choicelist)==0:
        messagebox.showerror('No Indices is selected',message='Please select indicies to do KMeans Classification.')
        tempband=np.copy(displaybandarray[currentfilename]['LabOstu'])
        ratio=findratio([tempband.shape[0],tempband.shape[1]],[850,850])
        tempband=cv2.resize(tempband,(int(tempband.shape[1]/ratio),int(tempband.shape[0]/ratio)),interpolation=cv2.INTER_LINEAR)
        pyplt.imsave('displayimg.png',tempband)
        indimg=cv2.imread('displayimg.png')
        #indimg=Image.open('displayimg.png')
        tempdict={}
        tempdict.update({'Size':tempband.shape})
        tempdict.update({'Image':ImageTk.PhotoImage(Image.fromarray(indimg))})
        displayimg['ColorIndices']=tempdict
        changedisplayimg(imageframe,'ColorIndices')
        return
    if int(kmeans.get())==1:
        tempband=np.copy(imageband)
        ratio=findratio([tempband.shape[0],tempband.shape[1]],[850,850])
        tempband=cv2.resize(tempband,(int(tempband.shape[1]/ratio),int(tempband.shape[0]/ratio)),interpolation=cv2.INTER_LINEAR)
        pyplt.imsave('displayimg.png',tempband)
        #bands=Image.fromarray(tempband)
        #bands=bands.convert('RGB')
        #bands.save('displayimg.png')
        indimg=cv2.imread('displayimg.png')
        tempdict={}
        tempdict.update({'Size':tempband.shape})
        tempdict.update({'Image':ImageTk.PhotoImage(Image.fromarray(indimg))})
        displayimg['ColorIndices']=tempdict
        changedisplayimg(imageframe,'ColorIndices')
    else:
        if ''.join(choicelist) in clusterdisplay:
            tempdict=clusterdisplay[''.join(choicelist)]
            if kmeans.get() in tempdict:
                displaylabels=tempdict[kmeans.get()]
            else:
                reshapemodified_tif=np.zeros((displaybandarray[currentfilename]['LabOstu'].shape[0]*displaybandarray[currentfilename]['LabOstu'].shape[1],len(choicelist)))
                displaylabels=kmeansclassify(choicelist,reshapemodified_tif)
            generateimgplant(displaylabels)
            pyplt.imsave('allcolorindex.png',displaylabels)
            #kmeanscanvas.update()
            return
        else:
            reshapemodified_tif=np.zeros((displaybandarray[currentfilename]['LabOstu'].shape[0]*displaybandarray[currentfilename]['LabOstu'].shape[1],len(choicelist)))
            displaylabels=kmeansclassify(choicelist,reshapemodified_tif)
            generateimgplant(displaylabels)
            pyplt.imsave('allcolorindex.png',displaylabels)
        #changedisplayimg(imageframe,'Color Deviation')




    print(kmeans.get())
    print(refvar.get())
    print(edge.get())
    print(bandchoice)
    print(checkboxdict)

def showinitcounting(tup):
    global multi_results,kernersizes
    labels=tup[0]
    counts=tup[1]
    colortable=tup[2]
    coinparts=tup[3]
    filename=tup[4]

def showcounting(tup,number=True):
    global multi_results,kernersizes#,pixelmmratio,kernersizes
    global font
    labels=tup[0]
    counts=tup[1]
    if len(mappath)>0:
        colortable=tkintercorestat.get_mapcolortable(labels,elesize.copy(),labellist.copy())
    else:
        colortable=tup[2]
    if type(refarea)!=type(None):
        colortable.update({65535:'Ref'})
        labels[refarea]=65535
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
    sizeuniq,sizecounts=np.unique(labels,return_counts=True)
    minsize=min(sizecounts)
    suggsize=int(minsize**0.5)
    if suggsize>22:
        suggsize=22
    if suggsize<14:
        suggsize=14
    font=ImageFont.truetype('cmb10.ttf',size=suggsize)
    #if labels.shape[1]<850:
    #    font=ImageFont.truetype('cmb10.ttf',size=16)
    #else:
    #    font=ImageFont.truetype('cmb10.ttf',size=22)
    if len(coinparts)>0:
        tempband=np.zeros(labels.shape)
        coinkeys=coinparts.keys()
        for coin in coinkeys:
            coinlocs=coinparts[coin]
            tempband[coinlocs]=1

    global recborder
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
            draw.polygon([(ulx,uly),(rlx,uly),(rlx,rly),(ulx,rly)],outline='red')
            if number==True:
                if uni in colortable:
                    canvastext = str(colortable[uni])
                else:
                    canvastext = 'No label'
                if imgtypevar.get()=='0':
                    draw.text((midx-1, midy+1), text=canvastext, font=font, fill='white')
                    draw.text((midx+1, midy+1), text=canvastext, font=font, fill='white')
                    draw.text((midx-1, midy-1), text=canvastext, font=font, fill='white')
                    draw.text((midx+1, midy-1), text=canvastext, font=font, fill='white')
                    #draw.text((midx,midy),text=canvastext,font=font,fill=(141,2,31,0))
                    draw.text((midx,midy),text=canvastext,font=font,fill='black')



    if refarea is not None:
        content='item count:'+str(len(uniquelabels)-1)+'\n File: '+filename
    else:
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
    height,width=displaybandarray[filename]['LabOstu'].shape
    ratio=findratio([height,width],[850,850])
    #if labels.shape[0]*labels.shape[1]<850*850:
    #    disimage=image.resize([int(labels.shape[1]*ratio),int(labels.shape[0]*ratio)],resample=Image.BILINEAR)
    #else:
    #    disimage=image.resize([int(labels.shape[1]/ratio),int(labels.shape[0]/ratio)],resample=Image.BILINEAR)
    if height*width<850*850:
        disimage=image.resize([int(width*ratio),int(height*ratio)],resample=Image.BILINEAR)
    else:
        disimage=image.resize([int(width/ratio),int(height/ratio)],resample=Image.BILINEAR)
    displayoutput=ImageTk.PhotoImage(disimage)
    return displayoutput,image
    #displayimg['Output']=displayoutput
    #changedisplayimg(imageframe,'Output')
    #time.sleep(5)
    #image.show()



def changeoutputimg(file,intnum):
    outputimg=outputimgdict[file]['iter'+str(int(intnum)-1)]
    tempdict={}
    tempdict.update({'Size':displayimg['ColorIndices']['Size']})
    tempdict.update({'Image':outputimg})
    displayimg['Output']=tempdict
    changedisplayimg(imageframe,'Output')

def export_result(iterver):
    files=multi_results.keys()
    path=filedialog.askdirectory()
    for file in files:
        labeldict=multi_results[file][0]
        totalitervalue=len(list(labeldict.keys()))
        #itervalue='iter'+str(int(iterver.get())-1)
        #itervalue='iter'+str(totalitervalue-1)
        #itervalue=int(iterver.get())
        itervalue='iter'+iterver
        print(itervalue)
        print(labeldict)
        labels=labeldict[itervalue]['labels']
        counts=labeldict[itervalue]['counts']
        if len(mappath)>0:
            colortable=tkintercorestat.get_mapcolortable(labels,elesize.copy(),labellist.copy())
        else:
            colortable=labeldict[itervalue]['colortable']
        #originheight,originwidth=Multigraybands[file].size
        #copylabels=np.copy(labels)
        #copylabels[refarea]=65535
        #labels=cv2.resize(copylabels.astype('float32'),dsize=(originwidth,originheight),interpolation=cv2.INTER_LINEAR)
        head_tail=os.path.split(file)
        originfile,extension=os.path.splitext(head_tail[1])
        if len(path)>0:
            imageband=outputimgbands[file][itervalue]
            draw=ImageDraw.Draw(imageband)
            uniquelabels=list(colortable.keys())
            tempdict={}
            if coinsize.get()=='1':
                if refarea is not None:
                    #realref=np.where(labels==65535.0)
                    pixelmmratio=((19.05/2)**2*3.14/len(refarea[0]))**0.5
                    #pixelmmratio=((19.05/2)**2*3.14/len(realref[0]))**0.5
                else:
                    pixelmmratio=1.0
            else:
                if coinsize.get()=='3':
                    specarea=float(sizeentry.get())
                    if refarea is not None:
                        #realref=np.where(labels==65535.0)
                        pixelmmratio=(specarea/len(refarea[0]))**0.5
                        #pixelmmratio=(specarea/len(realref[0]))**0.5
                    else:
                        pixelmmratio=1.0
                else:
                    pixelmmratio=1.0
            print('coinsize',coinsize.get(),'pixelmmratio',pixelmmratio)
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
                        if imgtypevar.get()=='0':
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
                                    width=tempwidth
                        if len(pointmatch)>0:
                            print('pointmatch',pointmatch)
                            pointmatchdict.update({(pointmatch[0],pointmatch[1]):width})
                    widthsort=sorted(pointmatchdict,key=pointmatchdict.get,reverse=True)
                    try:
                        pointmatch=widthsort[0]
                    except:
                        continue
                    if len(pointmatch)>0:
                        x0=int(pointmatch[0][0])
                        y0=int(pointmatch[0][1])
                        x1=int(pointmatch[1][0])
                        y1=int(pointmatch[1][1])
                        if imgtypevar.get()=='0':
                            draw.line([(x0,y0),(x1,y1)],fill='yellow')
                        print('kernelwidth='+str(width*pixelmmratio))
                        print('kernellength='+str(kernellength*pixelmmratio))
                        #print('kernelwidth='+str(kernelwidth*pixelmmratio))
                        tempdict.update({uni:[kernellength,width,pixelmmratio**2*len(pixelloc[0]),kernellength*pixelmmratio,width*pixelmmratio]})


                    #print(event.x, event.y, labels[event.x, event.y], ulx, uly, rlx, rly)

                    #recborder = canvas.create_rectangle(ulx, uly, rlx, rly, outline='red')
                    #drawcontents.append(recborder)

            kernersizes.update({file:tempdict})
            originheight,originwidth=Multigraybands[file].size
            image=imageband.resize([originwidth,originheight],resample=Image.BILINEAR)
            image.save(path+'/'+originfile+'-sizeresult'+'.png',"PNG")
            tup=(labels,counts,colortable,[],currentfilename)
            _band,segimg=showcounting(tup,False)
            segimage=segimg.resize([originwidth,originheight],resample=Image.BILINEAR)
            segimage.save(path+'/'+originfile+'-segmentresult'+'.png',"PNG")
            _band,segimg=showcounting(tup,True)
            segimage=segimg.resize([originwidth,originheight],resample=Image.BILINEAR)
            segimage.save(path+'/'+originfile+'-labelresult'+'.png',"PNG")
            originrestoredband=np.copy(labels)
            restoredband=originrestoredband.astype('float32')
            #restoredband=cv2.resize(src=restoredband,dsize=(originwidth,originheight),interpolation=cv2.INTER_LINEAR)
            print(restoredband.shape)
            currentsizes=kernersizes[file]
            indicekeys=list(originbandarray[file].keys())
            indeclist=[ 0 for i in range(len(indicekeys)*3)]
            datatable={}
            origindata={}
            for key in indicekeys:
                data=originbandarray[file][key]
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
                try:
                    sizes=currentsizes[uni]
                except:
                    continue
                #templist=[amount,length,width]
                templist=[amount,sizes[0],sizes[1],sizes[2],sizes[3],sizes[4]]
                tempdict={colortable[uni]:templist+indeclist}  #NIR,Redeyes,R,G,B,NDVI,area
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
                datatable.update(tempdict)
            filename=path+'/'+originfile+'-outputdata.csv'
            with open(filename,mode='w') as f:
                csvwriter=csv.writer(f)
                rowcontent=['Index','Plot','Area(#pixel)','Length(#pixel)','Width(#pixel)','Area(mm2)','Length(mm)','Width(mm)']
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

def single_kmenas(singlebandarray):
    numindec=0
    keys=bandchoice.keys()
    for key in keys:
        tup=bandchoice[key].get()
        if '1' in tup:
            numindec+=1
    reshapeworkimg=np.zeros((singlebandarray[cluster[0]].shape[0]*singlebandarray[cluster[0]].shape[1],numindec))
    j=0
    for i in range(len(cluster)):
        tup=bandchoice[cluster[i]].get()
        if '1' in tup:
            tempband=singlebandarray[cluster[i]]
            reshapeworkimg[:,j]=tempband.reshape(tempband.shape[0]*tempband.shape[1],1)[:,0]
            j+=1
    clusternumber=int(kmeans.get())
    clf=KMeans(n_clusters=clusternumber,init='k-means++',n_init=10,random_state=0)
    labels=clf.fit_predict(reshapeworkimg)
    temptif=labels.reshape(singlebandarray[cluster[0]].shape[0],singlebandarray[cluster[0]].shape[1])
    keys=checkboxdict.keys()
    plantchoice=[]
    for key in keys:
        plantchoice.append(checkboxdict[key].get())
    tempdisplayimg=np.zeros((singlebandarray[cluster[0]].shape))
    #for i in range(len(plantchoice)):
    #    tup=plantchoice[i]
    #    if '1' in tup:
    #        tempdisplayimg=np.where(temptif==i,1,tempdisplayimg)

    pixelarea=1.0
    minipixelareaclass=0
    for i in range(int(kmeans.get())):
        pixelloc=np.where(temptif==i)
        pixelnum=len(pixelloc[0])
        temparea=float(pixelnum/(temptif.shape[0]*temptif.shape[1]))
        if temparea<pixelarea:
            minipixelareaclass=i
            pixelarea=temparea
    tempdisplayimg=np.where(temptif==minipixelareaclass,1,tempdisplayimg)
        #clusterdisplay.update({''.join(choicelist):tempdict})
    #return displaylabels
    return tempdisplayimg



def batchextraction():
    global multi_results
    for file in filenames:
        if file!=currentfilename:
            tempdisplaybands=displaybandarray[file]
            displayband=single_kmenas(tempdisplaybands)
            coin=refvar.get()=='1'
            edgevar=edge.get()=='1'
            if edgevar:
                displayband=removeedge(displayband)
            nonzeros=np.count_nonzero(displayband)
            nonzeroloc=np.where(displayband!=0)
            ulx,uly=min(nonzeroloc[1]),min(nonzeroloc[0])
            rlx,rly=max(nonzeroloc[1]),max(nonzeroloc[0])
            nonzeroratio=float(nonzeros)/((rlx-ulx)*(rly-uly))
            print(nonzeroratio)
            if coin:
                boundaryarea=tkintercorestat.boundarywatershed(displayband,1,'inner')
                boundaryarea=np.where(boundaryarea<1,0,boundaryarea)
                coindict,miniarea=tkintercorestat.findcoin(boundaryarea)
                coinarea=0
                topkey=list(coindict.keys())[0]
                coinarea=len(coindict[topkey][0])
                displayband[coindict[topkey]]=0
                nocoinarea=float(np.count_nonzero(displayband))/(displayband.shape[0]*displayband.shape[1])
                #ratio=findratio([currentlabels.shape[0],currentlabels.shape[1]],[1000,1000])
                print('nocoinarea',nocoinarea)
                coinratio=coinarea/(displayband.shape[0]*displayband.shape[1])
                print('coinratio:',coinratio)
                time.sleep(3)
                ratio=float(nocoinarea/coinratio)
                print('ratio:',ratio)
                if nonzeroratio<0.20:
                    #if coinratio**0.5<=0.2:# and nonzeroratio>=0.1:
                    ratio=findratio([displayband.shape[0],displayband.shape[1]],[1600,1600])
                    workingimg=cv2.resize(displayband,(int(displayband.shape[1]/ratio),int(displayband.shape[0]/ratio)),interpolation=cv2.INTER_LINEAR)
                else:
                    ratio=findratio([displayband.shape[0],displayband.shape[1]],[1000,1000])
                    workingimg=cv2.resize(displayband,(int(displayband.shape[1]/ratio),int(displayband.shape[0]/ratio)),interpolation=cv2.INTER_LINEAR)
            else:
                if nonzeroratio<0.20:# and nonzeroratio>=0.1:
                    ratio=findratio([displayband.shape[0],displayband.shape[1]],[1600,1600])
                    workingimg=cv2.resize(displayband,(int(displayband.shape[1]/ratio),int(displayband.shape[0]/ratio)),interpolation=cv2.INTER_LINEAR)
                else:
                    #workingimg=np.copy(displayband)
                    #if nonzeroratio>0.15:
                    ratio=findratio([displayband.shape[0],displayband.shape[1]],[1000,1000])
                    workingimg=cv2.resize(displayband,(int(displayband.shape[1]/ratio),int(displayband.shape[0]/ratio)),interpolation=cv2.INTER_LINEAR)
                    #else:
                    #    if nonzeroratio<0.1:
                    #        ratio=findratio([displayband.shape[0],displayband.shape[1]],[1503,1503])
                    #        workingimg=cv2.resize(displayband,(int(displayband.shape[1]*ratio),int(displayband.shape[0]*ratio)),interpolation=cv2.INTER_LINEAR)


            labels,border,colortable,greatareas,tinyareas,coinparts,labeldict=tkintercorestat.init(workingimg,workingimg,'',workingimg,10,coin)
            multi_results.update({file:(labeldict,coinparts)})
            tempimgdict={}
            tempimgbands={}
            for key in labeldict:
                tup=(labeldict[key]['labels'],labeldict[key]['counts'],labeldict[key]['colortable'],coinparts,file)
                outputdisplay,outputimg=showcounting(tup)
                tempimgdict.update({key:outputdisplay})
                tempimgbands.update({key:outputimg})
            outputimgdict.update({file:tempimgdict})
            outputimgbands.update({file:tempimgbands})
    pass


def resegment():
    global loccanvas,maxx,minx,maxy,miny,linelocs,bins,ybins,reseglabels,figcanvas,refvar,refsubframe,panelA
    global labelplotmap,figdotlist
    figcanvas.unbind('<Any-Enter>')
    figcanvas.unbind('<Any-Leave>')
    figcanvas.unbind('<Button-1>')
    figcanvas.unbind('<B1-Motion>')
    figcanvas.delete(ALL)
    #panelA.unbind('<Button-1>')
    #refvar.set('0')
    #for widget in refsubframe.winfo_children():
    #    widget.config(state=DISABLED)
    thresholds=[cal_xvalue(linelocs[0]),cal_xvalue(linelocs[1])]
    minthres=min(thresholds)
    maxthres=max(thresholds)
    lwthresholds=[cal_yvalue(linelocs[2]),cal_yvalue(linelocs[3])]
    maxlw=max(lwthresholds)
    minlw=min(lwthresholds)
    print(minthres,maxthres)
    #labels=np.copy(reseglabels)
    labels=np.copy(reseglabels)
    #if reseglabels is None:
    #    reseglabels,border,colortable,labeldict=tkintercorestat.resegmentinput(labels,minthres,maxthres,minlw,maxlw)

    if refarea is not None:
        labels[refarea]=0
    reseglabels,border,colortable,labeldict=tkintercorestat.resegmentinput(labels,minthres,maxthres,minlw,maxlw)
    multi_results.update({currentfilename:(labeldict,{})})
    iterkeys=list(labeldict.keys())
    iternum=len(iterkeys)
    print(labeldict)
    #iternum=3
    tempimgdict={}
    tempimgbands={}
    for key in labeldict:
        tup=(labeldict[key]['labels'],labeldict[key]['counts'],labeldict[key]['colortable'],{},currentfilename)
        outputdisplay,outputimg=showcounting(tup)
        tempimgdict.update({key:outputdisplay})
        tempimgbands.update({key:outputimg})
    outputimgdict.update({currentfilename:tempimgdict})
    outputimgbands.update({currentfilename:tempimgbands})
    changeoutputimg(currentfilename,'1')
    '''
    data=np.asarray(border[1:])
    hist,bin_edges=np.histogram(data,density=False)
    #figcanvas=Canvas(frame,width=400,height=350,bg='white')
    #figcanvas.pack()
    restoplot=createBins.createBins(hist.tolist(),bin_edges.tolist(),len(bin_edges))

    minx,maxx=histograms.plot(restoplot,hist.tolist(),bin_edges.tolist(),figcanvas)
    bins=bin_edges.tolist()
    loccanvas=figcanvas
    linelocs=[minx,maxx]
    '''
    data=[]
    uniquelabels=list(colortable.keys())
    lenwid=[]
    labelplotmap={}
    templabelplotmap={}
    for uni in uniquelabels:
        if uni!=0:
            pixelloc = np.where(reseglabels == uni)
            try:
                ulx = min(pixelloc[1])
            except:
                continue
            uly = min(pixelloc[0])
            rlx = max(pixelloc[1])
            rly = max(pixelloc[0])
            length=rly-uly
            width=rlx-ulx
            lenwid.append((length+width))
            data.append(len(pixelloc[0]))
            templabelplotmap.update({(len(pixelloc[0]),length+width):uni})
    miny=min(lenwid)
    maxy=max(lenwid)
    minx=min(data)
    maxx=max(data)
    binwidth=(maxx-minx)/10
    ybinwidth=(maxy-miny)/10
    bin_edges=[]
    y_bins=[]
    for i in range(0,11):
        bin_edges.append(minx+i*binwidth)
    for i in range(0,11):
        y_bins.append(miny+i*ybinwidth)
    #bin_edges.append(maxx)
    #bin_edges.append(maxx+binwidth)
    #y_bins.append(maxy)
    #y_bins.append(maxy+ybinwidth)
    plotdata=[]
    for i in range(len(data)):
        plotdata.append((data[i],lenwid[i]))
    scaledDatalist=[]
    try:
        x_scalefactor=300/(maxx-minx)
    except:
        return
    y_scalefactor=250/(maxy-miny)
    for (x,y) in plotdata:
        xval=50+(x-minx)*x_scalefactor+50
        yval=300-(y-miny)*y_scalefactor+25
        scaledDatalist.append((int(xval),int(yval)))
    for key in templabelplotmap:
        x=key[0]
        y=key[1]
        xval=50+(x-minx)*x_scalefactor+50
        yval=300-(y-miny)*y_scalefactor+25
        unilabel=templabelplotmap[key]
        labelplotmap.update({(int(xval),int(yval)):unilabel})
    figdotlist={}
    axistest.drawdots(25+50,325+25,375+50,25+25,bin_edges,y_bins,scaledDatalist,figcanvas,figdotlist)


    #loccanvas=figcanvas
    #minx=25
    #maxx=375
    #maxy=325
    #miny=25
    #linelocs=[25+12,375-12,325-12,25+12]
    #linelocs=[25+12,375-12,25+12,325-12]
    linelocs=[75+12,425-12,350-12,50+12]
    bins=bin_edges
    ybins=y_bins

    figcanvas.bind('<Any-Enter>',item_enter)
    figcanvas.bind('<Any-Leave>',item_leave)
    figcanvas.bind('<Button-1>',item_start_drag)
    figcanvas.bind('<B1-Motion>',item_drag)
    if refarea is not None:
        reseglabels[refarea]=65535



def cal_yvalue(y):
    y_scalefactor=250/(maxy-miny)
    yval=(300+25-y)/y_scalefactor+miny
    return yval

def cal_xvalue(x):
    #print(maxx,minx,max(bins),min(bins))
    #binwidth=(maxx-minx)/(max(bins)-min(bins))
    #binwidth=(max(bins)-min(bins))/12
    #print(x,minx,binwidth)
    #xloc=((x-minx)/binwidth)
    #print(xloc,min(bins))
    #value=min(bins)+xloc*binwidth
    #print(value)
    print(x)
    x_scalefactor=300/(maxx-minx)
    print(x_scalefactor)
    xval=(x-50-50)/x_scalefactor+minx

    #print(x,xval)
    return xval



def item_enter(event):
    global figcanvas
    figcanvas.config(cursor='hand2')
    figcanvas._restorItem=None
    figcanvas._restoreOpts=None
    itemType=figcanvas.type(CURRENT)
    #print(itemType)

    pass

def item_leave(event):
    global figcanvas
    pass

def item_start_drag(event):
    global figcanvas,linelocs,dotflash
    itemType=figcanvas.type(CURRENT)
    print(itemType)
    if itemType=='line':
        fill=figcanvas.itemconfigure(CURRENT,'fill')[4]
        dash=figcanvas.itemconfigure(CURRENT,'dash')[4]
        print('dashlen',len(dash))
        if fill=='red' and len(dash)>0:
            figcanvas._lastX=event.x
            #loccanvas._lastY=event.y
            linelocs[0]=event.x
        if fill=='red' and len(dash)==0:
            figcanvas._lastX=event.x
            #loccanvas._lastY=event.y
            linelocs[1]=event.x
        if fill=='blue' and len(dash)>0:
            figcanvas._lastY=event.y
            linelocs[2]=event.y
            #print('blue')
        if fill=='blue' and len(dash)==0:
            figcanvas._lastY=event.y
            linelocs[3]=event.y
            #print('purple')
        #if fill!='red' and fill!='orange':
        #    figcanvas._lastX=None
        #if fill!='blue' and fill!='purple':
        #    figcanvas._lastY=None
        print('linelocs',linelocs)
    else:
        if itemType=='oval':
            outline=figcanvas.itemconfigure(CURRENT,'outline')[4]
            print('outline',outline)
        currx=event.x
        curry=event.y
        print('x',currx,'y',curry)
        if (currx,curry) in labelplotmap: #or (currx-1,curry) in labelplotmap or (currx+1,curry) in labelplotmap\
                #or (currx,curry-1) in labelplotmap or (currx,curry+1) in labelplotmap:
            labelkey=labelplotmap[(currx,curry)]
        else:
            plotlist=list(labelplotmap.keys())
            distlist=[]
            for i in range(len(plotlist)):
                dist=(abs(currx-plotlist[i][0])+abs(curry-plotlist[i][1]))**0.5
                distlist.append(dist)
            shortestdist=min(distlist)
            shortestdistindex=distlist.index(shortestdist)
            labelkey=labelplotmap[plotlist[shortestdistindex]]
            if len(dotflash)>0:
                for i in range(len(dotflash)):
                    figcanvas.delete(dotflash.pop(0))
            dotx=plotlist[shortestdistindex][0]
            doty=plotlist[shortestdistindex][1]
            a=figcanvas.create_oval(dotx-1,doty-1,dotx+1,doty+1,width=1,outline='Orange',fill='Orange')
            dotflash.append(a)
        print(labelkey)
        seedfigflash(labelkey)

        '''
        if (currx-1,curry) in labelplotmap:
            labelkey=labelplotmap[(currx-1,curry)]
        if (currx+1,curry) in labelplotmap:
            labelkey=labelplotmap[(currx+1,curry)]
        if (currx,curry-1) in labelplotmap:
            labelkey=labelplotmap[(currx,curry-1)]
        if (currx,curry+1) in labelplotmap:
            labelkey=labelplotmap[(currx,curry+1)]
        if type(labelkey)!=type(None):
            if len(dotflash)>0:
                for i in range(len(dotflash)):
                    figcanvas.delete(dotflash.pop(0))
            a=figcanvas.create_oval(currx-1,curry-1,currx+1,curry+1,width=1,outline='Orange',fill='Orange')
            dotflash.append(a)
            #labelkey=labelplotmap[(currx,curry)]
            print(labelkey)
            seedfigflash(labelkey)
        '''
    ''''
        else:
            tup=figcanvas.find_all()
            print(tup)
            tup=list(tup)
            redarrow=tup[-4]
            orangearrow=tup[-3]
            bluearrow=tup[-2]
            purplearrow=tup[-1]
            currx=event.x
            curry=event.y
            if currx<75:
                currx=75
            if currx>425:
                currx=425
            if curry<50:
                curry=50
            if curry>350:
                curry=350
            dist=[abs(linelocs[0]-currx),abs(linelocs[1]-currx),abs(linelocs[2]-curry),abs(linelocs[3]-curry)]
            print(dist)
            #print(loccanvas.bbox(redarrow),loccanvas.bbox(orangearrow),loccanvas.bbox(bluearrow),loccanvas.bbox(purplearrow))
            mindist=min(dist)
            mindistind=dist.index(mindist)
            if mindistind==0:
                figcanvas.move(redarrow,currx-linelocs[0],0)
                figcanvas._lastX=currx
                linelocs[0]=currx
            if mindistind==1:
                figcanvas.move(orangearrow,currx-linelocs[1],0)
                figcanvas._lastX=currx
                linelocs[1]=currx
            if mindistind==2:
                figcanvas.move(bluearrow,0,curry-linelocs[2])
                linelocs[2]=curry
                figcanvas._lastY=curry
            if mindistind==3:
                figcanvas.move(purplearrow,0,curry-linelocs[3])
                linelocs[3]=curry
                figcanvas._lastY=curry
    pass
    '''''

def item_drag(event):
    global figcanvas,linelocs,xvalue
    x=event.x
    y=event.y
    if x<75:
        x=75
    if x>425:
        x=425
    if y<50:
        y=50
    if y>350:
        y=350
    try:
        fill=figcanvas.itemconfigure(CURRENT,'fill')[4]
        dash=figcanvas.itemconfigure(CURRENT,'dash')[4]
        print('dashlen',len(dash))
        print(fill)
    except:
        return
    #itemType=loccanvas.type(CURRENT)
    #try:
    #    test=0-loccanvas._lastX
    #    test=0-loccanvas._lastY
    #except:
    #    return

    if fill=='red': #or fill=='orange':
        figcanvas.move(CURRENT,x-figcanvas._lastX,0)
    if fill=='blue': #or fill=='purple':
        figcanvas.move(CURRENT,0,y-figcanvas._lastY)
    figcanvas._lastX=x
    figcanvas._lastY=y
    if fill=='red' and len(dash)>0:
        linelocs[0]=x
    if fill=='red' and len(dash)==0:
        linelocs[1]=x
    if fill=='blue' and len(dash)>0:
        linelocs[2]=y
    if fill=='blue' and len(dash)==0:
        linelocs[3]=y
            #print(line_a)
    #print(minline)
    #print(maxline)
    print('linelocs',linelocs)
    print(cal_xvalue(linelocs[0]),cal_xvalue(linelocs[1]),cal_yvalue(linelocs[2]),cal_yvalue(linelocs[3]))
    pass

def gen_convband():
    global convband
    if reseglabels is None:
        return
    processlabel=np.copy(reseglabels)
    ratio=findratio([processlabel.shape[0],processlabel.shape[1]],[850,850])
    #tempband=cv2.resize(processlabel.astype('float32'),(int(processlabel.shape[1]/ratio),int(processlabel.shape[0]/ratio)),interpolation=cv2.INTER_LINEAR)
    print(ratio)
    if int(ratio)>1:
        if processlabel.shape[0]*processlabel.shape[1]>850*850:
            convband,cache=tkintercorestat.pool_forward(processlabel,{"f":int(ratio),"stride":int(ratio)})
        else:
            cache=(np.zeros((processlabel.shape[0]*ratio,processlabel.shape[1]*ratio)),{"f":int(ratio),"stride":int(ratio)})
            convband=tkintercorestat.pool_backward(processlabel,cache)
    else:
        convband=processlabel


def process():
    if originlabels is None:
        extraction()
    else:
        if changekmeans==True:
            extraction()
        else:
            if linelocs[1]==425 and linelocs[3]==50:
                extraction()
            else:
                resegment()
    gen_convband()
    #highlightcoin()

#def extraction(frame):
def extraction():
    global kernersizes,multi_results,workingimg,outputimgdict,outputimgbands,pixelmmratio
    global currentlabels,panelA,outputbutton,reseglabels,refbutton,figcanvas,resegbutton,refvar
    global refsubframe,loccanvas,originlabels,changekmeans,originlabeldict,refarea
    global figdotlist
    if int(kmeans.get())==1:
        messagebox.showerror('Invalid Class #',message='#Class = 1, try change it to 2 or more, and refresh Color-Index.')
        return
    refarea=None
    multi_results.clear()
    kernersizes.clear()
    itervar=IntVar()
    outputimgdict.clear()
    outputimgbands.clear()
    #for widget in frame.winfo_children():
    #    widget.pack_forget()
    coin=refvar.get()=='1'
    edgevar=edge.get()=='1'
    if edgevar:
        currentlabels=removeedge(currentlabels)
    nonzeros=np.count_nonzero(currentlabels)
    nonzeroloc=np.where(currentlabels!=0)
    try:
        ulx,uly=min(nonzeroloc[1]),min(nonzeroloc[0])
    except:
        messagebox.showerror('Invalid Colorindices',message='Need to process colorindicies')
        return
    rlx,rly=max(nonzeroloc[1]),max(nonzeroloc[0])
    nonzeroratio=float(nonzeros)/((rlx-ulx)*(rly-uly))
    print(nonzeroratio)
    #nonzeroratio=float(nonzeros)/(currentlabels.shape[0]*currentlabels.shape[1])
    if nonzeroratio<=0.2:# and nonzeroratio>=0.1:
        ratio=findratio([currentlabels.shape[0],currentlabels.shape[1]],[1600,1600])
        if currentlabels.shape[0]*currentlabels.shape[1]>1600*1600:
            workingimg=cv2.resize(currentlabels,(int(currentlabels.shape[1]/ratio),int(currentlabels.shape[0]/ratio)),interpolation=cv2.INTER_LINEAR)
        else:
            ratio=1
            print('ratio',ratio)
            workingimg=np.copy(currentlabels)
    else:
        #if nonzeroratio>0.16:
        #if imgtypevar.get()=='0':
        #print('imgtype',imgtypevar.get())
        dealpixel=nonzeroratio*currentlabels.shape[0]*currentlabels.shape[1]
        print('deal pixel',dealpixel)
        if dealpixel>512000:
            if currentlabels.shape[0]*currentlabels.shape[1]>1000*1000:
                ratio=findratio([currentlabels.shape[0],currentlabels.shape[1]],[1000,1000])
                if ratio<2:
                    ratio=2
                workingimg=cv2.resize(currentlabels,(int(currentlabels.shape[1]/ratio),int(currentlabels.shape[0]/ratio)),interpolation=cv2.INTER_LINEAR)
        else:
            ratio=1
            print('ratio',ratio)
            workingimg=np.copy(currentlabels)
    pixelmmratio=1.0
    coin=False
    print('ratio:',ratio)
    print('workingimgsize:',workingimg.shape)
    if originlabels is None:
        originlabels,border,colortable,originlabeldict=tkintercorestat.init(workingimg,workingimg,'',workingimg,10,coin)
        changekmeans=False
    else:
        if changekmeans==True:
            originlabels,border,colortable,originlabeldict=tkintercorestat.init(workingimg,workingimg,'',workingimg,10,coin)
            changekmeans=False
    multi_results.update({currentfilename:(originlabeldict,{})})

    reseglabels=originlabels
    labeldict=originlabeldict
    colortable=originlabeldict['iter0']['colortable']
    iterkeys=list(labeldict.keys())
    iternum=len(iterkeys)
    print(labeldict)
    #iternum=3
    itervar.set(len(iterkeys))
    tempimgdict={}
    tempimgbands={}
    for key in labeldict:
        tup=(labeldict[key]['labels'],labeldict[key]['counts'],labeldict[key]['colortable'],{},currentfilename)
        outputdisplay,outputimg=showcounting(tup)
        tempimgdict.update({key:outputdisplay})
        tempimgbands.update({key:outputimg})
    outputimgdict.update({currentfilename:tempimgdict})
    outputimgbands.update({currentfilename:tempimgbands})
    #time.sleep(5)
    #tup=(labeldict,coinparts,currentfilename)
    #resscaler=Scale(frame,from_=1,to=iternum,tickinterval=1,length=220,orient=HORIZONTAL,variable=itervar,command=partial(changeoutputimg,currentfilename))
    #resscaler.pack()
    changeoutputimg(currentfilename,'1')
    processlabel=np.copy(reseglabels)
    tempband=np.copy(convband)
    panelA.bind('<Button-1>',lambda event,arg=processlabel:customcoin(event,processlabel,tempband))
    panelA.config(cursor='hand2')
    '''
    data=np.asarray(border[1:])
    hist,bin_edges=np.histogram(data,density=False)
    figcanvas=Canvas(frame,width=400,height=350,bg='white')
    figcanvas.pack()
    restoplot=createBins.createBins(hist.tolist(),bin_edges.tolist(),len(bin_edges))
    global minx,maxx,bins,loccanvas,linelocs
    minx,maxx=histograms.plot(restoplot,hist.tolist(),bin_edges.tolist(),figcanvas)
    bins=bin_edges.tolist()
    loccanvas=figcanvas
    linelocs=[minx,maxx]
    '''
    global loccanvas,maxx,minx,maxy,miny,linelocs,bins,ybins,figcanvas
    global labelplotmap
    data=[]
    uniquelabels=list(colortable.keys())
    lenwid=[]
    figcanvas.delete(ALL)
    labelplotmap={}
    templabelplotmap={}
    for uni in uniquelabels:
        if uni!=0:
            pixelloc = np.where(originlabels == uni)
            try:
                ulx = min(pixelloc[1])
            except:
                continue
            uly = min(pixelloc[0])
            rlx = max(pixelloc[1])
            rly = max(pixelloc[0])
            length=rly-uly
            width=rlx-ulx
            lenwid.append((length+width))
            data.append(len(pixelloc[0]))
            templabelplotmap.update({(len(pixelloc[0]),length+width):uni})
    miny=min(lenwid)
    maxy=max(lenwid)
    minx=min(data)
    maxx=max(data)
    binwidth=(maxx-minx)/10
    ybinwidth=(maxy-miny)/10
    bin_edges=[]
    y_bins=[]
    for i in range(0,11):
        bin_edges.append(minx+i*binwidth)
    for i in range(0,11):
        y_bins.append(miny+i*ybinwidth)
    #bin_edges.append(maxx)
    #bin_edges.append(maxx+binwidth)
    #y_bins.append(maxy)
    #y_bins.append(maxy+ybinwidth)
    plotdata=[]
    for i in range(len(data)):
        plotdata.append((data[i],lenwid[i]))
    scaledDatalist=[]
    x_scalefactor=300/(maxx-minx)
    y_scalefactor=250/(maxy-miny)
    for (x,y) in plotdata:
        xval=50+(x-minx)*x_scalefactor+50
        yval=300-(y-miny)*y_scalefactor+25
        scaledDatalist.append((int(xval),int(yval)))
    for key in templabelplotmap:
        x=key[0]
        y=key[1]
        xval=50+(x-minx)*x_scalefactor+50
        yval=300-(y-miny)*y_scalefactor+25
        unilabel=templabelplotmap[key]
        labelplotmap.update({(int(xval),int(yval)):unilabel})
    #print(labelplotmap)
    #print(scaledDatalist)
    figdotlist={}
    axistest.drawdots(25+50,325+25,375+50,25+25,bin_edges,y_bins,scaledDatalist,figcanvas,figdotlist)


    #loccanvas=figcanvas
    #minx=25
    #maxx=375
    #maxy=325
    #miny=25
    #[25,375,325,25]
    #linelocs=[25+12,375-12,25+12,325-12]
    linelocs=[75+12,425-12,350-12,50+12]
    #linelocs=[25+12,375-12,325-12,25+12]
    bins=bin_edges
    ybins=y_bins

    figcanvas.bind('<Any-Enter>',item_enter)
    figcanvas.bind('<Any-Leave>',item_leave)
    figcanvas.bind('<Button-1>',item_start_drag)
    figcanvas.bind('<B1-Motion>',item_drag)

    #reseg=Button(frame,text='Re-process',command=partial(resegment,labels,figcanvas),padx=5,pady=5)
    #reseg.pack()

    #if outputbutton is None:
    #    outputbutton=Button(control_fr,text='Export Results',command=partial(export_result,'0'),padx=5,pady=5)
    #    outputbutton.pack()
    #batchextraction()
    #else:
    #    outputbutton.pack_forget()
    #    outputbutton.pack()
    refbutton.config(state=NORMAL)
    refvar.set('0')
    for widget in refsubframe.winfo_children():
        #widget.config(state=DISABLED)
        widget.config(state=NORMAL)
    outputbutton.config(state=NORMAL)
    #resegbutton.config(state=NORMAL)
    pass

def onFrameConfigure(inputcanvas):
    '''Reset the scroll region to encompass the inner frame'''
    inputcanvas.configure(scrollregion=inputcanvas.bbox(ALL))




def removeedge(bands):
    global pointcontainer,displayorigin
    copyband=np.copy(bands)
    size=copyband.shape
    for i in range(20):
        copyband[i,:]=0  #up
        copyband[:,i]=0  #left
        copyband[:,size[1]-1-i]=0 #right
        copyband[size[0]-1-i,:]=0
    img=ImageTk.PhotoImage(Image.fromarray(copyband.astype('uint8')))
    displayimg['ColorIndices']['Image']=img
    changedisplayimg(imageframe,'ColorIndices')
    return copyband

def clustercontent(var):
    global cluster,bandchoice,contentframe
    bandchoice={}
    #if var=='0':

    #if var=='1':
    cluster=['LabOstu','NDI','Greenness','VEG','CIVE','MExG','NDVI','NGRDI','HEIGHT','Band1','Band2','Band3']
    for widget in contentframe.winfo_children():
        widget.pack_forget()
    for key in cluster:
        tempdict={key:Variable()}
        bandchoice.update(tempdict)
        ch=ttk.Checkbutton(contentframe,text=key,variable=bandchoice[key])#,command=changecluster)#,command=partial(autosetclassnumber,clusternumberentry,bandchoice))
        #if filedropvar.get()=='seedsample.JPG':
        #    if key=='NDI':
        #        ch.invoke()
        ch.pack(fill=X)

def findtempbandgap(locs):
    xloc=list(locs[1])
    yloc=list(locs[0])
    sortedx=sorted(xloc)
    gaps={}
    last=0
    for i in range(len(sortedx)):
        if sortedx[i]==sortedx[last]:
            continue
        isone = sortedx[i]-sortedx[last]==1
        if isone == False:
            gaps.update({(last,i-1):i-1-last+1})
        last=i
    print('xgaps',gaps,'len',len(sortedx))
    gaps={}
    last=0
    sortedy=sorted(yloc)
    for i in range(len(sortedy)):
        if sortedy[i]==sortedy[last]:
            continue
        isone = sortedy[i]-sortedy[last]==1
        if isone == False:
            gaps.update({(last,i-1):i-1-last+1})
        last=i
    print('ygaps',gaps,'len',len(sortedy))



def customcoin(event,processlabels,tempband):
    global panelA#refarea,
    global coinbox,reflabel,minflash
    global dotflash,figcanvas
    x=event.x
    y=event.y
    if len(minflash)>0:
        for i in range(len(minflash)):
            panelA.delete(minflash.pop(0))
    if len(dotflash)>0:
        for i in range(len(dotflash)):
            figcanvas.delete(dotflash.pop(0))
    panelA.delete(coinbox)
    tempband=np.copy(convband)
    #ratio=findratio([processlabels.shape[0],processlabels.shape[1]],[850,850])
    #tempband=cv2.resize(processlabels.astype('float32'),(int(processlabels.shape[1]/ratio),int(processlabels.shape[0]/ratio)),interpolation=cv2.INTER_LINEAR)
    #if processlabels.shape[0]*processlabels.shape[1]>850*850
    #    tempband=
    #tempband=tempband.astype('uint8')
    print(tempband.shape)
    coinlabel=tempband[y,x]
    print('coinlabel',coinlabel,'x',x,'y',y)
    #refarea=None
    if coinlabel==0:
        #messagebox.showerror('Invalid',message='Please pick areas have items.')
        return
    else:
        #refarea=np.where(processlabels==coinlabel)
        reflabel=coinlabel
        coinarea=np.where(tempband==coinlabel)
        #findtempbandgap(coinarea)
        ulx,uly=min(coinarea[1]),min(coinarea[0])
        rlx,rly=max(coinarea[1]),max(coinarea[0])
        #copytempband=np.copy(tempband)
        #temparea=copytempband[uly:rly+1,ulx:rlx+1]
        #copytempband[uly:rly+1,ulx:rlx+1]=tkintercorestat.tempbanddenoice(temparea,coinlabel,len(refarea[0])/(ratio**2))
        #coinarea=np.where(copytempband==coinlabel)
        unix=np.unique(coinarea[1]).tolist()
        uniy=np.unique(coinarea[0]).tolist()
        if len(unix)==1:
            ulx,rlx=unix[0],unix[0]
        else:
            ulx,rlx=min(coinarea[1]),max(coinarea[1])
        if len(uniy)==1:
            uly,rly=uniy[0],uniy[0]
        else:
            uly,rly=min(coinarea[0]),max(coinarea[0])
        '''
        try:
            ulx,uly=min(coinarea[1]),min(coinarea[0])
            rlx,rly=max(coinarea[1]),max(coinarea[0])
        except:
            coinarea=np.where(tempband==coinlabel)
            ulx,uly=min(coinarea[1]),min(coinarea[0])
            rlx,rly=max(coinarea[1]),max(coinarea[0])
        '''
        coinbox=panelA.create_rectangle(ulx,uly,rlx+1,rly+1,outline='yellow')
        plotcoinarea=np.where(reseglabels==coinlabel)
        ulx,uly=min(plotcoinarea[1]),min(plotcoinarea[0])
        rlx,rly=max(plotcoinarea[1]),max(plotcoinarea[0])
        unix=np.unique(plotcoinarea[1]).tolist()
        uniy=np.unique(plotcoinarea[0]).tolist()
        if len(unix)==1:
            ulx,rlx=unix[0],unix[0]
        else:
            ulx,rlx=min(plotcoinarea[1]),max(plotcoinarea[1])
        if len(uniy)==1:
            uly,rly=uniy[0],uniy[0]
        else:
            uly,rly=min(plotcoinarea[0]),max(plotcoinarea[0])
        lw=rlx-ulx+rly-uly
        area=len(plotcoinarea[0])
        print('lw',lw,'area',area)
        plotflash(lw,area,'Orange','Orange')
        #panelA.unbind('<Button-1>')


def magnify(event):
    global panelA
    x=event.x
    y=event.y
    grabimg=ImageGrab.grab((x-2,y-2,x+2,y+2))
    subimg=grabimg.resize((10,10))
    magnifier=panelA.create_image(x-3,y-3,image=ImageTk.PhotoImage(subimg))
    panelA.update()

def runflash(ulx,uly,rlx,rly,color):
    global minflash,panelA
    print(ulx,uly,rlx,rly)
    a=panelA.create_rectangle(ulx,uly,rlx+2,rly+2,outline=color)
    minflash.append(a)

def plotflash(lw,area,outlinecolor,fillcolor):
    global dotflash,figcanvas
    x_scalefactor=300/(maxx-minx)
    y_scalefactor=250/(maxy-miny)
    xval=50+(area-minx)*x_scalefactor+50
    yval=300-(lw-miny)*y_scalefactor+25
    a=figcanvas.create_oval(xval-1,yval-1,xval+1,yval+1,width=1,outline=outlinecolor,fill=fillcolor)
    dotflash.append(a)

def seedfigflash(topkey):
    global panelA,coinbox
    global reflabel,minflash
    tempband=np.copy(convband)
    if len(minflash)>0:
        for i in range(len(minflash)):
            panelA.delete(minflash.pop(0))
    panelA.delete(coinbox)
    reflabel=topkey
    coinarea=np.where(tempband==topkey)
    print(coinarea)
    ulx,uly=min(coinarea[1]),min(coinarea[0])
    rlx,rly=max(coinarea[1]),max(coinarea[0])
    unix=np.unique(coinarea[1]).tolist()
    uniy=np.unique(coinarea[0]).tolist()
    if len(unix)==1:
        ulx,rlx=unix[0],unix[0]
    else:
        ulx,rlx=min(coinarea[1]),max(coinarea[1])
    if len(uniy)==1:
        uly,rly=uniy[0],uniy[0]
    else:
        uly,rly=min(coinarea[0]),max(coinarea[0])
    coinbox=panelA.create_rectangle(ulx,uly,rlx+2,rly+2,outline='yellow')
    panelA.after(300,lambda :runflash(ulx,uly,rlx,rly,'red'))
    panelA.after(600,lambda :runflash(ulx,uly,rlx,rly,'yellow'))
    panelA.after(900,lambda :runflash(ulx,uly,rlx,rly,'red'))
    panelA.after(1200,lambda :runflash(ulx,uly,rlx,rly,'yellow'))
    panelA.after(1500,lambda :runflash(ulx,uly,rlx,rly,'red'))
    panelA.after(1800,lambda :runflash(ulx,uly,rlx,rly,'yellow'))



#def highlightcoin(processlabel,coindict,miniarea):
def highlightcoin():
    global coinbox,panelA #refarea,
    global reflabel,minflash
    global dotflash,figcanvas
    if convband is None:
        return
    tempband=np.copy(convband)
    #uniquel=np.unique(tempband)
    #print(uniquel)
    processlabel=np.copy(reseglabels)
    coinarea=0
    if len(minflash)>0:
        for i in range(len(minflash)):
            panelA.delete(minflash.pop(0))
    if len(dotflash)>0:
        for i in range(len(dotflash)):
            figcanvas.delete(dotflash.pop(0))
    panelA.delete(coinbox)
    unique,counts=np.unique(processlabel,return_counts=True)
    hist=dict(zip(unique[1:],counts[1:]))
    sortedlist=sorted(hist,key=hist.get,reverse=True)
    if coinsize.get()=='3':
        panelA.bind('<Button-1>',lambda event,arg=processlabel:customcoin(event,processlabel,tempband))
        panelA.config(cursor='hand2')
        #panelA.bind('<Motion>',magnify)
    else:
        if coinsize.get()=='1':
            topkey=sortedlist[0]
        if coinsize.get()=='2':
            topkey=sortedlist[-1]
            coinarea=np.where(tempband==topkey)
            i=2
            while(len(coinarea[0])==0):
                topkey=sortedlist[-i]
                coinarea=np.where(tempband==topkey)
                i+=1
            #copyboundary=np.copy(processlabel)
        reflabel=topkey
        #refarea=np.where(processlabel==topkey)
        print(topkey)
        coinarea=np.where(tempband==topkey)
        print(coinarea)
        ulx,uly=min(coinarea[1]),min(coinarea[0])
        rlx,rly=max(coinarea[1]),max(coinarea[0])
        #copytempband=np.copy(tempband.astype('int64'))
        #temparea=copytempband[uly:rly+1,ulx:rlx+1]
        #copytempband[uly:rly+1,ulx:rlx+1]=tkintercorestat.tempbanddenoice(temparea,topkey,len(refarea[0])/(ratio**2))
        #coinarea=np.where(copytempband==topkey)
        '''
        try:
            ulx,uly=min(coinarea[1]),min(coinarea[0])
            rlx,rly=max(coinarea[1]),max(coinarea[0])
        except:
            coinarea=np.where(tempband==topkey)
            ulx,uly=min(coinarea[1]),min(coinarea[0])
            rlx,rly=max(coinarea[1]),max(coinarea[0])
        '''
        unix=np.unique(coinarea[1]).tolist()
        uniy=np.unique(coinarea[0]).tolist()
        if len(unix)==1:
            ulx,rlx=unix[0],unix[0]
        else:
            ulx,rlx=min(coinarea[1]),max(coinarea[1])
        if len(uniy)==1:
            uly,rly=uniy[0],uniy[0]
        else:
            uly,rly=min(coinarea[0]),max(coinarea[0])
        coinbox=panelA.create_rectangle(ulx,uly,rlx+2,rly+2,outline='yellow')
        print('coinbox',ulx,uly,rlx,rly)
        if coinsize.get()=='2':
            panelA.after(500,lambda :runflash(ulx,uly,rlx,rly,'red'))
            panelA.after(1000,lambda :runflash(ulx,uly,rlx,rly,'yellow'))
            panelA.after(1500,lambda :runflash(ulx,uly,rlx,rly,'red'))
            panelA.after(2000,lambda :runflash(ulx,uly,rlx,rly,'yellow'))
            panelA.after(2500,lambda :runflash(ulx,uly,rlx,rly,'red'))
            panelA.after(3000,lambda :runflash(ulx,uly,rlx,rly,'yellow'))

        plotcoinarea=np.where(reseglabels==topkey)
        plotulx,plotuly=min(plotcoinarea[1]),min(plotcoinarea[0])
        plotrlx,plotrly=max(plotcoinarea[1]),max(plotcoinarea[0])
        unix=np.unique(plotcoinarea[1]).tolist()
        uniy=np.unique(plotcoinarea[0]).tolist()
        if len(unix)==1:
            plotulx,plotrlx=unix[0],unix[0]
        else:
            plotulx,plotrlx=min(plotcoinarea[1]),max(plotcoinarea[1])
        if len(uniy)==1:
            plotuly,plotrly=uniy[0],uniy[0]
        else:
            plotuly,plotrly=min(plotcoinarea[0]),max(plotcoinarea[0])
        lw=plotrlx-plotulx+plotrly-plotuly
        area=len(plotcoinarea[0])
        print('lw',lw,'area',area)
        plotflash(lw,area,'Orange','Orange')
        #figcanvas.after(1000,lambda :plotflash(lw,area,'black','SkyBlue'))
        #figcanvas.after(1500,lambda :plotflash(lw,area,'yellow','yellow'))
        #figcanvas.after(2000,lambda :plotflash(lw,area,'black','SkyBlue'))
        #figcanvas.after(2500,lambda :plotflash(lw,area,'yellow','yellow'))
        #figcanvas.after(3000,lambda :plotflash(lw,area,'black','SkyBlue'))

def del_reflabel():
    global reseglabels,panelA,loccanvas,linelocs,bins,ybins,figcanvas,maxx,minx,maxy,miny,refvar,refsubframe
    global labelplotmap
    processlabel=np.copy(reseglabels)
    refarea=np.where(processlabel==reflabel)
    reseglabels[refarea]=0
    gen_convband()
    panelA.delete(coinbox)
    reseglabels=tkintercorestat.renamelabels(reseglabels)
    newcolortables=tkintercorestat.get_colortable(reseglabels)
    newunique,newcounts=np.unique(reseglabels,return_counts=True)
    tup=(reseglabels,newcounts,newcolortables,{},currentfilename)
    outputdisplay,outputimg=showcounting(tup)
    tempimgdict={}
    tempimgbands={}
    tempimgdict.update({'iter0':outputdisplay})
    tempimgbands.update({'iter0':outputimg})
    outputimgdict.update({currentfilename:tempimgdict})
    outputimgbands.update({currentfilename:tempimgbands})
    changeoutputimg(currentfilename,'1')
    #update plot
    print('done image')
    copyplotmap=labelplotmap.copy()
    for k,v in copyplotmap.items():
        if v==reflabel:
            figindex=figdotlist[k]
            figcanvas.delete(figindex)
    tup=list(figcanvas.find_all())
    figcanvas.delete(tup[-1])
    '''
    data=[]
    uniquelabels=list(newcolortables.keys())
    lenwid=[]
    labelplotmap={}
    templabelplotmap={}
    for uni in uniquelabels:
        if uni!=0:
            pixelloc = np.where(reseglabels == uni)
            try:
                ulx = min(pixelloc[1])
            except:
                continue
            uly = min(pixelloc[0])
            rlx = max(pixelloc[1])
            rly = max(pixelloc[0])
            length=rly-uly
            width=rlx-ulx
            lenwid.append((length+width))
            data.append(len(pixelloc[0]))
            templabelplotmap.update({(len(pixelloc[0]),length+width):uni})
    miny=min(lenwid)
    maxy=max(lenwid)
    minx=min(data)
    maxx=max(data)
    binwidth=(maxx-minx)/10
    ybinwidth=(maxy-miny)/10
    bin_edges=[]
    y_bins=[]
    for i in range(0,11):
        bin_edges.append(minx+i*binwidth)
    for i in range(0,11):
        y_bins.append(miny+i*ybinwidth)
    #bin_edges.append(maxx)
    #bin_edges.append(maxx+binwidth)
    #y_bins.append(maxy)
    #y_bins.append(maxy+ybinwidth)
    plotdata=[]
    for i in range(len(data)):
        plotdata.append((data[i],lenwid[i]))
    scaledDatalist=[]
    try:
        x_scalefactor=300/(maxx-minx)
    except:
        return
    y_scalefactor=250/(maxy-miny)
    for (x,y) in plotdata:
        xval=50+(x-minx)*x_scalefactor+50
        yval=300-(y-miny)*y_scalefactor+25
        scaledDatalist.append((int(xval),int(yval)))
    for key in templabelplotmap:
        x=key[0]
        y=key[1]
        xval=50+(x-minx)*x_scalefactor+50
        yval=300-(y-miny)*y_scalefactor+25
        unilabel=templabelplotmap[key]
        labelplotmap.update({(int(xval),int(yval)):unilabel})
    figcanvas.delete(ALL)
    axistest.drawdots(25+50,325+25,375+50,25+25,bin_edges,y_bins,scaledDatalist,figcanvas)


    #loccanvas=figcanvas
    #minx=25
    #maxx=375
    #maxy=325
    #miny=25
    #linelocs=[25+12,375-12,325-12,25+12]
    #linelocs=[25+12,375-12,25+12,325-12]
    linelocs=[75+12,425-12,350-12,50+12]
    bins=bin_edges
    ybins=y_bins

    figcanvas.bind('<Any-Enter>',item_enter)
    figcanvas.bind('<Any-Leave>',item_leave)
    figcanvas.bind('<Button-1>',item_start_drag)
    figcanvas.bind('<B1-Motion>',item_drag)
    '''


#def refchoice(refsubframe):
def refchoice():
    global coinsize,sizeentry,coinbox,panelA,boundaryarea,coindict,convband
    global refarea
    #refsubframe.grid_forget()
    #for widget in refsubframe.winfo_children():
    #    widget.pack_forget()
    #panelA.delete(coinbox)
    if refvar.get()=='1':
        if type(currentlabels)==type(None):
            messagebox.showerror('Invalid Option',message='Should get # class >=2 color index image first.')
            return
        processlabel=np.copy(reseglabels)
        refarea=np.where(processlabel==reflabel)
        print('refarea',len(refarea[0]))
        print('reflabel',reflabel)
        #refsubframe.pack(side=BOTTOM)
        #refsubframe.grid(row=1,column=0,columnspan=4)
        #refoption=[('Use Maximum','1'),('Use Minimum','2'),('User Specify','3')]
        #for widget in refsubframe.winfo_children():
        #    widget.config(state=NORMAL)
        '''
        if reseglabels is None:
            return
        processlabel=np.copy(reseglabels)
        ratio=findratio([processlabel.shape[0],processlabel.shape[1]],[850,850])
        #tempband=cv2.resize(processlabel.astype('float32'),(int(processlabel.shape[1]/ratio),int(processlabel.shape[0]/ratio)),interpolation=cv2.INTER_LINEAR)
        print(ratio)
        if int(ratio)>1:
            if processlabel.shape[0]*processlabel.shape[1]>850*850:
                convband,cache=tkintercorestat.pool_forward(processlabel,{"f":int(ratio),"stride":int(ratio)})
            else:
                cache=(np.zeros((processlabel.shape[0]*ratio,processlabel.shape[1]*ratio)),{"f":int(ratio),"stride":int(ratio)})
                convband=tkintercorestat.pool_backward(processlabel,cache)
        else:
            convband=processlabel
        '''
        #highlightcoin()
        #if reseglabels is None:
        #    boundaryarea=tkintercorestat.boundarywatershed(currentlabels,1,'inner')
        #    boundaryarea=np.where(boundaryarea<1,0,boundaryarea)
        #    coindict,miniarea=tkintercorestat.findcoin(boundaryarea)
        #    processlabels=np.copy(boundaryarea)
        #else:
        #coindict,miniarea=tkintercorestat.findcoin(reseglabels)
        #processlabels=np.copy(reseglabels)
    #if refvar.get()=='0':
    #    for widget in refsubframe.winfo_children():
    #        widget.config(state=DISABLED)
        #panelA.unbind('<Button-1>')
    else:
        refarea=None




## ----Interface----


## ----Display----
display_fr=Frame(root,width=640,height=640)
control_fr=Frame(root,width=320,height=320)
bottomframe=Frame(root)
bottomframe.pack(side=BOTTOM)
display_fr.pack(side=LEFT)
control_fr.pack(side=LEFT)
#display_label=Text(display_fr,height=1,width=100)
#display_label.tag_config("just",justify=CENTER)
#display_label.insert(END,'Display Panel',"just")
#display_label.configure(state=DISABLED)
#display_label.pack(padx=10,pady=10)

imgtypevar.set('0')
Open_File('seedsample.JPG')
singleband('seedsample.JPG')
#cal indices
generatedisplayimg('seedsample.JPG')



imageframe=LabelFrame(display_fr,bd=0)
imageframe.pack()

#panelA=Label(imageframe,text='Display Panel',image=displayimg['Origin']) #620 x 620
l=displayimg['Origin']['Size'][0]
w=displayimg['Origin']['Size'][1]
panelA=Canvas(imageframe,width=w,height=l,bg='white')
panelA.create_image(0,0,image=displayimg['Origin']['Image'],anchor=NW)
panelA.pack(padx=20,pady=20,expand=YES)


buttondisplay=LabelFrame(bottomframe,bd=0)
buttondisplay.config(cursor='hand2')
buttondisplay.pack(side=LEFT)

openfilebutton=Button(buttondisplay,text='Image',command=Open_Multifile,cursor='hand2')
openfilebutton.pack(side=LEFT,padx=20,pady=5)
mapbutton=Button(buttondisplay,text='Map',cursor='hand2',command=Open_Map)
mapbutton.pack(side=LEFT,padx=20,pady=5)

disbuttonoption={'Origin':'1','ColorIndices':'3','Color Deviation':'2','Output':'4'}
#disbuttonoption={'Origin':'1','ColorIndices':'3','Output':'4'}
for text in disbuttonoption:
    b=Radiobutton(buttondisplay,text=text,variable=displaybut_var,value=disbuttonoption[text],command=partial(changedisplayimg,imageframe,text))
    b.pack(side=LEFT,padx=20,pady=5)
    if disbuttonoption[text]=='1':
        b.invoke()
### ---open file----


## ----Control----
#control_label=Text(control_fr,height=1,width=50)
#control_label.tag_config("just",justify=CENTER)
#control_label.insert(END,'Control Panel',"just")
#control_label.configure(state=DISABLED)
#control_label.pack()

filter_fr=LabelFrame(control_fr,bd=0)
filter_fr.pack()
imgtypeframe=LabelFrame(filter_fr,text='Image type',bd=0)
#imgtypeframe.pack()
imgtypeoption=[('Crop plots','1'),('Grain kernel','0')]
for text,mode in imgtypeoption:
    b=Radiobutton(imgtypeframe,text=text,variable=imgtypevar,value=mode,command=partial(clustercontent,mode))
    #b.pack(side=LEFT,padx=6)

### ---change file---
changefileframe=LabelFrame(filter_fr,text='Change Files',cursor='hand2')
#changefileframe.pack()

filedropvar.set(filenames[0])
changefiledrop=OptionMenu(changefileframe,filedropvar,*filenames,command=partial(changeimage,imageframe))
changefiledrop.pack()
### ---choose color indices---
chframe=LabelFrame(filter_fr,text='Select indicies below',cursor='hand2',bd=0)
chframe.pack()
chcanvas=Canvas(chframe,width=200,height=110,scrollregion=(0,0,400,400))
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
    ch=ttk.Checkbutton(contentframe,text=key,variable=bandchoice[key])#,command=changecluster)#,command=partial(autosetclassnumber,clusternumberentry,bandchoice))
    if filedropvar.get()=='seedsample.JPG':
        if key=='LabOstu':
            ch.invoke()
    ch.pack(fill=X)

### ----Class NUM----
kmeansgenframe=LabelFrame(filter_fr,text='Select # of class',cursor='hand2',bd=0)
kmeansgenframe.pack()
kmeanslabel=LabelFrame(kmeansgenframe,bd=0)
checkboxframe=LabelFrame(kmeansgenframe,cursor='hand2',bd=0)#,text='Select classes',cursor='hand2')
kmeanslabel.pack()

kmeans.set(2)
#kmeansbar=Scale(kmeanslabel,from_=1,to=10,tickinterval=1,length=270,showvalue=0,orient=HORIZONTAL,variable=kmeans,command=partial(generatecheckbox,checkboxframe))
kmeansbar=ttk.Scale(kmeanslabel,from_=1,to=10,length=350,orient=HORIZONTAL,variable=kmeans,cursor='hand2',command=partial(generatecheckbox,checkboxframe))
kmeansbar.pack()

kmeansbar.bind('<ButtonRelease-1>',changecluster)

checkboxframe.pack()
for i in range(10):
    dictkey=str(i+1)
    tempdict={dictkey:Variable()}
    if i==1:
        tempdict[dictkey].set('1')
    else:
        tempdict[dictkey].set('0')
    checkboxdict.update(tempdict)
    ch=Checkbutton(checkboxframe,text=dictkey,variable=checkboxdict[dictkey],command=partial(changecluster,''))
    if i+1>int(kmeans.get()):
        ch.config(state=DISABLED)
    ch.pack(side=LEFT)

#kmeanscanvasframe=LabelFrame(kmeansgenframe,bd='0')
#kmeanscanvasframe.pack()
#kmeanscanvas=Canvas(kmeanscanvasframe,width=350,height=10,bg='blue')
#kmeanscanvas.pack()

reshapemodified_tif=np.zeros((displaybandarray[currentfilename]['LabOstu'].shape[0]*displaybandarray[currentfilename]['LabOstu'].shape[1],1))
colordicesband=kmeansclassify(['LabOstu'],reshapemodified_tif)
generateimgplant(colordicesband)
changedisplayimg(imageframe,'Origin')
#generatecheckbox(checkboxframe,2)

#refreshebutton=Button(filter_fr,text='Refresh ColorIndices',cursor='hand2',command=changecluster)
#refreshebutton.pack()
### --- ref and edge settings ---

#for text,mode in refoption:
#    b=Radiobutton(refframe,text=text,variable=refvar,value=mode,command=partial(refchoice,refsubframe))
    #b.pack(side=LEFT,padx=15)
#    b.grid(row=0,column=column)
#    column+=1

edgeframe=LabelFrame(filter_fr,text='Edge remove setting')
#edgeframe.pack()
edgeoption=[('Remove edge','1'),('Keep same','0')]

edge.set('0')
for text,mode in edgeoption:
    b=Radiobutton(edgeframe,text=text,variable=edge,value=mode)
    b.pack(side=LEFT,padx=6)

### ---start extraction---
#extractionframe=LabelFrame(control_fr,cursor='hand2',bd=0)
#extractionframe.pack(padx=5,pady=5)
resviewframe=LabelFrame(control_fr,cursor='hand2',bd=0)
figcanvas=Canvas(resviewframe,width=450,height=400,bg='white')
figcanvas.pack()
resviewframe.pack()
#refframe=LabelFrame(control_fr,cursor='hand2',bd=0)
refframe=LabelFrame(bottomframe,cursor='hand2',bd=0)
refframe.pack(side=LEFT)

refoption=[('Use Ref','1'),('No Ref','0')]
refvar.set('0')
refsubframe=LabelFrame(refframe,bd=0)
column=0
#refoption=[('Max','1'),('Min','2'),('Spec','3')]
#for text,mode in refoption:
#    b=Radiobutton(refsubframe,text=text,variable=coinsize,value=mode,command=highlightcoin)#,command=partial(highlightcoin,processlabels,coindict,miniarea))
#    b.pack(side=LEFT,padx=5)
#    if mode=='1':
#        b.invoke()
refsubframe.pack(side=LEFT)
delrefbutton=Button(refsubframe,text='Del',command=del_reflabel)
delrefbutton.pack(side=LEFT, padx=20,pady=5)
refbutton=Checkbutton(refsubframe,text='Ref',variable=refvar,command=refchoice)
#refbutton.config(state=DISABLED)
refbutton.pack(side=LEFT)
sizeentry=Entry(refsubframe,width=5)
sizeentry.insert(END,285)
sizeentry.pack(side=LEFT,padx=5)
sizeunit=Label(refsubframe,text='mm^2')
sizeunit.pack(side=LEFT)

#delrefbutton.config(state=DISABLED)
#refbutton=Checkbutton(refsubframe,text='Ref',variable=refvar,command=partial(refchoice,refsubframe))

for widget in refsubframe.winfo_children():
    widget.config(state=DISABLED)
#extractbutton=Button(refframe,text='Process',command=partial(extraction))
extractbutton=Button(refframe,text='Process',command=process)
extractbutton.configure(activebackground='blue')
extractbutton.pack(side=LEFT,padx=20,pady=5)
outputbutton=Button(refframe,text='Export',command=partial(export_result,'0'))
outputbutton.pack(side=LEFT,padx=20,pady=5)
outputbutton.config(state=DISABLED)
#resegbutton=Button(extractionframe,text='Re-Segment',command=resegment)
#resegbutton.pack(side=LEFT)
#resegbutton.config(state=DISABLED)
changekmeans=False
root.mainloop()

