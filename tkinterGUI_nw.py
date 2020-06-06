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
import scipy.linalg as la
from functools import partial
import threading
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
from multiprocessing import Pool
import lm_method
import batchprocess

class img():
    def __init__(self,size,bands):
        self.size=size
        self.bands=bands

import batchprocess

displayimg={'Origin':None,
            'PCs':None,
            'Color Deviation':None,
            'ColorIndices':None,
            'Output':None}
#cluster=['LabOstu','NDI'] #,'Greenness','VEG','CIVE','MExG','NDVI','NGRDI','HEIGHT']
cluster=['LabOstu','NDI','Greenness','VEG','CIVE','MExG','NDVI','NGRDI','HEIGHT','Band1','Band2','Band3']
colorbandtable=np.array([[255,0,0],[255,127,0],[255,255,0],[127,255,0],[0,255,255],[0,127,255],[0,0,255],[127,0,255],[75,0,130],[255,0,255]],'uint8')
#print('colortableshape',colortable.shape)
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
outputsegbands={}
originsegbands={}
oldpcachoice=[]
multiselectitems=[]
coinbox_list=[]
pre_checkbox=[]
originpcabands={}


batch={'PCs':[],
       'Kmeans':[],
       'Kmeans_sel':[],
       'Area_max':[],
       'Area_min':[],
       'shape_max':[],
       'shape_min':[]}



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

#minipixelareaclass=0

coinbox=None

currentfilename=''
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
havecolorstrip=True
originbinaryimg=None

maxx=0
minx=0
bins=None
loccanvas=None
linelocs=[0,0,0,0]
maxy=0
miny=0

segmentratio=0

zoombox=[]

def distance(p1,p2):
    return np.sum((p1-p2)**2)




def findratio(originsize,objectsize):
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

def deletezoom(event,widget):
    print('leave widget')
    if len(zoombox)>0:
        for i in range(len(zoombox)):
            #print('delete')
            widget.delete(zoombox.pop(0))
    widget.update()

def zoom(event,widget,img):
    global zoombox
    x=event.x
    y=event.y
    #print(x,y)
    if len(zoombox)>1:
        widget.delete(zoombox.pop(0))
        #print('delete')
    crop=img.crop((x-15,y-15,x+15,y+15))
    w,h=crop.size
    #print(w,h)
    crop=crop.resize([w*3,h*3],resample=Image.BILINEAR)
    w,h=crop.size
    crop=ImageTk.PhotoImage(crop)
    zoombox.append(widget.create_image(x+5,y-5,image=crop))
    root.update_idletasks()
    raise NameError
    #time.sleep(0.1)


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
    widget.update()
    if text=='Output':
        try:
            image=outputsegbands[currentfilename]['iter0']
        except:
            return
        widget.bind('<Motion>',lambda event,arg=widget:zoom(event,arg,image))
        widget.bind('<Leave>',lambda event,arg=widget:deletezoom(event,arg))
    else:
        if text=='Origin':
            try:
                image=originsegbands['Origin']
            except:
                return
            widget.bind('<Motion>',lambda event,arg=widget:zoom(event,arg,image))
            widget.bind('<Leave>',lambda event,arg=widget:deletezoom(event,arg))
        else:
            widget.unbind('<Motion>')

    #print('change to '+text)
    #time.sleep(1)

def generatedisplayimg(filename):  # init display images

    try:
        # firstimg=Multiimagebands[filename]
        #height,width=firstimg.size
        # height,width,c=displaybandarray[filename]['LabOstu'].shape
        bandsize=Multiimagebands[filename].size
        if bandsize[0]*bandsize[1]>2000*2000:
            ratio=findratio([bandsize[0],bandsize[1]],[2000,2000])
        else:
            ratio=1
        height,width=bandsize[0]/ratio,bandsize[1]/ratio
        ratio=findratio([height,width],[850,850])
        print('displayimg ratio',ratio)
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
        originimg=Image.fromarray(resize.astype('uint8'))
        originsegbands.update({'Origin':originimg})

        rgbimg=Image.fromarray(resize.astype('uint8'))
        draw=ImageDraw.Draw(rgbimg)
        suggsize=14
        font=ImageFont.truetype('cmb10.ttf',size=suggsize)
        content='\n File: '+currentfilename
        draw.text((10-1, 10+1), text=content, font=font, fill='white')
        draw.text((10+1, 10+1), text=content, font=font, fill='white')
        draw.text((10-1, 10-1), text=content, font=font, fill='white')
        draw.text((10+1, 10-1), text=content, font=font, fill='white')
        #draw.text((10,10),text=content,font=font,fill=(141,2,31,0))
        draw.text((10,10),text=content,font=font,fill='black')
        rgbimg=ImageTk.PhotoImage(rgbimg)
        tempdict={}
        tempdict.update({'Size':resize.shape})
        tempdict.update({'Image':rgbimg})
    except:
        tempdict={}
        tempimg=np.zeros((850,850))

        tempdict.update({'Size':tempimg.shape})
        tempdict.update({'Image':ImageTk.PhotoImage(Image.fromarray(tempimg.astype('uint8')))})
    displayimg['Origin']=tempdict
    #if height*width<850*850:
    #    resize=cv2.resize(Multigray[filename],(int(width*ratio),int(height*ratio)),interpolation=cv2.INTER_LINEAR)
    #else:
        #resize=cv2.resize(Multigray[filename],(int(width/ratio),int(height/ratio)),interpolation=cv2.INTER_LINEAR)
    tempimg=np.zeros((850,850))
    tempdict={}
    try:
        tempdict.update({'Size':resize.shape})
        tempdict.update({'Image':ImageTk.PhotoImage(Image.fromarray(np.zeros((int(resizeshape[1]),int(resizeshape[0]))).astype('uint8')))})

    except:
        tempdict.update({'Size':tempimg.shape})
    #if height*width<850*850:
    #    tempdict.update({'Image':ImageTk.PhotoImage(Image.fromarray(np.zeros((int(height*ratio),int(width*ratio))).astype('uint8')))})
    #else:
    #    tempdict.update({'Image':ImageTk.PhotoImage(Image.fromarray(np.zeros((int(height/ratio),int(width/ratio))).astype('uint8')))})
    # tempdict.update({'Image':ImageTk.PhotoImage(Image.fromarray(np.zeros((int(resizeshape[1]),int(resizeshape[0]))).astype('uint8')))})
        tempdict.update({'Image':ImageTk.PhotoImage(Image.fromarray(tempimg.astype('uint8')))})
    displayimg['Output']=tempdict

    tempdict={}
    try:
        tempdict.update({'Size':resize.shape})
        tempdict.update({'Image':ImageTk.PhotoImage(Image.fromarray(np.zeros((int(resizeshape[1]),int(resizeshape[0]))).astype('uint8')))})
    except:
        tempdict.update({'Size':tempimg.shape})
        tempdict.update({'Image':ImageTk.PhotoImage(Image.fromarray(tempimg.astype('uint8')))})
    displayimg['PCs']=tempdict

    tempdict={}

    try:
        tempband=np.zeros((displaybandarray[filename]['LabOstu'].shape))
        tempband=tempband+displaybandarray[filename]['LabOstu']
    # ratio=findratio([tempband.shape[0],tempband.shape[1]],[850,850])

    #if tempband.shape[0]*tempband.shape[1]<850*850:
    #    tempband=cv2.resize(ratio,(int(tempband.shape[1]*ratio),int(tempband.shape[0]*ratio)),interpolation=cv2.INTER_LINEAR)
    #else:
    #    tempband=cv2.resize(ratio,(int(tempband.shape[1]/ratio),int(tempband.shape[0]/ratio)),interpolation=cv2.INTER_LINEAR)
        tempband=cv2.resize(tempband,(int(resizeshape[0]),int(resizeshape[1])),interpolation=cv2.INTER_LINEAR)
        tempdict.update({'Size':tempband.shape})
        tempdict.update({'Image':ImageTk.PhotoImage(Image.fromarray(tempband[:,:,0].astype('uint8')))})

    # print('resizeshape',resizeshape)
    #pyplt.imsave('displayimg.png',tempband[:,:,0])
    #indimg=cv2.imread('displayimg.png')
    except:
        tempdict.update({'Size':tempimg.shape})
        tempdict.update({'Image':ImageTk.PhotoImage(Image.fromarray(tempimg.astype('uint8')))})

    displayimg['ColorIndices']=tempdict

    #resize=cv2.resize(Multigray[filename],(int(resizeshape[0]),int(resizeshape[1])),interpolation=cv2.INTER_LINEAR)
    #grayimg=ImageTk.PhotoImage(Image.fromarray(resize.astype('uint8')))
    #tempdict={}
    #tempdict.update({'Size':resize.shape})
    #tempdict.update({'Image':grayimg})
    tempdict={}
    try:
        colordeviate=np.zeros((tempband[:,:,0].shape[0],tempband[:,:,0].shape[1],3),'uint8')
        kvar=int(kmeans.get())
        for i in range(kvar):
            locs=np.where(tempband[:,:,0]==i)
            colordeviate[locs]=colorbandtable[i,:]

    # pyplt.imsave('colordeviation.png',colordeviate)
    # # colordevimg=Image.fromarray(colordeviate.astype('uint8'))
    # # colordevimg.save('colordeviation.png',"PNG")
    # testcolor=Image.open('colordeviation.png')
        print('colordeviation.png')
    # colortempdict={}
        tempdict.update({'Size':colordeviate.shape})
        tempdict.update({'Image':ImageTk.PhotoImage(Image.fromarray(colordeviate.astype('uint8')))})
    # colortempdict.update({'Size':colordeviate.shape})
    # colortempdict.update({'Image':ImageTk.PhotoImage(Image.fromarray(colordeviate.astype('uint8')))})

    # colortempdict.update({'Image':ImageTk.PhotoImage(testcolor)})

    # tempdict={}
    except:
        tempdict.update({'Size':tempimg.shape})
        tempdict.update({'Image':ImageTk.PhotoImage(Image.fromarray(tempimg.astype('uint8')))})

    # displayimg['Color Deviation']=colortempdict
    displayimg['Color Deviation']=tempdict



def Open_File(filename):   #add to multi-image,multi-gray  #call band calculation
    global Multiimage,Multigray,Multitype,Multiimagebands,Multigraybands,filenames
    try:
        Filersc=cv2.imread(filename,flags=cv2.IMREAD_ANYCOLOR)
        height,width,channel=np.shape(Filersc)
        Filesize=(height,width)
        print('filesize:',height,width)
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

def Open_Map():
    if proc_mode[proc_name].get()=='1':
        batchprocess.Open_batchfile()

        return

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
        mapdict,mapimage,smallset=showcounting(tup,True,True,False)
        tempimgbands={}
        tempimgdict={}
        tempsmall={}
        tempimgbands.update({'iter0':mapimage})
        tempimgdict.update({'iter0':mapdict})
        tempsmall.update({'iter0':smallset})
        outputimgdict.update({currentfilename:tempimgdict})
        outputimgbands.update({currentfilename:tempimgbands})
        outputsegbands.update({currentfilename:tempsmall})
        changeoutputimg(currentfilename,'1')

def Open_Multifile():
    global extractbutton,outputbutton
    if proc_mode[proc_name].get()=='1':
        batchprocess.Open_batchfolder()
        extractbutton.config(state=NORMAL)
        outputbutton.config(state=NORMAL)
        return
    # else:
    #     extractbutton.config(state=DISABLED)

    global Multiimage,Multigray,Multitype,Multiimagebands,changefileframe,imageframe,Multigraybands,filenames
    global changefiledrop,filedropvar,originbandarray,displaybandarray,clusterdisplay,currentfilename,resviewframe
    global refsubframe,reseglabels,refbutton,figcanvas,loccanvas,originlabels,changekmeans,refarea
    global originlabeldict,convband,panelA
    global havecolorstrip
    global colordicesband,oldpcachoice

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
        oldpcachoice=[]
        reseglabels=None
        originlabels=None
        originlabeldict=None
        #changekmeans=True
        convband=None
        refvar.set('0')
        kmeans.set('2')
        panelA.delete(ALL)
        panelA.unbind('<Button-1>')
        panelA.unbind('<Shift-Button-1>')
        refarea=None
        havecolorstrip=False
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
            generatedisplayimg(filenames[0])
            changedisplayimg(imageframe,'Origin')
            # imageframe.update()
            # raise NameError
            # yield
            thread=threading.Thread(target=singleband,args=(MULTIFILES[i],))
            # singleband(MULTIFILES[i])
            thread.start()
            thread.join()
        for widget in changefileframe.winfo_children():
            widget.pack_forget()
        currentfilename=filenames[0]
        filedropvar.set(filenames[0])
        changefiledrop=OptionMenu(changefileframe,filedropvar,*filenames,command=partial(changeimage,imageframe))
        changefiledrop.pack()
        #singleband(filenames[0])
        generatedisplayimg(filenames[0])
        getPCs()

        if len(bandchoice)>0:
            for i in range(len(cluster)):
                bandchoice[cluster[i]].set('')
        #changedisplayimg(imageframe,'Origin')
        kmeans.set(1)
        #reshapemodified_tif=np.zeros((displaybandarray[currentfilename]['LabOstu'].shape[0]*displaybandarray[currentfilename]['LabOstu'].shape[1],3))
        #colordicesband=kmeansclassify(['LabOstu'],reshapemodified_tif)
        colordicesband=kmeansclassify()
        generateimgplant(colordicesband)
        changedisplayimg(imageframe,'Origin')
        if len(bandchoice)>0:
            bandchoice['LabOstu'].set('1')

        global buttondisplay,pcaframe,kmeansbar
        for widget in buttondisplay.winfo_children():
            widget.config(state=NORMAL)
        for widget in pcaframe.winfo_children():
            widget.config(state=NORMAL)
        extractbutton.config(state=NORMAL)
        kmeansbar.state(["!disabled"])


def singleband(file):
    global displaybandarray,originbandarray,originpcabands
    try:
        bands=Multigraybands[file].bands
    except:
        return
    bandsize=Multigraybands[file].size
    print('bandsize',bandsize)
    try:
        channel,height,width=bands.shape
    except:
        channel=0
    if channel>1:
        bands=bands[0,:,:]
    #bands=cv2.GaussianBlur(bands,(3,3),cv2.BORDER_DEFAULT)
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
    print('ratio',ratio)
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
    fea_l,fea_w=bands.shape
    fea_vector=np.zeros((fea_l*fea_w,10))
    pyplt.imsave('bands.png',bands)
    displaybands=cv2.resize(bands,(int(bandsize[1]/ratio),int(bandsize[0]/ratio)),interpolation=cv2.INTER_LINEAR)
    pyplt.imsave('displaybands.png',displaybands)
    displayfea_l,displayfea_w=displaybands.shape
    displayfea_vector=np.zeros((displayfea_l*displayfea_w,10))
    # originfea_vector=np.zeros((bandsize[0],bandsize[1],10))

    # saveimg=np.copy(bands).astype('uint8')
    # pyplt.imsave('ostuimg.png',saveimg)

    if 'LabOstu' not in originbands:
        originbands.update({'LabOstu':bands})
        fea_bands=bands.reshape(fea_l*fea_w,1)[:,0]
        # originfea_vector[:,9]=originfea_vector[:,0]+fea_bands
        displayfea_bands=displaybands.reshape((displayfea_l*displayfea_w),1)[:,0]
        fea_vector[:,9]=fea_vector[:,0]+fea_bands
        displayfea_vector[:,9]=displayfea_vector[:,0]+displayfea_bands
        #displaybands=displaybands.reshape((int(bandsize[1]/ratio),int(bandsize[0]/ratio),3))
        #kernel=np.ones((2,2),np.float32)/4
        #displaybands=np.copy(bands)
        displays.update({'LabOstu':displaybands})
        #displaybandarray.update({'LabOstu':cv2.filter2D(displaybands,-1,kernel)})
    bands=Multiimagebands[file].bands
    #for i in range(3):
    #    bands[i,:,:]=cv2.GaussianBlur(bands[i,:,:],(3,3),cv2.BORDER_DEFAULT)
    NDI=128*((bands[1,:,:]-bands[0,:,:])/(bands[1,:,:]+bands[0,:,:])+1)
    tempdict={'NDI':NDI}
    # saveimg=np.copy(NDI).astype('uint8')
    # pyplt.imsave('NDIimg.png',saveimg)
    if 'NDI' not in originbands:
        originbands.update(tempdict)

        displaybands=cv2.resize(NDI,(int(bandsize[1]/ratio),int(bandsize[0]/ratio)),interpolation=cv2.INTER_LINEAR)
        fea_bands=NDI.reshape(fea_l*fea_w,1)[:,0]
        # originfea_vector[:,1]=originfea_vector[:,1]+fea_bands
        displayfea_bands=displaybands.reshape((displayfea_l*displayfea_w),1)[:,0]
        fea_vector[:,1]=fea_vector[:,1]+fea_bands
        displayfea_vector[:,1]=displayfea_vector[:,1]+displayfea_bands
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
    # saveimg=np.zeros((bandsize[0],bandsize[1],3),'uint8')
    # saveimg[:,:,0]=np.copy(Red).astype('uint8')
    # pyplt.imsave('Redimg.png',saveimg)
    # saveimg=np.zeros((bandsize[0],bandsize[1],3),'uint8')
    # saveimg[:,:,1]=np.copy(Green).astype('uint8')
    # pyplt.imsave('Greenimg.png',saveimg)
    # saveimg=np.zeros((bandsize[0],bandsize[1],3),'uint8')
    # saveimg[:,:,2]=np.copy(Blue).astype('uint8')
    # pyplt.imsave('Blueimg.png',saveimg)
    if 'Band1' not in originbands:
        originbands.update(tempdict)

        image=cv2.resize(Red,(int(bandsize[1]/ratio),int(bandsize[0]/ratio)),interpolation=cv2.INTER_LINEAR)
        displaydict={'Band1':image}
        displays.update(displaydict)
        fea_bands=Red.reshape(fea_l*fea_w,1)[:,0]
        # originfea_vector[:,2]=originfea_vector[:,2]+fea_bands
        displayfea_bands=image.reshape((displayfea_l*displayfea_w),1)[:,0]
        fea_vector[:,2]=fea_vector[:,2]+fea_bands
        displayfea_vector[:,2]=displayfea_vector[:,2]+displayfea_bands
    tempdict={'Band2':Green}
    if 'Band2' not in originbands:
        originbands.update(tempdict)

        image=cv2.resize(Green,(int(bandsize[1]/ratio),int(bandsize[0]/ratio)),interpolation=cv2.INTER_LINEAR)
        displaydict={'Band2':image}
        displays.update(displaydict)
        fea_bands=Green.reshape(fea_l*fea_w,1)[:,0]
        # originfea_vector[:,3]=originfea_vector[:,3]+fea_bands
        displayfea_bands=image.reshape((displayfea_l*displayfea_w),1)[:,0]
        fea_vector[:,3]=fea_vector[:,3]+fea_bands
        displayfea_vector[:,3]=displayfea_vector[:,3]+displayfea_bands
    tempdict={'Band3':Blue}
    if 'Band3' not in originbands:
        originbands.update(tempdict)
        # originfea_vector[:,4]=originfea_vector[:,4]+Blue
        image=cv2.resize(Blue,(int(bandsize[1]/ratio),int(bandsize[0]/ratio)),interpolation=cv2.INTER_LINEAR)
        displaydict={'Band3':image}
        displays.update(displaydict)
        fea_bands=Blue.reshape(fea_l*fea_w,1)[:,0]
        displayfea_bands=image.reshape((displayfea_l*displayfea_w),1)[:,0]
        fea_vector[:,4]=fea_vector[:,4]+fea_bands
        displayfea_vector[:,4]=displayfea_vector[:,4]+displayfea_bands
    Greenness = bands[1, :, :] / (bands[0, :, :] + bands[1, :, :] + bands[2, :, :])
    tempdict = {'Greenness': Greenness}
    if 'Greenness' not in originbands:
        originbands.update(tempdict)
        # originfea_vector[:,5]=originfea_vector[:,5]+Greenness
        image=cv2.resize(Greenness,(int(bandsize[1]/ratio),int(bandsize[0]/ratio)),interpolation=cv2.INTER_LINEAR)
        #image=image.reshape((int(bandsize[1]/ratio),int(bandsize[0]/ratio),3))
        displaydict={'Greenness':image}
        #displaybandarray.update(worktempdict)
        displays.update(displaydict)
        fea_bands=Greenness.reshape(fea_l*fea_w,1)[:,0]
        displayfea_bands=image.reshape((displayfea_l*displayfea_w),1)[:,0]
        fea_vector[:,5]=fea_vector[:,5]+fea_bands
        displayfea_vector[:,5]=displayfea_vector[:,5]+displayfea_bands
    VEG=bands[1,:,:]/(np.power(bands[0,:,:],0.667)*np.power(bands[2,:,:],(1-0.667)))
    tempdict={'VEG':VEG}
    if 'VEG' not in originbands:
        originbands.update(tempdict)
        # originfea_vector[:,6]=originfea_vector[:,6]+VEG
        image=cv2.resize(VEG,(int(bandsize[1]/ratio),int(bandsize[0]/ratio)),interpolation=cv2.INTER_LINEAR)
        kernel=np.ones((4,4),np.float32)/16
        #displaybandarray.update({'LabOstu':})
        #image=image.reshape((int(bandsize[1]/ratio),int(bandsize[0]/ratio),3))
        worktempdict={'VEG':cv2.filter2D(image,-1,kernel)}
        displays.update(worktempdict)
        fea_bands=VEG.reshape(fea_l*fea_w,1)[:,0]
        displayfea_bands=image.reshape((displayfea_l*displayfea_w),1)[:,0]
        fea_vector[:,6]=fea_vector[:,6]+fea_bands
        displayfea_vector[:,6]=displayfea_vector[:,6]+displayfea_bands
    CIVE=0.441*bands[0,:,:]-0.811*bands[1,:,:]+0.385*bands[2,:,:]+18.78745
    tempdict={'CIVE':CIVE}
    if 'CIVE' not in originbands:
        originbands.update(tempdict)
        # originfea_vector[:,7]=originfea_vector[:,7]+CIVE
        image=cv2.resize(CIVE,(int(bandsize[1]/ratio),int(bandsize[0]/ratio)),interpolation=cv2.INTER_LINEAR)
        #image=image.reshape((int(bandsize[1]/ratio),int(bandsize[0]/ratio),3))
        worktempdict={'CIVE':image}
        displays.update(worktempdict)
        fea_bands=CIVE.reshape(fea_l*fea_w,1)[:,0]
        displayfea_bands=image.reshape((displayfea_l*displayfea_w),1)[:,0]
        fea_vector[:,7]=fea_vector[:,7]+fea_bands
        displayfea_vector[:,7]=displayfea_vector[:,7]+displayfea_bands
    MExG=1.262*bands[1,:,:]-0.884*bands[0,:,:]-0.311*bands[2,:,:]
    tempdict={'MExG':MExG}
    if 'MExG' not in originbands:
        originbands.update(tempdict)
        # originfea_vector[:,8]=originfea_vector[:,8]+MExG
        image=cv2.resize(MExG,(int(bandsize[1]/ratio),int(bandsize[0]/ratio)),interpolation=cv2.INTER_LINEAR)
        #image=image.reshape((int(bandsize[1]/ratio),int(bandsize[0]/ratio),3))
        worktempdict={'MExG':image}
        displays.update(worktempdict)
        fea_bands=MExG.reshape(fea_l*fea_w,1)[:,0]
        displayfea_bands=image.reshape((displayfea_l*displayfea_w),1)[:,0]
        fea_vector[:,8]=fea_vector[:,8]+fea_bands
        displayfea_vector[:,8]=displayfea_vector[:,8]+displayfea_bands
    NDVI=(bands[0,:,:]-bands[2,:,:])/(bands[0,:,:]+bands[2,:,:])
    tempdict={'NDVI':NDVI}
    if 'NDVI' not in originbands:
        originbands.update(tempdict)
        # originfea_vector[:,0]=originfea_vector[:,9]+NDVI
        image=cv2.resize(NDVI,(int(bandsize[1]/ratio),int(bandsize[0]/ratio)),interpolation=cv2.INTER_LINEAR)
        #image=image.reshape((int(bandsize[1]/ratio),int(bandsize[0]/ratio),3))
        worktempdict={'NDVI':image}
        displays.update(worktempdict)
        fea_bands=NDVI.reshape(fea_l*fea_w,1)[:,0]
        displayfea_bands=image.reshape((displayfea_l*displayfea_w),1)[:,0]
        fea_vector[:,0]=fea_vector[:,9]+fea_bands
        displayfea_vector[:,0]=displayfea_vector[:,9]+displayfea_bands
    NGRDI=(bands[1,:,:]-bands[0,:,:])/(bands[1,:,:]+bands[0,:,:])
    tempdict={'NGRDI':NGRDI}
    if 'NGRDI' not in originbands:
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
    '''
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
    '''

    M=np.mean(displayfea_vector.T,axis=1)
    OM=np.mean(fea_vector.T,axis=1)
    print('M',M,'M shape',M.shape, 'OM',OM,'OM Shape',OM.shape)
    C=displayfea_vector-M
    OC=fea_vector-OM
    #max=np.max(C.T,axis=1)
    #print('MAX',max)
    #C=C/max
    print('C',C,'OC',OC)
    #V=np.cov(C.T)
    V=np.corrcoef(C.T)
    OV=np.corrcoef(OC.T)
    std=np.std(displayfea_vector.T,axis=1)
    O_std=np.std(fea_vector.T,axis=1)
    print(std,O_std)
    std_displayfea=C/std
    O_stddisplayfea=OC/O_std
    print(std_displayfea,O_stddisplayfea)
    #eigvalues,eigvectors=np.linalg.eig(V)
    #n,m=displayfea_vector.shape
    #C=np.dot(displayfea_vector.T,displayfea_vector)/(n-1)
    V_var=np.cov(std_displayfea.T)
    print('COV',V_var)
    print('COR',V)
    eigvalues=la.eigvals(V_var)
    #eigvalues=np.linalg.eigvals(C)
    print('eigvalue',eigvalues)
    idx=np.argsort(eigvalues)
    print('idx',idx)
    eigvalues,eigvectors=np.linalg.eig(V)
    print('eigvalue',eigvalues)
    print('eigvectors',eigvectors)
    eigvalueperc={}
    featurechannel=0
    for i in range(len(eigvalues)):
        print('percentage',i,eigvalues[i]/sum(eigvalues))
        eigvalueperc.update({i:eigvalues[i]/sum(eigvalues)})
        #if eigvalues[i]>0:
        featurechannel+=1
    o_eigenvalue,o_eigenvector=np.linalg.eig(OV)
    pcabands=np.zeros((displayfea_vector.shape[0],featurechannel))
    o_pcabands=np.zeros((fea_vector.shape[0],featurechannel))
    pcavar={}
    for i in range(featurechannel):
        pcn=eigvectors[:,i]
        opcn=o_eigenvector[:,i]
        #pcnbands=np.dot(displayfea_vector,pcn)
        pcnbands=np.dot(std_displayfea,pcn)
        opcnbands=np.dot(O_stddisplayfea,opcn)
        pcvar=np.var(pcnbands)
        print('pc',i+1,' var=',pcvar)
        temppcavar={i:pcvar}
        pcavar.update(temppcavar)
        pcnbands=np.dot(C,pcn)
        opcnbands=np.dot(OC,opcn)
        pcabands[:,i]=pcabands[:,i]+pcnbands
        o_pcabands[:,i]=o_pcabands[:,i]+opcnbands
    # sortvar=sorted(pcavar,key=pcavar.get)
    # print(sortvar)
    # for i in range(len(sortvar)):
    #     pcn=eigvectors[:,sortvar[i]]
    #     pcnbands=np.dot(displayfea_vector,pcn)
    #     pcabands[:,i]=pcabands[:,i]+pcnbands
    #np.savetxt('pcs.csv',pcabands,delimiter=',',fmt='%s')
    #np.savetxt('color-index.csv',displayfea_vector,delimiter=',',fmt='%s')
    #high,width=pcabands.shape
    #fp=open('pcs.csv',w)
    #fc=open('color-index.csv',w)
    #head=['Otsu','NDI','R','G','B','Greenness','VEG','CIVE','MExG','NDVI']
    #for i in range(high):

    #threedplot(pcabands)
    originpcabands.update({file:o_pcabands})
    pcabandsdisplay=pcabands.reshape(displayfea_l,displayfea_w,featurechannel)
    #originbands={'LabOstu':pcabandsdisplay}
    tempdictdisplay={'LabOstu':pcabandsdisplay}
    #displaybandarray.update({file:displays})
    displaybandarray.update({file:tempdictdisplay})
    originbandarray.update({file:originbands})

from mpl_toolkits.mplot3d import Axes3D
def threedplot(area):
    fig=pyplt.figure()
    ax=fig.add_subplot(111,projection='3d')
    n=100
    xs=np.copy(area[0:n,0])
    ys=np.copy(area[0:n,1])
    zs=np.copy(area[0:n,3])
    colors=("red","green","blue")
    groups=("PC1","PC2","PC3")
    #for c,l in [('r','o'),('g','^')]:

    ax.scatter(xs,ys,np.max(zs),c='r',marker='o')
    ax.scatter(xs,np.min(ys),zs,c='b',marker='^')
    ax.scatter(np.max(xs),ys,zs,c='g')

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')

    pyplt.show()



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


def generatecheckbox(frame,classnum):
    global checkboxdict,havecolorstrip
    for widget in frame.winfo_children():
        widget.pack_forget()
    checkboxdict={}
    havecolorstrip=False
    addcolorstrip()
    for i in range(10):
        dictkey=str(i+1)
        tempdict={dictkey:Variable()}
        tempdict[dictkey].set('0')
        checkboxdict.update(tempdict)
        ch=Checkbutton(checkboxframe,text=dictkey,variable=checkboxdict[dictkey])#,command=partial(changecluster,''))
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
    global currentlabels,changekmeans,colordicesband,originbinaryimg,pre_checkbox
    colordicesband=np.copy(displaylabels)
    keys=checkboxdict.keys()
    plantchoice=[]
    pre_checkbox=[]
    for key in keys:
        plantchoice.append(checkboxdict[key].get())
        pre_checkbox.append(checkboxdict[key].get())
    tempdisplayimg=np.zeros((displaybandarray[currentfilename]['LabOstu'].shape[0],
                             displaybandarray[currentfilename]['LabOstu'].shape[1]))
    colordivimg=np.zeros((displaybandarray[currentfilename]['LabOstu'].shape[0],
                          displaybandarray[currentfilename]['LabOstu'].shape[1]))
    sel_count=plantchoice.count('1')
    if sel_count == int(kmeans.get()):
        tempdisplayimg=tempdisplayimg+1
    else:
        for i in range(int(kmeans.get())):
            tup=plantchoice[i]
            if '1' in tup:
                tempdisplayimg=np.where(displaylabels==i,1,tempdisplayimg)
    # uniquecolor=np.unique(tempdisplayimg)
    # if len(uniquecolor)==1 and uniquecolor[0]==1:
    #     tempdisplayimg=np.copy(displaylabels).astype('float32')
    currentlabels=np.copy(tempdisplayimg)
    originbinaryimg=np.copy(tempdisplayimg)

    tempcolorimg=np.copy(displaylabels).astype('float32')
    ratio=findratio([tempdisplayimg.shape[0],tempdisplayimg.shape[1]],[850,850])
    if tempdisplayimg.shape[0]*tempdisplayimg.shape[1]<850*850:
        tempdisplayimg=cv2.resize(tempdisplayimg,(int(tempdisplayimg.shape[1]*ratio),int(tempdisplayimg.shape[0]*ratio)))
        colordivimg=cv2.resize(tempcolorimg,(int(colordivimg.shape[1]*ratio),int(colordivimg.shape[0]*ratio)))
    else:
        tempdisplayimg=cv2.resize(tempdisplayimg,(int(tempdisplayimg.shape[1]/ratio),int(tempdisplayimg.shape[0]/ratio)))
        colordivimg=cv2.resize(tempcolorimg,(int(colordivimg.shape[1]/ratio),int(colordivimg.shape[0]/ratio)))
    binaryimg=np.zeros((tempdisplayimg.shape[0],tempdisplayimg.shape[1],3))
    kvar=int(kmeans.get())
    locs=np.where(tempdisplayimg==1)
    binaryimg[locs]=[240,228,66]
    colordeimg=np.zeros((colordivimg.shape[0],colordivimg.shape[1],3))
    if kvar==1:
        grayimg=Image.fromarray(colordivimg.astype('uint8'),'L')
        #grayimg.show()
        colordivdict={}
        colordivdict.update({'Size':colordivimg.shape})
        colordivdict.update({'Image':ImageTk.PhotoImage(grayimg)})
        displayimg['Color Deviation']=colordivdict


        binaryimg=np.zeros((tempdisplayimg.shape[0],tempdisplayimg.shape[1],3))
        tempdict={}
        tempdict.update({'Size':tempdisplayimg.shape})
        tempdict.update({'Image':ImageTk.PhotoImage(Image.fromarray(binaryimg.astype('uint8')))})
        displayimg['ColorIndices']=tempdict

        # changedisplayimg(imageframe,'Color Deviation')
    else:
        for i in range(kvar):
            locs=np.where(colordivimg==i)
            colordeimg[locs]=colorbandtable[i]
        #pyplt.imsave('displayimg.png',tempdisplayimg)
        #pyplt.imsave('allcolorindex.png',colordivimg)
        #bands=Image.fromarray(tempdisplayimg)
        #bands=bands.convert('L')
        #bands.save('displayimg.png')
        #indimg=cv2.imread('displayimg.png')
        Image.fromarray(colordeimg.astype('uint8')).save('allcolorindex.png',"PNG")
        tempdict={}
        tempdict.update({'Size':tempdisplayimg.shape})
        tempdict.update({'Image':ImageTk.PhotoImage(Image.fromarray(binaryimg.astype('uint8')))})
        displayimg['ColorIndices']=tempdict

        #indimg=cv2.imread('allcolorindex.png')
        #tempdict.update({'Image':ImageTk.PhotoImage(Image.fromarray(indimg))})
        #
        colorimg=cv2.imread('allcolorindex.png')
        Image.fromarray((binaryimg.astype('uint8'))).save('binaryimg.png',"PNG")
        colordivdict={}
        colordivdict.update({'Size':tempdisplayimg.shape})
        colordivdict.update({'Image':ImageTk.PhotoImage(Image.fromarray(colordeimg.astype('uint8')))})
        displayimg['Color Deviation']=colordivdict

        # changedisplayimg(imageframe,'ColorIndices')
    # print('sel count',sel_count)
    if sel_count==0:
        changedisplayimg(imageframe,'Color Deviation')
    else:
        changedisplayimg(imageframe,'ColorIndices')
    changekmeans=True


#def kmeansclassify(choicelist,reshapedtif):
def kmeansclassify():
    global clusterdisplay
        #,minipixelareaclass
    if int(kmeans.get())==0:
        return
    #for i in range(len(choicelist)):
    #    tempband=displaybandarray[currentfilename][choicelist[i]]
        #tempband=cv2.resize(tempband,(450,450),interpolation=cv2.INTER_LINEAR)
    #    reshapedtif[:,i]=tempband.reshape(tempband.shape[0]*tempband.shape[1],2)[:,0]
    #if len(choicelist)==0:
    originpcabands=displaybandarray[currentfilename]['LabOstu']
    pcah,pcaw,pcac=originpcabands.shape
    pcacount={}
    keys=list(pcaboxdict.keys())
    for item in keys:
        if pcaboxdict[item].get()=='1':
            pcacount.update({item:pcaboxdict[item]})
    pcakeys=list(pcacount.keys())
    tempband=np.zeros((pcah,pcaw,len(pcakeys)))
    for i in range(len(pcakeys)):
        channel=int(pcakeys[i])-1
        tempband[:,:,i]=tempband[:,:,i]+originpcabands[:,:,channel]
    if int(kmeans.get())==1:
        print('kmeans=1')
        displaylabels=np.mean(tempband,axis=2)
        pyplt.imsave('k=1.png',displaylabels)
    else:

    #tempband=displaybandarray[currentfilename]['LabOstu']
        if int(kmeans.get())>1:
            h,w,c=tempband.shape
            print('shape',tempband.shape)
            reshapedtif=tempband.reshape(tempband.shape[0]*tempband.shape[1],c)
            print('reshape',reshapedtif.shape)
            clf=KMeans(n_clusters=int(kmeans.get()),init='k-means++',n_init=10,random_state=0)
            tempdisplayimg=clf.fit(reshapedtif)
            # print('label=0',np.any(tempdisplayimg==0))
            displaylabels=tempdisplayimg.labels_.reshape((displaybandarray[currentfilename]['LabOstu'].shape[0],
                                                  displaybandarray[currentfilename]['LabOstu'].shape[1]))
        clusterdict={}
        displaylabels=displaylabels+10
        for i in range(int(kmeans.get())):
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


    # pixelarea=1.0
    # for i in range(int(kmeans.get())):
    #     pixelloc=np.where(displaylabels==i)
    #     pixelnum=len(pixelloc[0])
    #     temparea=float(pixelnum/(displaylabels.shape[0]*displaylabels.shape[1]))
    #     if temparea<pixelarea:
    #         #minipixelareaclass=i
    #         pixelarea=temparea
    if kmeans.get() not in clusterdisplay:
        tempdict={kmeans.get():displaylabels}
        #clusterdisplay.update({''.join(choicelist):tempdict})
        clusterdisplay.update(tempdict)
    return displaylabels

def addcolorstrip():
    global kmeanscanvasframe,havecolorstrip
    if havecolorstrip is False:
        colornum=int(kmeans.get())
        for widget in kmeanscanvasframe.winfo_children():
            widget.pack_forget()
        widget.delete(ALL)
        widget.config(width=350,height=10)
        widget.create_image(3,0,image=colorstripdict['colorstrip'+str(colornum)],anchor=NW)
        widget.pack()
        havecolorstrip=True

def getPCs():
    global displayimg
    originpcabands=displaybandarray[currentfilename]['LabOstu']
    pcah,pcaw,pcac=originpcabands.shape
    pcacount={}
    keys=list(pcaboxdict.keys())
    for item in keys:
        if pcaboxdict[item].get()=='1':
            pcacount.update({item:pcaboxdict[item]})
    pcakeys=list(pcacount.keys())
    tempband=np.zeros((pcah,pcaw,len(pcakeys)))
    for i in range(len(pcakeys)):
        channel=int(pcakeys[i])-1
        tempband[:,:,i]=tempband[:,:,i]+originpcabands[:,:,channel]
    # if int(kmeans.get())==1:
    print('kmeans=1')
    displaylabels=np.mean(tempband,axis=2)
    pyplt.imsave('k=1.png',displaylabels)
    ratio=findratio([originpcabands.shape[0],originpcabands.shape[1]],[850,850])
    tempcolorimg=np.copy(displaylabels)
    colordivimg=np.zeros((displaylabels.shape[0],
                          displaylabels.shape[1]))
    if originpcabands.shape[0]*originpcabands.shape[1]<850*850:
        # tempdisplayimg=cv2.resize(originpcabands,(int(originpcabands.shape[1]*ratio),int(originpcabands.shape[0]*ratio)))
        colordivimg=cv2.resize(tempcolorimg,(int(colordivimg.shape[1]*ratio),int(colordivimg.shape[0]*ratio)))
    else:
        # tempdisplayimg=cv2.resize(originpcabands,(int(originpcabands.shape[1]/ratio),int(originpcabands.shape[0]/ratio)))
        colordivimg=cv2.resize(tempcolorimg,(int(colordivimg.shape[1]/ratio),int(colordivimg.shape[0]/ratio)))
    if colordivimg.min()<0:
        colordivimg=colordivimg-colordivimg.min()
    grayimg=Image.fromarray(colordivimg.astype('uint8'),'L')
    # grayimg=Image.fromarray(np.uint8(colordivimg*255),'L')
    displayimg['PCs']['Image']=ImageTk.PhotoImage(grayimg)

def changepca(event):
    global clusterdisplay,colordicesband,oldpcachoice

    if len(oldpcachoice)>0:
        keys=pcaboxdict.keys()
        newlist=[]
        for key in keys:
            newlist.append(pcaboxdict[key].get())
        samecount=0
        print('oldlist',oldpcachoice)
        print('newlist',newlist)
        for i in range(len(oldpcachoice)):
            if oldpcachoice[i]==newlist[i]:
                samecount+=1
        if samecount==len(oldpcachoice):
            return
    getPCs()
    clusterdisplay={}
    keys=pcaboxdict.keys()
    oldpcachoice=[]
    for key in keys:
        oldpcachoice.append(pcaboxdict[key].get())
    displaylabels=kmeansclassify()
    colordicesband=np.copy(displaylabels)
    generateimgplant(displaylabels)
    return

def savePCAimg(path,originfile,file):
    originpcabands=displaybandarray[currentfilename]['LabOstu']
    pcah,pcaw,pcac=originpcabands.shape
    pcacount={}
    keys=list(pcaboxdict.keys())
    for item in keys:
        if pcaboxdict[item].get()=='1':
            pcacount.update({item:pcaboxdict[item]})
    pcakeys=list(pcacount.keys())
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
    originheight,originwidth=Multigraybands[file].size
    origingray=grayimg.resize([originwidth,originheight],resample=Image.BILINEAR)
    origingray.save(path+'/'+originfile+'-PCAimg.png',"PNG")
    # addcolorstrip()
    return


def changecluster(event):
    global havecolorstrip,pre_checkbox
    imageband=np.copy(displaybandarray[currentfilename]['LabOstu'])
    if int(kmeans.get())==1:
        originpcabands=displaybandarray[currentfilename]['LabOstu']
        pcah,pcaw,pcac=originpcabands.shape
        pcacount={}
        keys=list(pcaboxdict.keys())
        for item in keys:
            if pcaboxdict[item].get()=='1':
                pcacount.update({item:pcaboxdict[item]})
        pcakeys=list(pcacount.keys())
        tempband=np.zeros((pcah,pcaw,len(pcakeys)))
        for i in range(len(pcakeys)):
            channel=int(pcakeys[i])-1
            tempband[:,:,i]=tempband[:,:,i]+originpcabands[:,:,channel]
        displaylabels=np.mean(tempband,axis=2)
        generateimgplant(displaylabels)
        # grayimg=(((displaylabels-displaylabels.min())/(displaylabels.max()-displaylabels.min()))*255.9).astype(np.uint8)
        # pyplt.imsave('k=1.png',displaylabels.astype('uint8'))
        # pyplt.imsave('k=1.png',grayimg)
        grayimg=Image.fromarray(displaylabels.astype('uint8'),'L')
        print('max',displaylabels.max())
        print('min',displaylabels.min())
        grayimg.save('k=1.png',"PNG")
        # originheight,originwidth=Multigraybands[filenames[0]].size
        # origingray=grayimg.resize([originwidth,originheight],resample=Image.BILINEAR)
        # origingray.save('PCAimg.png',"PNG")
        addcolorstrip()
        return
    else:
        if kmeans.get() in clusterdisplay:
            displaylabels=clusterdisplay[kmeans.get()]
            if len(pre_checkbox)>0:
                keys=checkboxdict.keys()
                plantchoice=[]
                for key in keys:
                    plantchoice.append(checkboxdict[key].get())
                allsame=True
                for i in range(len(pre_checkbox)):
                    if pre_checkbox[i]!=plantchoice[i]:
                        allsame=False
                if allsame==True:
                    print('allsame=true')
                    return
        else:
            havecolorstrip=False
            choicelist=[]
            #reshapemodified_tif=np.zeros((displaybandarray[currentfilename]['LabOstu'].shape[0]*displaybandarray[currentfilename]['LabOstu'].shape[1],len(choicelist)))
            #displaylabels=kmeansclassify(choicelist,reshapemodified_tif)
            displaylabels=kmeansclassify()
        generateimgplant(displaylabels)
        # pyplt.imsave('allcolorindex.png',displaylabels)
        #kmeanscanvas.update()
        addcolorstrip()
        return

def showcounting(tup,number=True,frame=True,header=True,whext=False,blkext=False):
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
    imgrsc=cv2.imread(filename,flags=cv2.IMREAD_ANYCOLOR)
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


    #print('showcounting img',image.size)
    #image.save('beforeresize.gif',append_images=[image])
    #image=image.resize([labels.shape[1],labels.shape[0]],resample=Image.BILINEAR)
    print('showcounting_resize',image.size)
    image.save('beforlabel.gif',append_images=[image])
    draw=ImageDraw.Draw(image)
    #font=ImageFont.load_default()
    sizeuniq,sizecounts=np.unique(labels,return_counts=True)
    minsize=min(sizecounts)
    suggsize=int(minsize**0.5)
    if suggsize>22:
        suggsize=22
    if suggsize<14:
        suggsize=14
    #suggsize=8
    #print('fontsize',suggsize)
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
            if frame==True:
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


    if header==True:
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
    height,width,channel=displaybandarray[filename]['LabOstu'].shape
    ratio=findratio([height,width],[850,850])
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

def export_ext(iterver,path,whext=False,blkext=False):
    suggsize=8
    print('fontsize',suggsize)
    smallfont=ImageFont.truetype('cmb10.ttf',size=suggsize)
    files=multi_results.keys()
    # path=filedialog.askdirectory()
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
            tup=(labels,counts,colortable,[],currentfilename)
            _band,segimg,small_segimg=showcounting(tup,False,True,True,whext,blkext)
            #imageband=outputimgbands[file][itervalue]
            imageband=segimg
            draw=ImageDraw.Draw(imageband)
            uniquelabels=list(colortable.keys())
            tempdict={}
            if refarea is not None:
                specarea=float(sizeentry.get())
                pixelmmratio=(specarea/len(refarea[0]))**0.5
            else:
                pixelmmratio=1.0
            #print('coinsize',coinsize.get(),'pixelmmratio',pixelmmratio)
            print('pixelmmratio',pixelmmratio)
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
                        if imgtypevar.get()=='0':
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
                    if imgtypevar.get()=='0':
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
            originheight,originwidth=Multigraybands[file].size
            image=imageband.resize([originwidth,originheight],resample=Image.BILINEAR)
            extcolor=""
            if whext==True:
                extcolor= "-extwht"
            if blkext==True:
                extcolor="-extblk"
            image.save(path+'/'+originfile+extcolor+'-sizeresult'+'.png',"PNG")
            tup=(labels,counts,colortable,[],currentfilename)
            _band,segimg,small_segimg=showcounting(tup,False,True,True,whext,blkext)
            segimage=segimg.resize([originwidth,originheight],resample=Image.BILINEAR)
            segimage.save(path+'/'+originfile+extcolor+'-segmentresult'+'.png',"PNG")
            _band,segimg,small_segimg=showcounting(tup,True,True,True,whext,blkext)
            segimage=segimg.resize([originwidth,originheight],resample=Image.BILINEAR)
            segimage.save(path+'/'+originfile+extcolor+'-labelresult'+'.png',"PNG")

def export_result(iterver):
    global batch
    if proc_mode[proc_name].get()=='1':
        batchprocess.batch_exportpath()
        return
    suggsize=8
    print('fontsize',suggsize)
    smallfont=ImageFont.truetype('cmb10.ttf',size=suggsize)
    files=multi_results.keys()
    path=filedialog.askdirectory()
    export_ext(iterver,path,True,False)
    export_ext(iterver,path,False,True)
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
            tup=(labels,counts,colortable,[],currentfilename)
            _band,segimg,small_segimg=showcounting(tup,False)
            #imageband=outputimgbands[file][itervalue]
            imageband=segimg
            draw=ImageDraw.Draw(imageband)
            uniquelabels=list(colortable.keys())
            tempdict={}
            if refarea is not None:
                specarea=float(sizeentry.get())
                pixelmmratio=(specarea/len(refarea[0]))**0.5
            else:
                pixelmmratio=1.0
            #print('coinsize',coinsize.get(),'pixelmmratio',pixelmmratio)
            print('pixelmmratio',pixelmmratio)
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
                        if imgtypevar.get()=='0':
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
                    if imgtypevar.get()=='0':
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
            originheight,originwidth=Multigraybands[file].size
            image=imageband.resize([originwidth,originheight],resample=Image.BILINEAR)
            image.save(path+'/'+originfile+'-sizeresult'+'.png',"PNG")
            tup=(labels,counts,colortable,[],currentfilename)
            _band,segimg,small_segimg=showcounting(tup,False)
            segimage=segimg.resize([originwidth,originheight],resample=Image.BILINEAR)
            segimage.save(path+'/'+originfile+'-segmentresult'+'.png',"PNG")
            _band,segimg,small_segimg=showcounting(tup,True)
            segimage=segimg.resize([originwidth,originheight],resample=Image.BILINEAR)
            segimage.save(path+'/'+originfile+'-labelresult'+'.png',"PNG")
            originrestoredband=np.copy(labels)
            restoredband=originrestoredband.astype('uint8')
            colordiv=np.zeros((colordicesband.shape[0],colordicesband.shape[1],3))
            savePCAimg(path,originfile,file)

            # kvar=int(kmeans.get())
            # print('kvar',kvar)
            # for i in range(kvar):
            #     locs=np.where(colordicesband==i)
            #     colordiv[locs]=colorbandtable[i]
            # colordivimg=Image.fromarray(colordiv.astype('uint8'))
            # colordivimg.save(path+'/'+originfile+'-colordevice'+'.jpeg',"JPEG")
            colordivimg=Image.open('allcolorindex.png')
            copycolordiv=colordivimg.resize([originwidth,originheight],resample=Image.BILINEAR)
            copycolordiv.save(path+'/'+originfile+'-colordevice'+'.png',"PNG")
            #pyplt.imsave(path+'/'+originfile+'-colordevice'+'.png',colordiv.astype('uint8'))
            # copybinary=np.zeros((originbinaryimg.shape[0],originbinaryimg.shape[1],3),dtype='float')
            # nonzeros=np.where(originbinaryimg==1)
            # copybinary[nonzeros]=[255,255,0]
            # binaryimg=Image.fromarray(copybinary.astype('uint8'))
            binaryimg=Image.open('binaryimg.png')
            copybinaryimg=binaryimg.resize([originwidth,originheight],resample=Image.BILINEAR)
            copybinaryimg.save(path+'/'+originfile+'-binaryimg'+'.png',"PNG")
            # pyplt.imsave(path+'/'+originfile+'-binaryimg'+'.png',originbinaryimg.astype('uint8'))

            #restoredband=cv2.resize(src=restoredband,dsize=(originwidth,originheight),interpolation=cv2.INTER_LINEAR)
            print(restoredband.shape)
            currentsizes=kernersizes[file]
            indicekeys=list(originbandarray[file].keys())
            indeclist=[ 0 for i in range(len(indicekeys)*3)]
            pcalist=[0 for i in range(3)]
            temppcabands=np.zeros((originpcabands[file].shape[0],len(batch['PCs'])))
            for i in range(len(batch['PCs'])):
                temppcabands[:,i]=temppcabands[:,i]+originpcabands[file][:,batch['PCs'][i]-1]
            pcabands=np.mean(temppcabands,axis=1)
            pcabands=pcabands.reshape((originheight,originwidth))
            datatable={}
            origindata={}
            for key in indicekeys:
                data=originbandarray[file][key]
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
    tx=root.winfo_x()
    ty=root.winfo_y()
    top=Toplevel()
    top.attributes("-topmost",True)
    w = 300
    h = 150
    dx=100
    dy=100
    top.geometry("%dx%d+%d+%d" % (w, h, tx + dx, ty + dy))
    top.title('Saved')
    Message(top,text='Results are saved to '+path,padx=20,pady=20).pack()
    okbut=Button(top,text='Okay',command=top.destroy)
    okbut.pack(side=BOTTOM)
    top.after(10000,top.destroy)
    thresholds=[cal_xvalue(linelocs[0]),cal_xvalue(linelocs[1])]
    minthres=min(thresholds)
    maxthres=max(thresholds)
    lwthresholds=[cal_yvalue(linelocs[2]),cal_yvalue(linelocs[3])]
    maxlw=max(lwthresholds)
    minlw=min(lwthresholds)


    batch['Area_max']=[maxthres]
    batch['Area_min']=[minthres]
    batch['shape_max']=[maxlw]
    batch['shape_min']=[minlw]

    print('batch',batch)

    batchfile=path+'/'+originfile+'-batch'+'.txt'
    with open(batchfile,'w') as f:
        for key in batch.keys():
            f.write(key)
            f.write(',')
            for i in range(len(batch[key])):
                f.write(str(batch[key][i]))
                f.write(',')
            f.write('\n')
        f.close()




def resegment():
    global loccanvas,maxx,minx,maxy,miny,linelocs,bins,ybins,reseglabels,figcanvas,refvar,refsubframe,panelA
    global labelplotmap,figdotlist
    global batch
    figcanvas.unbind('<Any-Enter>')
    figcanvas.unbind('<Any-Leave>')
    figcanvas.unbind('<Button-1>')
    figcanvas.unbind('<B1-Motion>')
    #figcanvas.unbind('<Shift-Button-1>')
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
    # if segmentratio>1:
    #     workingimg=cv2.resize(labels,(int(labels.shape[1]/segmentratio),int(labels.shape[0]/segmentratio)),interpolation=cv2.INTER_LINEAR)
    # else:
    #     workingimg=np.copy(labels)

    reseglabels,border,colortable,labeldict=tkintercorestat.resegmentinput(labels,minthres,maxthres,minlw,maxlw)


    # if segmentratio>1:
    #     cache=(np.zeros(labels.shape),{"f":int(segmentratio),"stride":int(segmentratio)})
    #     reseglabels=tkintercorestat.pool_backward(reseglabels,cache)
    #     #labeldict['iter0']['labels']=reseglabels
    multi_results.update({currentfilename:(labeldict,{})})
    iterkeys=list(labeldict.keys())
    iternum=len(iterkeys)
    print(labeldict)
    #iternum=3
    tempimgdict={}
    tempimgbands={}
    tempsmall={}
    for key in labeldict:
        tup=(labeldict[key]['labels'],labeldict[key]['counts'],labeldict[key]['colortable'],{},currentfilename)
        outputdisplay,outputimg,small_seg=showcounting(tup,False,True,False)
        tempimgdict.update({key:outputdisplay})
        tempimgbands.update({key:outputimg})
        tempsmall.update({key:small_seg})
    outputimgdict.update({currentfilename:tempimgdict})
    outputimgbands.update({currentfilename:tempimgbands})
    outputsegbands.update({currentfilename:tempsmall})
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
    lenlist=[]
    widlist=[]
    labelplotmap={}
    templabelplotmap={}
    unitable=[]
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
            lenlist.append(length)
            widlist.append(width)
            data.append(len(pixelloc[0]))
            unitable.append(uni)
            # templabelplotmap.update({(len(pixelloc[0]),length+width):uni})
    residual,area=lm_method.lm_method(lenlist,widlist,data)
    lenwid=list(residual)
    data=list(area)
    for i in range(len(unitable)):
        templabelplotmap.update({(data[i],lenwid[i]):unitable[i]})
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
    #figcanvas.bind('<Shift-Button-1>',item_multiselect)
    if refarea is not None:
        reseglabels[refarea]=65535

    pcasel=[]
    pcakeys=list(pcaboxdict.keys())
    for i in range(len(pcakeys)):
        currvar=pcaboxdict[pcakeys[i]].get()
        if currvar=='1':
            pcasel.append(i+1)
    kchoice=[]
    kchoicekeys=list(checkboxdict.keys())
    for i in range(len(kchoicekeys)):
        currvar=checkboxdict[kchoicekeys[i]].get()
        if currvar=='1':
            kchoice.append(i+1)
    batch['PCs']=pcasel.copy()
    batch['Kmeans']=[int(kmeans.get())]
    batch['Kmeans_sel']=kchoice.copy()
    batch['Area_max']=[maxthres]
    batch['Area_min']=[minthres]
    batch['L+W_max']=[maxlw]
    batch['L+W_min']=[minlw]
    print(batch)


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

def item_multiselect(event):
    print(event.type,'event item_multiselect')
    currx=event.x
    curry=event.y
    print('mul_x',currx,'mul_y',curry)
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
        #if len(dotflash)>0:
        #    for i in range(len(dotflash)):
        #        figcanvas.delete(dotflash.pop(0))
        dotx=plotlist[shortestdistindex][0]
        doty=plotlist[shortestdistindex][1]
        a=figcanvas.create_oval(dotx-1,doty-1,dotx+1,doty+1,width=1,outline='Orange',fill='Orange')
        dotflash.append(a)
    print(labelkey)
    seedfigflash(labelkey,True)

def item_start_drag(event):
    global figcanvas,linelocs,dotflash
    itemType=figcanvas.type(CURRENT)
    print(itemType)
    print(event.type,'event start_drag')
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
        if labelkey in reseglabels:
            seedfigflash(labelkey)

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
    displaysize=outputsegbands[currentfilename]['iter0'].size
    print('reseglabels shape',reseglabels.shape)
    print('displaysize',displaysize)
    forward=0
    if displaysize[0]*displaysize[1]<reseglabels.shape[0]*reseglabels.shape[1]:
        ratio=int(max(reseglabels.shape[0]/displaysize[1],reseglabels.shape[1]/displaysize[0]))
        forward=1
    else:
        ratio=int(max(displaysize[0]/reseglabels.shape[1],displaysize[1]/reseglabels.shape[0]))
        forward=-1
    #tempband=cv2.resize(processlabel.astype('float32'),(int(processlabel.shape[1]/ratio),int(processlabel.shape[0]/ratio)),interpolation=cv2.INTER_LINEAR)
    print(ratio)
    if int(ratio)>1:
        #if processlabel.shape[0]*processlabel.shape[1]<850*850:
        if forward==-1:
            print('pool_backward')
            cache=(np.zeros((processlabel.shape[0]*ratio,processlabel.shape[1]*ratio)),{"f":int(ratio),"stride":int(ratio)})
            convband=tkintercorestat.pool_backward(processlabel,cache)
        else:
            if forward==1:
                print('pool_forward')
                convband,cache=tkintercorestat.pool_forward(processlabel,{"f":int(ratio),"stride":int(ratio)})
    else:
         convband=processlabel
    print('convband shape',convband.shape)


def process():
    # global outputbutton
    if proc_mode[proc_name].get()=='1':
        batchprocess.batch_process()
        # outputbutton.config(state=NORMAL)
        return
    # else:
    #     outputbutton.config(state=DISABLED)

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
    global currentlabels,panelA,reseglabels,refbutton,figcanvas,resegbutton,refvar
    global refsubframe,loccanvas,originlabels,changekmeans,originlabeldict,refarea
    global figdotlist,segmentratio
    global batch
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
    # coin=refvar.get()=='1'
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
    dealpixel=nonzeroratio*currentlabels.shape[0]*currentlabels.shape[1]
    ratio=1
    if nonzeroratio<=0.2:# and nonzeroratio>=0.1:
        ratio=findratio([currentlabels.shape[0],currentlabels.shape[1]],[1600,1600])
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
                segmentratio=findratio([currentlabels.shape[0],currentlabels.shape[1]],[850,850])
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
    if originlabels is None:
        originlabels,border,colortable,originlabeldict=tkintercorestat.init(workingimg,workingimg,'',workingimg,10,coin)
        changekmeans=False
    else:
        if changekmeans==True:
            originlabels,border,colortable,originlabeldict=tkintercorestat.init(workingimg,workingimg,'',workingimg,10,coin)
            changekmeans=False
    # if segmentratio>1:
    #     cache=(np.zeros((currentlabels.shape[0],currentlabels.shape[1])),{"f":int(segmentratio),"stride":int(segmentratio)})
    #     orisize_originlabels=tkintercorestat.pool_backward(originlabels,cache)
    #     #originlabels=orisize_originlabels
    #     originlabeldict['iter0']['labels']=orisize_originlabels
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
    tempsmall={}
    for key in labeldict:
        tup=(labeldict[key]['labels'],labeldict[key]['counts'],labeldict[key]['colortable'],{},currentfilename)
        outputdisplay,outputimg,smallset=showcounting(tup,False,True,False)
        tempimgdict.update({key:outputdisplay})
        tempimgbands.update({key:outputimg})
        tempsmall.update({key:smallset})
    outputimgdict.update({currentfilename:tempimgdict})
    outputimgbands.update({currentfilename:tempimgbands})
    outputsegbands.update({currentfilename:tempsmall})
    #time.sleep(5)
    #tup=(labeldict,coinparts,currentfilename)
    #resscaler=Scale(frame,from_=1,to=iternum,tickinterval=1,length=220,orient=HORIZONTAL,variable=itervar,command=partial(changeoutputimg,currentfilename))
    #resscaler.pack()
    changeoutputimg(currentfilename,'1')
    processlabel=np.copy(reseglabels)
    tempband=np.copy(convband)
    panelA.bind('<Button-1>',lambda event,arg=processlabel:customcoin(event,processlabel,tempband))
    panelA.bind('<Shift-Button-1>',customcoin_multi)
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
    lenlist=[]
    widlist=[]
    figcanvas.delete(ALL)
    labelplotmap={}

    templabelplotmap={}
    unitable=[]
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
            lenlist.append(length)
            widlist.append(width)
            data.append(len(pixelloc[0]))
            unitable.append(uni)
            # templabelplotmap.update({(len(pixelloc[0]),length+width):uni})
    residual,area=lm_method.lm_method(lenlist,widlist,data)
    lenwid=list(residual)
    data=list(area)
    for i in range(len(unitable)):
        templabelplotmap.update({(data[i],lenwid[i]):unitable[i]})
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
    if maxx-minx==0:
        maxx=minx+10
        x_scalefactor=300/10
    if maxy-miny==0:
        maxy=miny+10
        y_scalefactor=250/10
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
    #figcanvas.bind('<Shift-Button-1>',item_multiselect)
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
    pcasel=[]
    pcakeys=list(pcaboxdict.keys())
    for i in range(len(pcakeys)):
        currvar=pcaboxdict[pcakeys[i]].get()
        if currvar=='1':
            pcasel.append(i+1)
    kchoice=[]
    kchoicekeys=list(checkboxdict.keys())
    for i in range(len(kchoicekeys)):
        currvar=checkboxdict[kchoicekeys[i]].get()
        if currvar=='1':
            kchoice.append(i+1)
    batch['PCs']=pcasel.copy()
    batch['Kmeans']=[int(kmeans.get())]
    batch['Kmeans_sel']=kchoice.copy()
    print(batch)

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

def customcoin_multi(event):
    global panelA,multiselectitems
    global coinbox_list,minflash
    global dotflash,figcanvas
    x=event.x
    y=event.y
    tempband=np.copy(convband)
    print(tempband.shape)
    coinlabel=tempband[y,x]
    print('coinlabel',coinlabel,'x',x,'y',y)
    if coinlabel==0:
        return
    else:
        multiselectitems.append(coinlabel)
        coinarea=np.where(tempband==coinlabel)
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
        a=panelA.create_rectangle(ulx,uly,rlx+1,rly+1,outline='yellow')
        # coinbox_list.append(a)
        # plotcoinarea=np.where(reseglabels==coinlabel)
        # ulx,uly=min(plotcoinarea[1]),min(plotcoinarea[0])
        # rlx,rly=max(plotcoinarea[1]),max(plotcoinarea[0])
        # unix=np.unique(plotcoinarea[1]).tolist()
        # uniy=np.unique(plotcoinarea[0]).tolist()
        # if len(unix)==1:
        #     ulx,rlx=unix[0],unix[0]
        # else:
        #     ulx,rlx=min(plotcoinarea[1]),max(plotcoinarea[1])
        # if len(uniy)==1:
        #     uly,rly=uniy[0],uniy[0]
        # else:
        #     uly,rly=min(plotcoinarea[0]),max(plotcoinarea[0])
        # lw=rlx-ulx+rly-uly
        # area=len(plotcoinarea[0])
        # print('lw',lw,'area',area)
        for k,v in labelplotmap:
            templabel=labelplotmap[(k,v)]
            if templabel==reflabel:
                xval=k
                yval=v
                print('lw',yval,'area',xval)
                plotflash(yval,xval,'Orange','Orange')
                break

def customcoin(event,processlabels,tempband):
    global panelA#refarea,
    global coinbox,reflabel,minflash,coinbox_list
    global dotflash,figcanvas
    global multiselectitems
    x=event.x
    y=event.y
    multiselectitems=[]
    if len(minflash)>0:
        for i in range(len(minflash)):
            panelA.delete(minflash.pop(0))
    if len(dotflash)>0:
        for i in range(len(dotflash)):
            figcanvas.delete(dotflash.pop(0))
    if len(coinbox_list)>0:
        for i in range(len(coinbox_list)):
            panelA.delete(coinbox_list.pop(0))
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
        # plotcoinarea=np.where(reseglabels==coinlabel)
        # ulx,uly=min(plotcoinarea[1]),min(plotcoinarea[0])
        # rlx,rly=max(plotcoinarea[1]),max(plotcoinarea[0])
        # unix=np.unique(plotcoinarea[1]).tolist()
        # uniy=np.unique(plotcoinarea[0]).tolist()
        # if len(unix)==1:
        #     ulx,rlx=unix[0],unix[0]
        # else:
        #     ulx,rlx=min(plotcoinarea[1]),max(plotcoinarea[1])
        # if len(uniy)==1:
        #     uly,rly=uniy[0],uniy[0]
        # else:
        #     uly,rly=min(plotcoinarea[0]),max(plotcoinarea[0])
        # lw=rlx-ulx+rly-uly
        # area=len(plotcoinarea[0])
        for k,v in labelplotmap:
            templabel=labelplotmap[(k,v)]
            if templabel==reflabel:
                xval=k
                yval=v
                print('lw',yval,'area',xval)
                plotflash(yval,xval,'Orange','Orange')
                break
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

def plotflash(yval,xval,outlinecolor,fillcolor):
    global dotflash,figcanvas
    # x_scalefactor=300/(maxx-minx)
    # y_scalefactor=250/(maxy-miny)
    # xval=50+(area-minx)*x_scalefactor+50
    # yval=300-(lw-miny)*y_scalefactor+25
    a=figcanvas.create_oval(xval-1,yval-1,xval+1,yval+1,width=1,outline=outlinecolor,fill=fillcolor)
    dotflash.append(a)

def seedfigflash(topkey,multi=False):
    global panelA,coinbox
    global reflabel,minflash,multiselectitems
    tempband=np.copy(convband)
    if len(minflash)>0:
        for i in range(len(minflash)):
            panelA.delete(minflash.pop(0))
    panelA.delete(coinbox)
    if multi==False:
        multiselectitems=[]
    else:
        multiselectitems.append(topkey)
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
    global labelplotmap,multiselectitems,dotflash,minflash,coinbox_list
    processlabel=np.copy(reseglabels)
    refarea=np.where(processlabel==reflabel)
    reseglabels[refarea]=0
    if len(minflash)>0:
        for i in range(len(minflash)):
            panelA.delete(minflash.pop(0))
    if len(dotflash)>0:
        for i in range(len(dotflash)):
            figcanvas.delete(dotflash.pop(0))
    if len(coinbox_list)>0:
        for i in range(len(coinbox_list)):
            panelA.delete(coinbox_list.pop(0))
    if len(multiselectitems)>0:
        for i in range(len(multiselectitems)):
            refarea=np.where(processlabel==multiselectitems[i])
            reseglabels[refarea]=0
    thresholds=[cal_xvalue(linelocs[0]),cal_xvalue(linelocs[1])]
    minthres=min(thresholds)
    maxthres=max(thresholds)
    lwthresholds=[cal_yvalue(linelocs[2]),cal_yvalue(linelocs[3])]
    maxlw=max(lwthresholds)
    minlw=min(lwthresholds)
    unique,counts=np.unique(processlabel,return_counts=True)
    unique=unique[1:]
    counts=counts[1:]
    hist=dict(zip(unique,counts))
    outsizethreshold=[]
    for key in hist:
        if hist[key]>maxthres:
            outsizethreshold.append(key)
        if hist[key]<minthres:
            outsizethreshold.append(key)
    lenlist=[]
    widlist=[]
    data=[]
    for uni in unique:
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
            lenlist.append(length)
            widlist.append(width)
            data.append(len(pixelloc[0]))
    residual,area=lm_method.lm_method(lenlist,widlist,data)
    residual=list(residual)
    for i in range(len(residual)):
        if residual[i]>maxlw:
            outsizethreshold.append(unique[1:][i])
        if residual[i]<minlw:
            outsizethreshold.append(unique[1:][i])
    for i in range(len(outsizethreshold)):
        deletlabel=outsizethreshold[i]
        refarea=np.where(processlabel==deletlabel)
        reseglabels[refarea]=0




    gen_convband()
    panelA.delete(coinbox)
    reseglabels=tkintercorestat.renamelabels(reseglabels)
    newcolortables=tkintercorestat.get_colortable(reseglabels)
    newunique,newcounts=np.unique(reseglabels,return_counts=True)
    tup=(reseglabels,newcounts,newcolortables,{},currentfilename)
    outputdisplay,outputimg,smallset=showcounting(tup,False)
    tempimgdict={}
    tempimgbands={}
    tempsmall={}
    tempimgdict.update({'iter0':outputdisplay})
    tempimgbands.update({'iter0':outputimg})
    tempsmall.update({'iter0':smallset})
    outputimgdict.update({currentfilename:tempimgdict})
    outputimgbands.update({currentfilename:tempimgbands})
    outputsegbands.update({currentfilename:tempsmall})
    changeoutputimg(currentfilename,'1')
    #update plot
    print('done image')
    copyplotmap=labelplotmap.copy()
    for k,v in copyplotmap.items():
        if v==reflabel:
            figindex=figdotlist[k]
            figcanvas.delete(figindex)
    if len(multiselectitems)>0:
        for k,v in copyplotmap.items():
            if v in multiselectitems and v!=reflabel:
                figindex=figdotlist[k]
                figcanvas.delete(figindex)
        if len(dotflash)>0:
            for i in range(len(dotflash)):
                figcanvas.delete(dotflash.pop(0))
    #tup=list(figcanvas.find_all())
    #figcanvas.delete(tup[-1])
    multiselectitems=[]
    if len(outsizethreshold)>0:
        for k,v in copyplotmap.items():
            if v in outsizethreshold and v!=reflabel:
                figindex=figdotlist[k]
                figcanvas.delete(figindex)
    outsizethreshold=[]
    labels=np.copy(reseglabels)
    reseglabels,border,colortable,labeldict=tkintercorestat.resegmentinput(labels,minthres,maxthres,minlw,maxlw)

#     update plot





#def refchoice(refsubframe):
def refchoice():
    #global coinsize,sizeentry,coinbox,panelA,boundaryarea,coindict,convband
    global sizeentry,coinbox,panelA,boundaryarea,coindict,convband
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
# Open_File('seedsample.JPG')
# singleband('seedsample.JPG')
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

proc_name='batch_mode'
proc_mode={proc_name:Variable()}
proc_mode[proc_name].set('0')
proc_but=Checkbutton(buttondisplay,text=proc_name,variable=proc_mode[proc_name])
proc_but.pack(side=LEFT,padx=20,pady=5)


openfilebutton=Button(buttondisplay,text='Image',command=Open_Multifile,cursor='hand2')
openfilebutton.pack(side=LEFT,padx=20,pady=5)
mapbutton=Button(buttondisplay,text='Map',cursor='hand2',command=Open_Map)
mapbutton.pack(side=LEFT,padx=20,pady=5)

disbuttonoption={'Origin':'1','PCs':'5','Color Deviation':'2','ColorIndices':'3','Output':'4'}
buttonname={'Raw':'1','PCs':'5','Clusters':'2','Selected':'3','Output':'4'}
#disbuttonoption={'Origin':'1','ColorIndices':'3','Output':'4'}
for (text,v1),(name,v2) in zip(disbuttonoption.items(),buttonname.items()):
    b=Radiobutton(buttondisplay,text=name,variable=displaybut_var,value=disbuttonoption[text],command=partial(changedisplayimg,imageframe,text))
    b.pack(side=LEFT,padx=20,pady=5)
    b.configure(state=DISABLED)
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

# filedropvar.set(filenames[0])
# changefiledrop=OptionMenu(changefileframe,filedropvar,*filenames,command=partial(changeimage,imageframe))
# changefiledrop.pack()
### ---choose color indices---
# '''
# chframe=LabelFrame(filter_fr,text='Select indicies below',cursor='hand2',bd=0)
# chframe.pack()
# chcanvas=Canvas(chframe,width=200,height=110,scrollregion=(0,0,400,400))
# chcanvas.pack(side=LEFT)
# chscroller=Scrollbar(chframe,orient=VERTICAL)
# chscroller.pack(side=RIGHT,fill=Y,expand=True)
# chcanvas.config(yscrollcommand=chscroller.set)
# chscroller.config(command=chcanvas.yview)
# contentframe=LabelFrame(chcanvas)
# chcanvas.create_window((4,4),window=contentframe,anchor=NW)
# contentframe.bind("<Configure>",lambda event,arg=chcanvas:onFrameConfigure(arg))
#
# for key in cluster:
#     tempdict={key:Variable()}
#     bandchoice.update(tempdict)
#     ch=ttk.Checkbutton(contentframe,text=key,variable=bandchoice[key])#,command=changecluster)#,command=partial(autosetclassnumber,clusternumberentry,bandchoice))
#     if filedropvar.get()=='seedsample.JPG':
#         if key=='LabOstu':
#             ch.invoke()
#     ch.pack(fill=X)
# '''

### ----Class NUM----
kmeansgenframe=LabelFrame(filter_fr,cursor='hand2',bd=0)
kmeansgenframe.pack()
pcaframe=LabelFrame(kmeansgenframe,text='By PCs',cursor='hand2',bd=0)
pcaframe.pack()
kmeanslabel=LabelFrame(kmeansgenframe,text='By Clusters',bd=0)
checkboxframe=LabelFrame(kmeansgenframe,cursor='hand2',bd=0)#,text='Select classes',cursor='hand2')
kmeanslabel.pack()
pcaboxdict={}
for i in range(10):
    dictkey=str(i+1)
    tempdict={dictkey:Variable()}
    if i==0:
        tempdict[dictkey].set('1')
    else:
        tempdict[dictkey].set('0')
    pcaboxdict.update(tempdict)
    ch=Checkbutton(pcaframe,text=dictkey,variable=pcaboxdict[dictkey])#,command=changepca)
    ch.configure(state=DISABLED)
    ch.pack(side=LEFT)
pcaframe.bind('<Leave>',changepca)
keys=pcaboxdict.keys()
oldpcachoice=[]
for key in keys:
    oldpcachoice.append(pcaboxdict[key].get())
kmeans.set(1)
#kmeansbar=Scale(kmeanslabel,from_=1,to=10,tickinterval=1,length=270,showvalue=0,orient=HORIZONTAL,variable=kmeans,command=partial(generatecheckbox,checkboxframe))
kmeansbar=ttk.Scale(kmeanslabel,from_=1,to=10,length=350,orient=HORIZONTAL,variable=kmeans,cursor='hand2',command=partial(generatecheckbox,checkboxframe))
kmeansbar.pack()
kmeansbar.bind('<ButtonRelease-1>',changecluster)
kmeansbar.state(["disabled"])

checkboxframe.pack()
checkboxframe.bind('<Leave>',changecluster)
for i in range(10):
    dictkey=str(i+1)
    tempdict={dictkey:Variable()}
    if i==0:
        tempdict[dictkey].set('1')
    else:
        tempdict[dictkey].set('0')
    checkboxdict.update(tempdict)
    ch=Checkbutton(checkboxframe,text=dictkey,variable=checkboxdict[dictkey],command=partial(changecluster,''))
    if i+1>int(kmeans.get()):
        ch.config(state=DISABLED)
    ch.pack(side=LEFT)

kmeanscanvasframe=LabelFrame(kmeansgenframe,bd='0')
kmeanscanvasframe.pack()
kmeanscanvas=Canvas(kmeanscanvasframe,width=350,height=10,bg='Black')


#reshapemodified_tif=np.zeros((displaybandarray[currentfilename]['LabOstu'].shape[0]*displaybandarray[currentfilename]['LabOstu'].shape[1],3))
#colordicesband=kmeansclassify(['LabOstu'],reshapemodified_tif)
#colordicesband=kmeansclassify([],reshapemodified_tif)

# colordicesband=kmeansclassify()
# generateimgplant(colordicesband)
# changedisplayimg(imageframe,'Origin')
# getPCs()

colorstrip=np.zeros((15,35*2,3),'uint8')
for i in range(2):
    for j in range(0,35):
        colorstrip[:,i*35+j]=colorbandtable[i,:]
#pyplt.imsave('colorstrip.jpeg',colorstrip)
kmeanscanvas.delete(ALL)
#colorimg=cv2.imread('colorstrip.jpeg',flags=cv2.IMREAD_ANYCOLOR)
colorimg=np.copy(colorstrip)
colorimg=ImageTk.PhotoImage(Image.fromarray(colorimg.astype('uint8')))
kmeanscanvas.create_image(0,0,image=colorimg,anchor=NW)
kmeanscanvas.pack()
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
refbutton=Checkbutton(refsubframe,text='Ref',variable=refvar,command=refchoice)
#refbutton.config(state=DISABLED)
refbutton.pack(side=LEFT,padx=20,pady=5)
sizeentry=Entry(refsubframe,width=5)
sizeentry.insert(END,285)
sizeentry.pack(side=LEFT,padx=5)
sizeunit=Label(refsubframe,text='mm^2')
sizeunit.pack(side=LEFT)
delrefbutton=Button(refsubframe,text='Delete',command=del_reflabel)
delrefbutton.pack(side=LEFT)

#delrefbutton.config(state=DISABLED)
#refbutton=Checkbutton(refsubframe,text='Ref',variable=refvar,command=partial(refchoice,refsubframe))

for widget in refsubframe.winfo_children():
    widget.config(state=DISABLED)
#extractbutton=Button(refframe,text='Process',command=partial(extraction))
extractbutton=Button(refframe,text='Segment',command=process)
extractbutton.configure(activebackground='blue',state=DISABLED)
extractbutton.pack(side=LEFT,padx=20,pady=5)
outputbutton=Button(refframe,text='Export',command=partial(export_result,'0'))
outputbutton.pack(side=LEFT,padx=20,pady=5)
outputbutton.config(state=DISABLED)
#resegbutton=Button(extractionframe,text='Re-Segment',command=resegment)
#resegbutton.pack(side=LEFT)
#resegbutton.config(state=DISABLED)
changekmeans=False
colorstripdict={}
for i in range(1,11):
    colorstrip=np.zeros((15,35*i,3),'uint8')
    for j in range(i):
        for k in range(35):
            colorstrip[:,j*35+k]=colorbandtable[j,:]
    #loadimg=cv2.imread('colorstrip'+str(i)+'.png')
    photoimg=ImageTk.PhotoImage(Image.fromarray(colorstrip.astype('uint8')))
    colorstripdict.update({'colorstrip'+str(i):photoimg})
root.mainloop()

