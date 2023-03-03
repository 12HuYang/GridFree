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
#from matplotlib.figure import Figure

import numpy as np
import os
#import time
import csv
import scipy.linalg as la
from functools import partial
#import threading
#import sys

#import kplus
from sklearn.cluster import KMeans
import tkintercorestat
#import tkintercorestat_plot
import tkintercore
import cal_kernelsize
#import histograms
#import createBins
import axistest
#from multiprocessing import Pool
import lm_method
#import batchprocess
import sel_area

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
previewimg={'Color Deviation':None,
            'ColorIndices':None}
#cluster=['LabOstu','NDI'] #,'Greenness','VEG','CIVE','MExG','NDVI','NGRDI','HEIGHT']
#cluster=['LabOstu','NDI','Greenness','VEG','CIVE','MExG','NDVI','NGRDI','HEIGHT','Band1','Band2','Band3']
cluster=['PAT_R','PAT_G','PAT_B',
         'DIF_R','DIF_G','DIF_B',
         'GLD_R','GLD_G','GLD_B',
         'Band1','Band2','Band3']
         # 'ROO_R','ROO_G','ROO_B',

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
colorindicearray={}
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


batch={'PCweight':[],
       'PCsel':[],
       'Kmeans':[],
       'Kmeans_sel':[],
       'Area_max':[],
       'Area_min':[],
       'shape_max':[],
       'shape_min':[],
       'nonzero':[]}



root=Tk()
root.title('GridFree v.1.1.0 ')
root.geometry("")
root.option_add('*tearoff',False)
emptymenu=Menu(root)
root.config(menu=emptymenu)
screenheight=root.winfo_screenheight()
screenwidth=root.winfo_screenwidth()
print('screenheight',screenheight,'screenwidth',screenwidth)
screenstd=min(screenheight-100,screenwidth-100,850)

coinsize=StringVar()
selarea=StringVar()
refvar=StringVar()
imgtypevar=StringVar()
edge=StringVar()
kmeans=IntVar()
pc_combine_up=DoubleVar()
pc_combine_down=IntVar()
filedropvar=StringVar()
displaybut_var=StringVar()
buttonvar=IntVar()
bandchoice={}
checkboxdict={}

#minipixelareaclass=0

coinbox=None

currentfilename=''
currentlabels=None
displaylabels=None
workingimg=None
displaypclabels=None

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
kmeanschanged=False
pcweightchanged=False
originbinaryimg=None
clusterchanged=False
# originselarea=False
# binaryselarea=False
# pcselarea=False
zoomoff=False
kmeanspolygon=False
selview=''
selareapos=[]
kmeanselareapose=[]
binaryselareaspose=[]

maxx=0
minx=0
bins=None
loccanvas=None
linelocs=[0,0,0,0]
maxy=0
miny=0

segmentratio=0

zoombox=[]

displayfea_l=0
displayfea_w=0
resizeshape=[]
previewshape=[]

pcbuttons=[]
pcbuttonsgroup=[]

drawpolygon=False
app=''
pcfilter=[]

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
    # if oria*orib>850 * 850:
    if oria*orib>screenstd * screenstd:
        if ratio<2:
            ratio=2
    return ratio

def getkeys(dict):
    return [*dict]

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

def changedisplay_pc(frame):
    for widget in frame.winfo_children():
        widget.pack_forget()
    #widget.configure(image=displayimg[text])
    #widget.image=displayimg[text]
    #widget.pack()
    w=displayimg['PCs']['Size'][1]
    l=displayimg['PCs']['Size'][0]
    widget.config(width=w,height=l)
    widget.create_image(0,0,image=displayimg['PCs']['Image'],anchor=NW)
    widget.pack()
    widget.update()

def pcweightupdate(displayframe):
    getPCs()
    changedisplay_pc(displayframe)


def buttonpress(val,displayframe,buttonframe):
    global buttonvar,pc_combine_up,kmeans
    buttonvar.set(val)
    kmeans.set(1)
    pc_combine_up.set(0.5)
    buttonchildren=buttonframe.winfo_children()
    for child in buttonchildren:
        child.config(highlightbackground='white')
    print(buttonchildren[val])
    buttonchild=buttonchildren[val]
    buttonchild.config(highlightbackground='red')
    print('press button ',buttonvar.get())
    getPCs()
    changedisplay_pc(displayframe)
    # if kmeans.get()>1:
    changekmeansbar('')
    beforecluster('')
        # changecluster('')

def PCbuttons(frame,displayframe):
    #display pc buttons
    # buttonvar=IntVar()
    #buttonvar.set(0)
    for widget in frame.winfo_children():
        widget.pack_forget()
    buttonframe=LabelFrame(frame)
    buttonframe.pack()
    for i in range(len(pcbuttons)):
        butimg=pcbuttons[i]
        but=Button(buttonframe,text='',image=butimg,compound=TOP,command=partial(buttonpress,i,displayframe,buttonframe))
        if i==buttonvar.get():
            but.config(highlightbackground='red')
        row=int(i/3)
        col=i%3
        # print(row,col)
        but.grid(row=int(i/3),column=col)
    print('default button',buttonvar.get())
    # change cluster,display






def displaypreview(text):
    global figcanvas,resviewframe
    for widget in resviewframe.winfo_children():
        widget.pack_forget()
    # previewframe=Canvas(frame,width=450,height=400,bg='white')
    figcanvas.pack()
    figcanvas.delete(ALL)
    if text=='Color Deviation':
        previewtext='ColorIndices'
    if text=='ColorIndices':
        previewtext='Color Deviation'

    previewimage=previewimg[previewtext]['Image']

    figcanvas.create_image(0,0,image=previewimage,anchor=NW)
    figcanvas.update()

def switchevent_shift(event,widget):
    global zoomoff,zoomfnid_m,zoomfnid_l,zoombox
    global drawpolygon,app,rects
    drawpolygon=not drawpolygon
    app = sel_area.Application(widget)
    app.end(rects)
    rects=app.start(selview,0,0,drawpolygon)
    print("Drawpolygon to :",drawpolygon)


    # zoomoff = True
    # if zoomoff==True:
    #     widget.unbind('<Motion>',zoomfnid_m)
    #     widget.unbind('<Leave>',zoomfnid_l)
    #     if len(zoombox)>0:
    #         for i in range(len(zoombox)):
    #             widget.delete(zoombox.pop(0))
    #     widget.update()
    # else:
    #     zoomfnid_m=widget.bind('<Motion>',lambda event,arg=widget:zoom(event,arg,img))
    #     zoomfnid_l=widget.bind('<Leave>',lambda event,arg=widget:deletezoom(event,arg))


def switchevent(event,widget,img):
    global zoomoff,zoomfnid_m,zoomfnid_l,zoombox
    zoomoff= not zoomoff
    if zoomoff==True:
        widget.unbind('<Motion>',zoomfnid_m)
        widget.unbind('<Leave>',zoomfnid_l)
        if len(zoombox)>0:
            for i in range(len(zoombox)):
                widget.delete(zoombox.pop(0))
        widget.update()
    else:
        zoomfnid_m=widget.bind('<Motion>',lambda event,arg=widget:zoom(event,arg,img))
        zoomfnid_l=widget.bind('<Leave>',lambda event,arg=widget:deletezoom(event,arg))

def changedisplayimg(frame,text):
    global displaybut_var,figcanvas,resviewframe,reflabel
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
    global rects,selareapos,app,delapp,delrects,delselarea,selview
    global zoomfnid_m,zoomfnid_l,kmeanselareapose,drawpolygon
    global app
    try:
        selareadim = app.getinfo(rects[1])

        if selareadim != [0, 0, 1, 1]:
            selareapos = selareadim
            selview = app.getselview()
            # drawpolygon = app.getdrawpolygon()
        app.end(rects)

    except:
        pass
    try:
        # widget.unbind('<Shift-Double-Button-1>')
        # widget.unbind('<Double-Button-1>')
        widget.unbind('<Motion>')
    except:
        pass
    app = sel_area.Application(widget)
    # delapp=sel_area.Application(widget)
    if text=='Output':
        try:
            image=outputsegbands[currentfilename]['iter0']
            displayfig()
        except:
            return
        zoomfnid_m=widget.bind('<Motion>',lambda event,arg=widget:zoom(event,arg,image))
        zoomfnid_l=widget.bind('<Leave>',lambda event,arg=widget:deletezoom(event,arg))
        delrects=app.start('Output',zoomfnid_m,zoomfnid_l,False)
        widget.bind('<Double-Button-1>',lambda event,arg=widget:switchevent(event,arg,image))
        print('delrects',delrects)
    else:
        reflabel=0
        print('reflabel=',reflabel)
        try:
            delelareadim=app.getinfo(delrects[1])
            if delelareadim!=[]:
                delselarea=delelareadim
            app.end(rects)
        except:
            pass
        if text=='Origin':
            try:
                image=originsegbands['Origin']
                zoomfnid_m=widget.bind('<Motion>',lambda event,arg=widget:zoom(event,arg,image))
                zoomfnid_l=widget.bind('<Leave>',lambda event,arg=widget:deletezoom(event,arg))

            except:
                return
            widget.bind('<Double-Button-1>',lambda event,arg=widget:switchevent(event,arg,image))
            widget.bind('<Shift-Double-Button-1>', lambda event,arg=widget:switchevent_shift(event,arg))
            for widget in resviewframe.winfo_children():
                widget.pack_forget()
            rects=app.start('Origin')
            selview = app.getselview()
            print(rects)

        if text=='PCs':
            if selview=='Origin' and len(selareapos)>0 and selareapos!=[0,0,1,1]:
                # selareadim=app.getinfo(rects[1])
                # if selareadim!=[0,0,1,1] and selareadim!=[] and selareadim!=selareapos:
                #     selareapos=selareadim
                # if selareapos!=[0,0,1,1] and originselarea==True:
                    #need to redo PCA
                # kmeanselareapose=selareapos.copy()
                npfilter=np.zeros((displayimg['Origin']['Size'][0],displayimg['Origin']['Size'][1]))
                print("npfilter shape:", npfilter.shape)
                filter=Image.fromarray(npfilter)
                draw=ImageDraw.Draw(filter)
                if drawpolygon==False:
                    draw.ellipse(selareapos,fill='red')
                else:
                    draw.polygon(selareapos,fill='red')
                global pcfilter
                pcfilter=selareapos.copy()
                filter=np.array(filter)
                filter=np.divide(filter,np.max(filter))
                filter=cv2.resize(filter,(displayfea_w,displayfea_l),interpolation=cv2.INTER_LINEAR)
                partialsingleband(filter)
            for widget in resviewframe.winfo_children():
                widget.pack_forget()
            PCbuttons(resviewframe,frame)
            # try:
            #     image = displayimg['ColorIndices']['Image']
            #     zoomfnid_m = widget.bind('<Motion>', lambda event, arg=widget: zoom(event, arg, image))
            #     zoomfnid_l = widget.bind('<Leave>', lambda event, arg=widget: deletezoom(event, arg))
            #
            # except:
            #     return
            # widget.bind('<Double-Button-1>', lambda event, arg=widget: switchevent(event, arg, image))
            widget.bind('<Shift-Double-Button-1>', lambda event, arg=widget: switchevent_shift(event, arg))
            rects = app.start('PCs',0,0,drawpolygon)
            selview = app.getselview()
            print(rects)
        # else:
        #     widget.unbind('<Motion>')
        if text=='Color Deviation':
            widget.bind('<Shift-Double-Button-1>', lambda event, arg=widget: switchevent_shift(event, arg))
            rects = app.start('Color Deviation',0,0,drawpolygon)
            selview = app.getselview()
            print(rects)

            #displaypreview
            displaypreview(text)
            pass
        if text=='ColorIndices':
            #displaypreview
            displaypreview(text)

            # widget.bind('<Shift-Double-Button-1>', lambda event,arg=widget:switchevent_shift(event,arg))
            # for widget in resviewframe.winfo_children():
            #     widget.pack_forget()
            # rects=app.start('ColorIndices')
            # selview=app.getselview()
            # print(rects)




    #print('change to '+text)
    #time.sleep(1)

def updateresizeshape(shape,content):
    shape.append(int(content))
    return shape

def generatedisplayimg(filename):  # init display images
    global resizeshape,previewshape
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
        # ratio=findratio([height,width],[850,850])
        ratio=findratio([height,width],[screenstd,screenstd])
        print('displayimg ratio',ratio)
        resizeshape=[]
        # if height*width<850*850:
        if height*width<screenstd*screenstd:
            #resize=cv2.resize(Multiimage[filename],(int(width*ratio),int(height*ratio)),interpolation=cv2.INTER_LINEAR)
            updateresizeshape(resizeshape,width*ratio)
            updateresizeshape(resizeshape,height*ratio)
            # resizeshape.append(width*ratio)
            # resizeshape.append(height*ratio)
            if height>screenstd:
                resizeshape=[]
                ratio=round(height/screenstd)
                updateresizeshape(resizeshape,width*ratio)
                updateresizeshape(resizeshape,height*ratio)
            if width>screenstd:
                resizeshape=[]
                ratio=round(width/screenstd)
                updateresizeshape(resizeshape,width*ratio)
                updateresizeshape(resizeshape,height*ratio)
        else:
            #resize=cv2.resize(Multiimage[filename],(int(width/ratio),int(height/ratio)),interpolation=cv2.INTER_LINEAR)
            updateresizeshape(resizeshape,width/ratio)
            updateresizeshape(resizeshape,height/ratio)


        ratio=findratio([height,width],[400,450])
        previewshape=[]
        if height*width<450*400:
            #resize=cv2.resize(Multiimage[filename],(int(width*ratio),int(height*ratio)),interpolation=cv2.INTER_LINEAR)
            updateresizeshape(previewshape,width*ratio)
            updateresizeshape(previewshape,height*ratio)
            if height>400:
                ratio=round(height/screenstd)
                if ratio!=0:
                    previewshape = []
                    updateresizeshape(previewshape,width/ratio)
                    updateresizeshape(previewshape,height/ratio)
            if width>450:
                ratio=round(width/screenstd)
                if ratio!=0:
                    previewshape = []
                    updateresizeshape(previewshape,width/ratio)
                    updateresizeshape(previewshape,height/ratio)
        else:
            #resize=cv2.resize(Multiimage[filename],(int(width/ratio),int(height/ratio)),interpolation=cv2.INTER_LINEAR)
            updateresizeshape(previewshape,width/ratio)
            updateresizeshape(previewshape,height/ratio)

        resize=cv2.resize(Multiimage[filename],(int(resizeshape[0]),int(resizeshape[1])),interpolation=cv2.INTER_LINEAR)
        originimg=Image.fromarray(resize.astype('uint8'))
        originsegbands.update({'Origin':originimg})

        rgbimg=Image.fromarray(resize.astype('uint8'))
        draw=ImageDraw.Draw(rgbimg)
        suggsize=14
        font=ImageFont.truetype('cmb10.ttf',size=suggsize)
        content='\n File: '+filename
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
        tempimg=np.zeros((screenstd,screenstd))

        tempdict.update({'Size':tempimg.shape})
        tempdict.update({'Image':ImageTk.PhotoImage(Image.fromarray(tempimg.astype('uint8')))})
    displayimg['Origin']=tempdict
    #if height*width<850*850:
    #    resize=cv2.resize(Multigray[filename],(int(width*ratio),int(height*ratio)),interpolation=cv2.INTER_LINEAR)
    #else:
        #resize=cv2.resize(Multigray[filename],(int(width/ratio),int(height/ratio)),interpolation=cv2.INTER_LINEAR)
    tempimg=np.zeros((screenstd,screenstd))
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
    temppreviewdict={}
    temppreviewimg=np.zeros((450,400))

    try:
        tempband=np.zeros((displaybandarray[filename]['LabOstu'][:,:,0].shape))
        # tempband=tempband+displaybandarray[filename]['LabOstu']
    # ratio=findratio([tempband.shape[0],tempband.shape[1]],[850,850])

    #if tempband.shape[0]*tempband.shape[1]<850*850:
    #    tempband=cv2.resize(ratio,(int(tempband.shape[1]*ratio),int(tempband.shape[0]*ratio)),interpolation=cv2.INTER_LINEAR)
    #else:
    #    tempband=cv2.resize(ratio,(int(tempband.shape[1]/ratio),int(tempband.shape[0]/ratio)),interpolation=cv2.INTER_LINEAR)
        tempband=cv2.resize(tempband,(int(resizeshape[0]),int(resizeshape[1])),interpolation=cv2.INTER_LINEAR)
        tempdict.update({'Size':tempband.shape})
        tempdict.update({'Image':ImageTk.PhotoImage(Image.fromarray(tempband[:,:,2].astype('uint8')))})

        temppreview=cv2.resize(tempband,(int(previewshape[0]),int(previewshape[1])),interpolation=cv2.INTER_LINEAR)
        temppreview=Image.fromarray(temppreview.astype('uint8'))
        temppreviewdict.update({'Size':previewshape})
        temppreviewdict.update({'Image':ImageTk.PhotoImage(temppreview)})

    # print('resizeshape',resizeshape)
    #pyplt.imsave('displayimg.png',tempband[:,:,0])
    #indimg=cv2.imread('displayimg.png')
    except:
        tempdict.update({'Size':tempimg.shape})
        tempdict.update({'Image':ImageTk.PhotoImage(Image.fromarray(tempimg.astype('uint8')))})

        temppreviewdict.update({'Size':temppreviewimg.shape})
        temppreviewdict.update({'Image':ImageTk.PhotoImage(Image.fromarray(temppreviewimg.astype('uint8')))})

    displayimg['ColorIndices']=tempdict
    previewimg['ColorIndices']=temppreviewdict
    #resize=cv2.resize(Multigray[filename],(int(resizeshape[0]),int(resizeshape[1])),interpolation=cv2.INTER_LINEAR)
    #grayimg=ImageTk.PhotoImage(Image.fromarray(resize.astype('uint8')))
    #tempdict={}
    #tempdict.update({'Size':resize.shape})
    #tempdict.update({'Image':grayimg})
    tempdict={}
    temppreviewdict={}

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
        colordeviate=cv2.resize(colordeviate,(int(resizeshape[0]),int(resizeshape[1])),interpolation=cv2.INTER_LINEAR)
        tempdict.update({'Size':colordeviate.shape})
        tempdict.update({'Image':ImageTk.PhotoImage(Image.fromarray(colordeviate.astype('uint8')))})
    # colortempdict.update({'Size':colordeviate.shape})
    # colortempdict.update({'Image':ImageTk.PhotoImage(Image.fromarray(colordeviate.astype('uint8')))})

    # colortempdict.update({'Image':ImageTk.PhotoImage(testcolor)})

    # tempdict={}
        temppreview=cv2.resize(colordeviate,(int(previewshape[0]),int(previewshape[1])),interpolation=cv2.INTER_LINEAR)
        temppreviewdict.update({'Size':temppreview.shape})
        temppreviewdict.update({'Image':ImageTk.PhotoImage(Image.fromarray(temppreview[:,:,0].astype('uint8')))})
    except:
        tempdict.update({'Size':tempimg.shape})
        tempdict.update({'Image':ImageTk.PhotoImage(Image.fromarray(tempimg.astype('uint8')))})

        temppreviewdict.update({'Size':temppreviewimg.shape})
        temppreviewdict.update({'Image':ImageTk.PhotoImage(Image.fromarray(temppreviewimg.astype('uint8')))})

    # displayimg['Color Deviation']=colortempdict
    displayimg['Color Deviation']=tempdict
    previewimg['Color Deviation']=temppreviewdict


def Open_File(filename):   #add to multi-image,multi-gray  #call band calculation
    global Multiimage,Multigray,Multitype,Multiimagebands,Multigraybands,filenames
    try:
        Filersc=cv2.imread(filename,flags=cv2.IMREAD_ANYCOLOR)
        ndim=np.ndim(Filersc)
        if ndim==2:
            height,width=np.shape(Filersc)
            channel=1
            Filersc.reshape((height,width,channel))
        else:
            height,width,channel=np.shape(Filersc)
        Filesize=(height,width)
        print('filesize:',height,width)
        RGBfile=cv2.cvtColor(Filersc,cv2.COLOR_BGR2RGB)
        Multiimage.update({filename:RGBfile})
        if ndim==2:
            Grayfile=np.copy(Filersc)
        else:
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
            nans=np.isnan(band)
            band[nans]=1e-6
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
        mapdict,mapimage,smallset=showcounting(tup,True,True,True)
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
    global pccombinebar_up
    global displaylabels,displaypclabels
    global buttonvar
    global colorindicearray
    global selarea
    global app,drawpolygon,binaryselareaspose
    MULTIFILES=filedialog.askopenfilenames()
    root.update()
    if len(MULTIFILES)>0:
        Multiimage={}
        Multigray={}
        Multitype={}
        Multiimagebands={}
        Multigraybands={}
        filenames=[]
        originbandarray={}
        colorindicearray={}
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
        displaypclabels=None
        app = ''
        drawpolygon=False
        binaryselareaspose=[]
        buttonvar.set(0)
        # if 'NDI' in bandchoice:
        #     bandchoice['NDI'].set('1')
        # if 'NDVI' in bandchoice:
        #     bandchoice['NDVI'].set('1')
        refbutton.config(state=DISABLED)
        # selareabutton.configure(state=DISABLED)
        selarea.set('0')
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
            # thread=threading.Thread(target=singleband,args=(MULTIFILES[i],))
            singleband(MULTIFILES[i])
            # thread.start()
            # thread.join()
        for widget in changefileframe.winfo_children():
            widget.pack_forget()
        currentfilename=filenames[0]
        # filedropvar.set(filenames[0])
        # changefiledrop=OptionMenu(changefileframe,filedropvar,*filenames,command=partial(changeimage,imageframe))
        # changefiledrop.pack()
        #singleband(filenames[0])
        generatedisplayimg(filenames[0])
        # changedisplayimg(imageframe,'Origin')
        getPCs()

        if len(bandchoice)>0:
            for i in range(len(cluster)):
                bandchoice[cluster[i]].set('')
        #changedisplayimg(imageframe,'Origin')
        kmeans.set(1)
        #reshapemodified_tif=np.zeros((displaybandarray[currentfilename]['LabOstu'].shape[0]*displaybandarray[currentfilename]['LabOstu'].shape[1],3))
        #colordicesband=kmeansclassify(['LabOstu'],reshapemodified_tif)

        displaylabels=kmeansclassify()
        generateimgplant('')

        changedisplayimg(imageframe,'Origin')
        # if len(bandchoice)>0:
        #     bandchoice['LabOstu'].set('1')

        global buttondisplay,pcaframe,kmeansbar
        for widget in buttondisplay.winfo_children():
            widget.config(state=NORMAL)
        # for widget in pcaframe.winfo_children():
        # for widget in pcselframe.winfo_children():
        #     widget.config(state=NORMAL)
        extractbutton.config(state=NORMAL)
        kmeansbar.state(["!disabled"])
        pccombinebar_up.state(["!disabled"])

def fillpartialbands(vector,vectorindex,band,filter_vector):
    nonzero=np.where(filter_vector!=0)
    vector[nonzero,vectorindex]=vector[nonzero,vectorindex]+band

def fillbands(originbands,displaybands,vector,vectorindex,name,band,filter=0):
    tempdict={name:band}
    if isinstance(filter,int):
        if name not in originbands:
            originbands.update(tempdict)
            image=cv2.resize(band,(displayfea_w,displayfea_l),interpolation=cv2.INTER_LINEAR)
            displaydict={name:image}
            displaybands.update(displaydict)
            fea_bands=image.reshape((displayfea_l*displayfea_w),1)[:,0]
            vector[:,vectorindex]=vector[:,vectorindex]+fea_bands
    else:
        if name not in originbands:
            originbands.update(tempdict)
            image=cv2.resize(band,(displayfea_w,displayfea_l),interpolation=cv2.INTER_LINEAR)
            image=np.multiply(image,filter)
            displaydict={name:image}
            displaybands.update(displaydict)
            fea_bands=image.reshape((displayfea_l*displayfea_w),1)[:,0]
            vector[:,vectorindex]=vector[:,vectorindex]+fea_bands
    return

def plot3d(pcas):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    import matplotlib.pyplot as plt

    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')

    x=pcas[:,0]
    y=pcas[:,1]
    z=pcas[:,2]*0+np.min(pcas[:,2])

    ax.scatter(x,y,z,color='tab:purple')

    x=pcas[:,0]*0+np.min(pcas[:,0])
    y=pcas[:,1]
    z=pcas[:,2]

    ax.scatter(x,y,z,color='tab:pink')

    x=pcas[:,0]
    y=pcas[:,1]*0+np.max(pcas[:,1])
    z=pcas[:,2]

    ax.scatter(x,y,z,color='tab:olive')

    ax.set_xlabel('Color Indices PC1')
    ax.set_ylabel('Color Indices PC2')
    ax.set_zlabel('Color Indices PC3')

    # plt.show()
    plt.savefig('3dplot_PC.png')


def partialoneband(filter):
    global displaybandarray,originpcabands
    global pcbuttons
    global nonzero_vector,partialpca

    partialpca=True
    bands=Multiimagebands[currentfilename].bands
    channel,fea_l,fea_w=bands.shape
    nonzero=np.where(filter!=0)
    RGB_vector=np.zeros((displayfea_l*displayfea_w,3))
    colorindex_vector=np.zeros((displayfea_l*displayfea_w,12))
    filter_vector=filter.reshape((displayfea_l*displayfea_w),1)[:,0]
    originbands={}
    displays={}

    Red=cv2.resize(bands[0,:,:],(displayfea_w,displayfea_l),interpolation=cv2.INTER_LINEAR)[nonzero]

    Green=cv2.resize(bands[0,:,:],(displayfea_w,displayfea_l),interpolation=cv2.INTER_LINEAR)[nonzero]

    # Red=cv2.adaptiveThreshold(Red,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    # Green=cv2.adaptiveThreshold(Green,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

    Blue=cv2.resize(bands[0,:,:],(displayfea_w,displayfea_l),interpolation=cv2.INTER_LINEAR)[nonzero]
    # Blue=cv2.threshold(Blue,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    fillpartialbands(RGB_vector,0,Red,filter_vector)
    fillpartialbands(RGB_vector,1,Green,filter_vector)
    fillpartialbands(RGB_vector,2,Blue,filter_vector)

    PAT_R=Red
    PAT_G=Red
    PAT_B=Red

    ROO_R=Red
    ROO_G=Red
    ROO_B=Red

    DIF_R=Red
    DIF_G=Red
    DIF_B=Red

    GLD_R=Red
    GLD_G=Red
    GLD_B=Red

    fillpartialbands(colorindex_vector,0,PAT_R,filter_vector)
    fillpartialbands(colorindex_vector,1,PAT_G,filter_vector)
    fillpartialbands(colorindex_vector,2,PAT_B,filter_vector)
    # fillpartialbands(colorindex_vector,3,ROO_R,filter_vector)
    # fillpartialbands(colorindex_vector,4,ROO_G,filter_vector)
    # fillpartialbands(colorindex_vector,5,ROO_B,filter_vector)
    fillpartialbands(colorindex_vector,3,DIF_R,filter_vector)
    fillpartialbands(colorindex_vector,4,DIF_G,filter_vector)
    fillpartialbands(colorindex_vector,5,DIF_B,filter_vector)
    fillpartialbands(colorindex_vector,6,GLD_R,filter_vector)
    fillpartialbands(colorindex_vector,7,GLD_G,filter_vector)
    fillpartialbands(colorindex_vector,8,GLD_B,filter_vector)
    fillpartialbands(colorindex_vector,9,Red,filter_vector)
    fillpartialbands(colorindex_vector,10,Green,filter_vector)
    fillpartialbands(colorindex_vector,11,Blue,filter_vector)

    nonzero_vector=np.where(filter_vector!=0)

    # displayfea_vector=np.concatenate((RGB_vector,colorindex_vector),axis=1)
    displayfea_vector=np.copy(colorindex_vector)

    featurechannel=12
    # np.savetxt('color-index.csv',displayfea_vector,delimiter=',',fmt='%10.5f')

    # displayfea_vector=np.concatenate((RGB_vector,colorindex_vector),axis=1)
    originpcabands.update({currentfilename:displayfea_vector})
    pcabandsdisplay=displayfea_vector[:,:12]
    pcabandsdisplay=pcabandsdisplay.reshape(displayfea_l,displayfea_w,featurechannel)
    tempdictdisplay={'LabOstu':pcabandsdisplay}
    displaybandarray.update({currentfilename:tempdictdisplay})
    # originbandarray.update({currentfilename:originbands})

    # Red=displays['Band1']
    # Green=displays['Band2']
    # Blue=displays['Band3']

    # convimg=np.zeros((Red.shape[0],Red.shape[1],3))
    # convimg[:,:,0]=Red
    # convimg[:,:,1]=Green
    # convimg[:,:,2]=Blue
    # convimg=Image.fromarray(convimg.astype('uint8'))
    # convimg.save('convimg.png','PNG')

    pcbuttons=[]
    need_w=int(450/3)
    need_h=int(400/4)
    for i in range(2,3):
        band=np.copy(pcabandsdisplay[:,:,i])
        # imgband=(band-band.min())*255/(band.max()-band.min())
        imgband=np.copy(band)
        pcimg=Image.fromarray(imgband.astype('uint8'),'L')
        # pcimg.save('pc'+'_'+str(i)+'.png',"PNG")
        pcimg.thumbnail((need_w,need_h),Image.ANTIALIAS)
        # pcimg.save('pc'+'_'+str(i)+'.png',"PNG")
        # ratio=max(displayfea_l/need_h,displayfea_w/need_w)
        # print('origin band range',band.max(),band.min())
        # # band,cache=tkintercorestat.pool_forward(band,{"f":int(ratio),"stride":int(ratio)})
        # band=cv2.resize(band,(need_w,need_h),interpolation=cv2.INTER_LINEAR)
        # bandrange=band.max()-band.min()
        # print('band range',band.max(),band.min())
        # band=(band-band.min())/bandrange*255
        # print('button img range',band.max(),band.min())
        # buttonimg=Image.fromarray(band.astype('uint8'),'L')
        pcbuttons.append(ImageTk.PhotoImage(pcimg))



def partialsingleband(filter):
    global displaybandarray,originpcabands
    global pcbuttons
    global nonzero_vector,partialpca

    partialpca=True
    bands=Multiimagebands[currentfilename].bands
    channel,fea_l,fea_w=bands.shape
    nonzero=np.where(filter!=0)
    RGB_vector=np.zeros((displayfea_l*displayfea_w,3))
    colorindex_vector=np.zeros((displayfea_l*displayfea_w,12))
    filter_vector=filter.reshape((displayfea_l*displayfea_w),1)[:,0]
    originbands={}
    displays={}
    if channel==1:
        # Red=cv2.resize(bands[0,:,:],(displayfea_w,displayfea_l),interpolation=cv2.INTER_LINEAR)[nonzero]
        # Green=cv2.resize(bands[0,:,:],(displayfea_w,displayfea_l),interpolation=cv2.INTER_LINEAR)[nonzero]
        # Blue=cv2.resize(bands[0,:,:],(displayfea_w,displayfea_l),interpolation=cv2.INTER_LINEAR)[nonzero]
        # fillpartialbands(RGB_vector,0,Red,filter_vector)
        # fillpartialbands(RGB_vector,1,Green,filter_vector)
        # fillpartialbands(RGB_vector,2,Blue,filter_vector)
        partialoneband(filter)
        return
    else:
        Red=cv2.resize(bands[0,:,:],(displayfea_w,displayfea_l),interpolation=cv2.INTER_LINEAR)[nonzero]
        Green=cv2.resize(bands[1,:,:],(displayfea_w,displayfea_l),interpolation=cv2.INTER_LINEAR)[nonzero]
        Blue=cv2.resize(bands[2,:,:],(displayfea_w,displayfea_l),interpolation=cv2.INTER_LINEAR)[nonzero]
        fillpartialbands(RGB_vector,0,Red,filter_vector)
        fillpartialbands(RGB_vector,1,Green,filter_vector)
        fillpartialbands(RGB_vector,2,Blue,filter_vector)

    PAT_R=Red/(Red+Green)
    PAT_G=Green/(Green+Blue)
    PAT_B=Blue/(Blue+Red)

    ROO_R=Red/Green
    ROO_G=Green/Blue
    ROO_B=Blue/Red

    DIF_R=2*Red-Green-Blue
    DIF_G=2*Green-Blue-Red
    DIF_B=2*Blue-Red-Green

    GLD_R=Red/(np.multiply(np.power(Blue,0.618),np.power(Green,0.382)))
    GLD_G=Green/(np.multiply(np.power(Blue,0.618),np.power(Red,0.382)))
    GLD_B=Blue/(np.multiply(np.power(Green,0.618),np.power(Red,0.382)))

    fillpartialbands(colorindex_vector,0,PAT_R,filter_vector)
    fillpartialbands(colorindex_vector,1,PAT_G,filter_vector)
    fillpartialbands(colorindex_vector,2,PAT_B,filter_vector)
    # fillpartialbands(colorindex_vector,3,ROO_R,filter_vector)
    # fillpartialbands(colorindex_vector,4,ROO_G,filter_vector)
    # fillpartialbands(colorindex_vector,5,ROO_B,filter_vector)
    fillpartialbands(colorindex_vector,3,DIF_R,filter_vector)
    fillpartialbands(colorindex_vector,4,DIF_G,filter_vector)
    fillpartialbands(colorindex_vector,5,DIF_B,filter_vector)
    fillpartialbands(colorindex_vector,6,GLD_R,filter_vector)
    fillpartialbands(colorindex_vector,7,GLD_G,filter_vector)
    fillpartialbands(colorindex_vector,8,GLD_B,filter_vector)
    fillpartialbands(colorindex_vector,9,Red,filter_vector)
    fillpartialbands(colorindex_vector,10,Green,filter_vector)
    fillpartialbands(colorindex_vector,11,Blue,filter_vector)


    for i in range(12):
        perc=np.percentile(colorindex_vector[:,i],1)
        print('perc',perc)
        colorindex_vector[:,i]=np.where(colorindex_vector[:,i]<perc,perc,colorindex_vector[:,i])
        perc=np.percentile(colorindex_vector[:,i],99)
        print('perc',perc)
        colorindex_vector[:,i]=np.where(colorindex_vector[:,i]>perc,perc,colorindex_vector[:,i])

    # for i in range(3):
    #     perc=np.percentile(RGB_vector[:,i],1)
    #     print('perc',perc)
    #     RGB_vector[:,i]=np.where(RGB_vector[:,i]<perc,perc,RGB_vector[:,i])
    #     perc=np.percentile(RGB_vector[:,i],99)
    #     print('perc',perc)
    #     RGB_vector[:,i]=np.where(RGB_vector[:,i]>perc,perc,RGB_vector[:,i])

    nonzero_vector=np.where(filter_vector!=0)
    rgb_M=np.mean(RGB_vector[nonzero_vector,:].T,axis=1)
    colorindex_M=np.mean(colorindex_vector[nonzero_vector,:].T,axis=1)
    print('rgb_M',rgb_M,'colorindex_M',colorindex_M)
    rgb_C=RGB_vector[nonzero_vector,:][0]-rgb_M.T
    colorindex_C=colorindex_vector[nonzero_vector,:][0]-colorindex_M.T
    rgb_V=np.corrcoef(rgb_C.T)
    color_V=np.corrcoef(colorindex_C.T)
    nans=np.isnan(color_V)
    color_V[nans]=1e-6
    rgb_std=rgb_C/(np.std(RGB_vector[nonzero_vector,:].T,axis=1)).T
    color_std=colorindex_C/(np.std(colorindex_vector[nonzero_vector,:].T,axis=1)).T
    nans=np.isnan(color_std)
    color_std[nans]=1e-6
    rgb_eigval,rgb_eigvec=np.linalg.eig(rgb_V)
    color_eigval,color_eigvec=np.linalg.eig(color_V)
    print('rgb_eigvec',rgb_eigvec)
    print('color_eigvec',color_eigvec)
    featurechannel=12
    pcabands=np.zeros((colorindex_vector.shape[0],featurechannel))
    rgbbands=np.zeros((colorindex_vector.shape[0],3))
    for i in range(0,featurechannel):
        pcn=color_eigvec[:,i]
        pcnbands=np.dot(color_std,pcn)
        pcvar=np.var(pcnbands)
        print('color index pc',i+1,'var=',pcvar)
        pcabands[nonzero_vector,i]=pcabands[nonzero_vector,i]+pcnbands

    # for i in range(9,12):
    #     pcn=rgb_eigvec[:,i-9]
    #     pcnbands=np.dot(rgb_std,pcn)
    #     pcvar=np.var(pcnbands)
    #     print('rgb pc',i-9+1,'var=',pcvar)
    #     pcabands[nonzero_vector,i]=pcabands[nonzero_vector,i]+pcnbands
    #     rgbbands[nonzero_vector,i-9]=rgbbands[nonzero_vector,i-9]+pcnbands
    # plot3d(pcabands)
    # np.savetxt('rgb.csv',rgbbands,delimiter=',',fmt='%10.5f')
    # pcabands[:,1]=np.copy(pcabands[:,1])
    # pcabands[:,2]=pcabands[:,2]*0
    # indexbands=np.zeros((colorindex_vector.shape[0],3))

        # if i<5:
        #     indexbands[:,i-2]=indexbands[:,i-2]+pcnbands

    for i in range(12):
        perc=np.percentile(pcabands[:,i],1)
        print('perc',perc)
        pcabands[:,i]=np.where(pcabands[:,i]<perc,perc,pcabands[:,i])
        perc=np.percentile(pcabands[:,i],99)
        print('perc',perc)
        pcabands[:,i]=np.where(pcabands[:,i]>perc,perc,pcabands[:,i])

    '''save to csv'''
    # indexbands[:,0]=indexbands[:,0]+pcabands[:,2]
    # indexbands[:,1]=indexbands[:,1]+pcabands[:,3]
    # indexbands[:,2]=indexbands[:,2]+pcabands[:,4]
    # plot3d(indexbands)
    # np.savetxt('pcs.csv',pcabands,delimiter=',',fmt='%10.5f')

    # displayfea_vector=np.concatenate((RGB_vector,colorindex_vector),axis=1)
    # np.savetxt('color-index.csv',displayfea_vector,delimiter=',',fmt='%10.5f')
    displayfea_vector=np.copy(colorindex_vector)

    # displayfea_vector=np.concatenate((RGB_vector,colorindex_vector),axis=1)
    originpcabands.update({currentfilename:displayfea_vector})
    pcabandsdisplay=pcabands.reshape(displayfea_l,displayfea_w,featurechannel)
    tempdictdisplay={'LabOstu':pcabandsdisplay}
    displaybandarray.update({currentfilename:tempdictdisplay})
    # originbandarray.update({currentfilename:originbands})

    # Red=displays['Band1']
    # Green=displays['Band2']
    # Blue=displays['Band3']

    # convimg=np.zeros((Red.shape[0],Red.shape[1],3))
    # convimg[:,:,0]=Red
    # convimg[:,:,1]=Green
    # convimg[:,:,2]=Blue
    # convimg=Image.fromarray(convimg.astype('uint8'))
    # convimg.save('convimg.png','PNG')

    pcbuttons=[]
    need_w=int(450/3)
    need_h=int(400/4)
    for i in range(12):
        band=np.copy(pcabandsdisplay[:,:,i])
        imgband=(band-band.min())*255/(band.max()-band.min())
        pcimg=Image.fromarray(imgband.astype('uint8'),'L')
        # pcimg.save('pc'+'_'+str(i)+'.png',"PNG")
        pcimg.thumbnail((need_w,need_h),Image.ANTIALIAS)
        # pcimg.save('pc'+'_'+str(i)+'.png',"PNG")
        # ratio=max(displayfea_l/need_h,displayfea_w/need_w)
        # print('origin band range',band.max(),band.min())
        # # band,cache=tkintercorestat.pool_forward(band,{"f":int(ratio),"stride":int(ratio)})
        # band=cv2.resize(band,(need_w,need_h),interpolation=cv2.INTER_LINEAR)
        # bandrange=band.max()-band.min()
        # print('band range',band.max(),band.min())
        # band=(band-band.min())/bandrange*255
        # print('button img range',band.max(),band.min())
        # buttonimg=Image.fromarray(band.astype('uint8'),'L')
        pcbuttons.append(ImageTk.PhotoImage(pcimg))

def oneband(file):
    global displaybandarray,originbandarray,originpcabands,displayfea_l,displayfea_w
    global pcbuttons
    global partialpca
    partialpca=False

    try:
        bands=Multiimagebands[file].bands
    except:
        return
    pcbuttons=[]
    channel,fea_l,fea_w=bands.shape
    print('bandsize',fea_l,fea_w)
    if fea_l*fea_w>2000*2000:
        ratio=findratio([fea_l,fea_w],[2000,2000])
    else:
        ratio=1
    print('ratio',ratio)

    originbands={}
    displays={}
    displaybands=cv2.resize(bands[0,:,:],(int(fea_w/ratio),int(fea_l/ratio)),interpolation=cv2.INTER_LINEAR)
    displayfea_l,displayfea_w=displaybands.shape
    RGB_vector=np.zeros((displayfea_l*displayfea_w,3))
    colorindex_vector=np.zeros((displayfea_l*displayfea_w,12))
    Red=bands[0,:,:].astype('uint8')
    # _,Red=cv2.threshold(Red,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    Green=bands[0,:,:].astype('uint8')
    # _,Green=cv2.threshold(Green,0,255,cv2.THRESH_OTSU)
    Blue=bands[0,:,:].astype('uint8')
    # _,Blue=cv2.threshold(Blue,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    fillbands(originbands,displays,RGB_vector,0,'Band1',Red)
    fillbands(originbands,displays,RGB_vector,1,'Band2',Green)
    fillbands(originbands,displays,RGB_vector,2,'Band3',Blue)

    PAT_R=bands[0,:,:].astype('uint8')
    # PAT_R=cv2.adaptiveThreshold(PAT_R,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    PAT_G=bands[0,:,:]
    # PAT_G=cv2.adaptiveThreshold(PAT_G,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    PAT_B=bands[0,:,:]

    ROO_R=bands[0,:,:]
    ROO_G=bands[0,:,:]
    ROO_B=bands[0,:,:]

    DIF_R=bands[0,:,:]
    DIF_G=bands[0,:,:]
    DIF_B=bands[0,:,:]

    GLD_R=bands[0,:,:]
    GLD_G=bands[0,:,:]
    GLD_B=bands[0,:,:]

    fillbands(originbands,displays,colorindex_vector,0,'PAT_R',PAT_R)
    fillbands(originbands,displays,colorindex_vector,1,'PAT_G',PAT_G)
    fillbands(originbands,displays,colorindex_vector,2,'PAT_B',PAT_B)
    fillbands(originbands,displays,colorindex_vector,3,'ROO_R',ROO_R)
    fillbands(originbands,displays,colorindex_vector,4,'ROO_G',ROO_G)
    fillbands(originbands,displays,colorindex_vector,5,'ROO_B',ROO_B)
    fillbands(originbands,displays,colorindex_vector,6,'DIF_R',DIF_R)
    fillbands(originbands,displays,colorindex_vector,7,'DIF_G',DIF_G)
    fillbands(originbands,displays,colorindex_vector,8,'DIF_B',DIF_B)
    fillbands(originbands,displays,colorindex_vector,9,'GLD_R',GLD_R)
    fillbands(originbands,displays,colorindex_vector,10,'GLD_G',GLD_G)
    fillbands(originbands,displays,colorindex_vector,11,'GLD_B',GLD_B)

    displayfea_vector=np.concatenate((RGB_vector,colorindex_vector),axis=1)
    # np.savetxt('color-index.csv',displayfea_vector,delimiter=',',fmt='%10.5f')
    featurechannel=14


    originpcabands.update({file:displayfea_vector})
    # pcabandsdisplay=pcabands.reshape(displayfea_l,displayfea_w,featurechannel)
    # pcabandsdisplay=np.concatenate((RGB_vector,colorindex_vector),axis=2)
    pcabandsdisplay=displayfea_vector[:,:14]
    pcabandsdisplay=pcabandsdisplay.reshape(displayfea_l,displayfea_w,featurechannel)
    tempdictdisplay={'LabOstu':pcabandsdisplay}
    displaybandarray.update({file:tempdictdisplay})
    originbandarray.update({file:originbands})

    # Red=displays['Band1']
    # Green=displays['Band2']
    # Blue=displays['Band3']

    # convimg=np.zeros((Red.shape[0],Red.shape[1],3))
    # convimg[:,:,0]=Red
    # convimg[:,:,1]=Green
    # convimg[:,:,2]=Blue
    # convimg=Image.fromarray(convimg.astype('uint8'))
    # convimg.save('convimg.png','PNG')

    need_w=int(450/3)
    need_h=int(400/4)
    for i in range(2,3):
        band=np.copy(pcabandsdisplay[:,:,i])
        # band=np.copy(Red)
        # imgband=(band-band.min())*255/(band.max()-band.min())
        imgband=np.copy(band)
        pcimg=Image.fromarray(imgband.astype('uint8'),'L')
        # pcimg.save('pc'+'_'+str(i)+'.png',"PNG")
        pcimg.thumbnail((need_w,need_h),Image.ANTIALIAS)
        # pcimg.save('pc'+'_'+str(i)+'.png',"PNG")
        # ratio=max(displayfea_l/need_h,displayfea_w/need_w)
        # print('origin band range',band.max(),band.min())
        # # band,cache=tkintercorestat.pool_forward(band,{"f":int(ratio),"stride":int(ratio)})
        # band=cv2.resize(band,(need_w,need_h),interpolation=cv2.INTER_LINEAR)
        # bandrange=band.max()-band.min()
        # print('band range',band.max(),band.min())
        # band=(band-band.min())/bandrange*255
        # print('button img range',band.max(),band.min())
        # buttonimg=Image.fromarray(band.astype('uint8'),'L')
        pcbuttons.append(ImageTk.PhotoImage(pcimg))




def singleband(file):
    global displaybandarray,originbandarray,originpcabands,displayfea_l,displayfea_w
    global pcbuttons
    global partialpca
    partialpca=False

    try:
        bands=Multiimagebands[file].bands
    except:
        return
    pcbuttons=[]
    channel,fea_l,fea_w=bands.shape
    print('bandsize',fea_l,fea_w)
    if fea_l*fea_w>2000*2000:
        ratio=findratio([fea_l,fea_w],[2000,2000])
    else:
        ratio=1
    print('ratio',ratio)

    originbands={}
    displays={}
    displaybands=cv2.resize(bands[0,:,:],(int(fea_w/ratio),int(fea_l/ratio)),interpolation=cv2.INTER_LINEAR)
    displayfea_l,displayfea_w=displaybands.shape
    print(displayfea_l,displayfea_w)
    RGB_vector=np.zeros((displayfea_l*displayfea_w,3))
    colorindex_vector=np.zeros((displayfea_l*displayfea_w,12))
    if channel==1:
        # Red=bands[0,:,:]
        # Green=bands[0,:,:]
        # Blue=bands[0,:,:]
        oneband(file)
        return
    else:
        Red=bands[0,:,:]
        Green=bands[1,:,:]
        Blue=bands[2,:,:]
    # fillbands(originbands,displays,RGB_vector,0,'Band1',Red)
    # fillbands(originbands,displays,RGB_vector,1,'Band2',Green)
    # fillbands(originbands,displays,RGB_vector,2,'Band3',Blue)

    # import matplotlib.pyplot as plt
    # fig,axs=plt.subplots(1,3)
    # for i in range(3):
    #     minpc2=np.min(RGB_vector[:,i])
    #     maxpc2=np.max(RGB_vector[:,i])
    #     print(minpc2,maxpc2)
    #     bins=range(int(minpc2),int(maxpc2),10)
    #     axs[i].hist(RGB_vector[:,i],bins,range=(minpc2,maxpc2))
    #     axs[i].set_title('RGBband_'+str(i+1))
    # # plt.hist(pcabands[:,13],bins,range=(minpc2,maxpc2))
    # plt.show()




    # secondsmallest_R=np.partition(Red,1)[1][0]
    # secondsmallest_G=np.partition(Green,1)[1][0]
    # secondsmallest_B=np.partition(Blue,1)[1][0]
    #
    # Red=Red+secondsmallest_R
    # Green=Green+secondsmallest_G
    # Blue=Blue+secondsmallest_B

    # Red=Red/255+1
    # Green=Green/255+1
    # Blue=Blue/255+1


    PAT_R=Red/(Red+Green)
    PAT_G=Green/(Green+Blue)
    PAT_B=Blue/(Blue+Red)

    # ROO_R=Red/(Green+1e-6)
    # ROO_G=Green/(Blue+1e-6)
    # ROO_B=Blue/(Red+1e-6)

    DIF_R=2*Red-Green-Blue
    DIF_G=2*Green-Blue-Red
    DIF_B=2*Blue-Red-Green

    GLD_R=Red/(np.multiply(np.power(Blue,0.618),np.power(Green,0.382))+1e-6)
    GLD_G=Green/(np.multiply(np.power(Blue,0.618),np.power(Red,0.382))+1e-6)
    GLD_B=Blue/(np.multiply(np.power(Green,0.618),np.power(Red,0.382))+1e-6)

    fillbands(originbands,displays,colorindex_vector,0,'PAT_R',PAT_R)
    fillbands(originbands,displays,colorindex_vector,1,'PAT_G',PAT_G)
    fillbands(originbands,displays,colorindex_vector,2,'PAT_B',PAT_B)
    # fillbands(originbands,displays,colorindex_vector,3,'ROO_R',ROO_R)
    # fillbands(originbands,displays,colorindex_vector,4,'ROO_G',ROO_G)
    # fillbands(originbands,displays,colorindex_vector,5,'ROO_B',ROO_B)
    fillbands(originbands,displays,colorindex_vector,3,'DIF_R',DIF_R)
    fillbands(originbands,displays,colorindex_vector,4,'DIF_G',DIF_G)
    fillbands(originbands,displays,colorindex_vector,5,'DIF_B',DIF_B)
    fillbands(originbands,displays,colorindex_vector,6,'GLD_R',GLD_R)
    fillbands(originbands,displays,colorindex_vector,7,'GLD_G',GLD_G)
    fillbands(originbands,displays,colorindex_vector,8,'GLD_B',GLD_B)
    fillbands(originbands,displays,colorindex_vector,9,'Band1',Red)
    fillbands(originbands,displays,colorindex_vector,10,'Band2',Green)
    fillbands(originbands,displays,colorindex_vector,11,'Band3',Blue)

    # for i in [5,11]:
    #     colorindex_vector[:,i]=np.log10(colorindex_vector[:,i])
    #     perc=np.percentile(colorindex_vector[:,i],99)
    #     print('perc',perc)
    #     colorindex_vector[:,i]=np.where(colorindex_vector[:,i]>perc,perc,colorindex_vector[:,i])
    #
    # for i in [0,1,3,4,9,10]:
    #     colorindex_vector[:,i]=np.log10(colorindex_vector[:,i])
    #     perc=np.percentile(colorindex_vector[:,i],90)
    #     print('perc',perc)
    #     colorindex_vector[:,i]=np.where(colorindex_vector[:,i]>perc,perc,colorindex_vector[:,i])

    # for i in [5,11]:
    #     colorindex_vector[:,i]=np.log10(colorindex_vector[:,i])
    #     perc=np.percentile(colorindex_vector[:,i],99)
    #     print('perc',perc)
    #     colorindex_vector[:,i]=np.where(colorindex_vector[:,i]>perc,perc,colorindex_vector[:,i])
    #
    # for i in [3,4,9,10]:
    #     colorindex_vector[:,i]=np.log10(colorindex_vector[:,i])
    #     perc=np.percentile(colorindex_vector[:,i],1)
    #     print('perc',perc)
    #     colorindex_vector[:,i]=np.where(colorindex_vector[:,i]<perc,perc,colorindex_vector[:,i])
    #     perc=np.percentile(colorindex_vector[:,i],99)
    #     print('perc',perc)
    #     colorindex_vector[:,i]=np.where(colorindex_vector[:,i]>perc,perc,colorindex_vector[:,i])
    #
    # for i in [0,1]:
    #     colorindex_vector[:,i]=np.log10(colorindex_vector[:,i])
    #     perc=np.percentile(colorindex_vector[:,i],2)
    #     print('perc',perc)
    #     colorindex_vector[:,i]=np.where(colorindex_vector[:,i]<perc,perc,colorindex_vector[:,i])
    # for i in [0,1,3,4,9,10]:
    #     colorindex_vector[:,i]=np.log10(colorindex_vector[:,i])
    for i in range(12):
        perc=np.percentile(colorindex_vector[:,i],1)
        print('perc',perc)
        colorindex_vector[:,i]=np.where(colorindex_vector[:,i]<perc,perc,colorindex_vector[:,i])
        perc=np.percentile(colorindex_vector[:,i],99)
        print('perc',perc)
        colorindex_vector[:,i]=np.where(colorindex_vector[:,i]>perc,perc,colorindex_vector[:,i])

    # for i in [3,4,5]:
    #     perc=np.percentile(colorindex_vector[:,i],10)
    #     colorindex_vector[:,i]=np.where(colorindex_vector[:,i]<perc,perc,colorindex_vector[:,i])

    # for i in range(3):
    #     perc=np.percentile(RGB_vector[:,i],1)
    #     print('perc',perc)
    #     RGB_vector[:,i]=np.where(RGB_vector[:,i]<perc,perc,RGB_vector[:,i])
    #     perc=np.percentile(RGB_vector[:,i],99)
    #     print('perc',perc)
    #     RGB_vector[:,i]=np.where(RGB_vector[:,i]>perc,perc,RGB_vector[:,i])

    # import matplotlib.pyplot as plt
    # fig,axs=plt.subplots(4,3)
    # for i in range(12):
    #     minpc2=np.min(colorindex_vector[:,i])
    #     maxpc2=np.max(colorindex_vector[:,i])
    #     print(minpc2,maxpc2)
    #     # bins=range(int(minpc2),int(maxpc2)+1,10)
    #     axs[int(i/3),i%3].hist(colorindex_vector[:,i],10,range=(minpc2,maxpc2))
    #     axs[int(i/3),i%3].set_title('Colorindex_'+str(i+1))
    #     # axs[i].hist(colorindex_vector[:,i],10,range=(minpc2,maxpc2))
    #     # axs[i].set_title('Colorindex_'+str(i+1))
    # # plt.hist(pcabands[:,13],bins,range=(minpc2,maxpc2))
    # plt.show()
    # header=['PAT_R','PAT_G','PAT_B',
    #         'DIF_R','DIF_G','DIF_B',
    #         'GLD_R','GLD_G','GLD_B',
    #         'R','G','B']
    # # displayfea_vector=np.concatenate((RGB_vector,colorindex_vector),axis=1)
    # displayfea_vector=np.copy(colorindex_vector)
    # with open('color-index.csv','w') as f:
    #     writer=csv.writer(f)
    #     writer.writerow(header)
    #     for i in range(displayfea_vector.shape[0]):
    #         writer.writerow(list(displayfea_vector[i,:]))

    rgb_M=np.mean(RGB_vector.T,axis=1)
    colorindex_M=np.mean(colorindex_vector.T,axis=1)
    print('rgb_M',rgb_M,'colorindex_M',colorindex_M)
    rgb_C=RGB_vector-rgb_M
    colorindex_C=colorindex_vector-colorindex_M
    rgb_V=np.corrcoef(rgb_C.T)
    color_V=np.corrcoef(colorindex_C.T)
    nans=np.isnan(color_V)
    color_V[nans]=1e-6
    try:
        rgb_std=rgb_C/np.std(RGB_vector.T,axis=1)
    except:
        pass
    color_std=colorindex_C/np.std(colorindex_vector.T,axis=1)
    nans=np.isnan(color_std)
    color_std[nans]=1e-6
    try:
        rgb_eigval,rgb_eigvec=np.linalg.eig(rgb_V)
        print('rgb_eigvec',rgb_eigvec)
    except:
        pass
    color_eigval,color_eigvec=np.linalg.eig(color_V)

    print('color_eigvec',color_eigvec)
    featurechannel=12
    pcabands=np.zeros((colorindex_vector.shape[0],featurechannel))
    rgbbands=np.zeros((colorindex_vector.shape[0],3))

    # plot3d(pcabands)
    # np.savetxt('rgb.csv',rgbbands,delimiter=',',fmt='%10.5f')
    # pcabands[:,1]=np.copy(pcabands[:,1])
    # pcabands[:,2]=pcabands[:,2]*0
    indexbands=np.zeros((colorindex_vector.shape[0],3))
    # for i in range(3,featurechannel):
    # csvpcabands=np.zeros((colorindex_vector.shape[0],15))

    for i in range(12):
        pcn=color_eigvec[:,i]
        pcnbands=np.dot(color_std,pcn)
        pcvar=np.var(pcnbands)
        print('color index pc',i+1,'var=',pcvar)
        pcabands[:,i]=pcabands[:,i]+pcnbands
        # if i<5:
        #     indexbands[:,i-2]=indexbands[:,i-2]+pcnbands
    # for i in range(9,12):
    #     pcn=rgb_eigvec[:,i-9]
    #     pcnbands=np.dot(rgb_std,pcn)
    #     pcvar=np.var(pcnbands)
    #     print('rgb pc',i+1,'var=',pcvar)
    #     pcabands[:,i]=pcabands[:,i]+pcnbands
    #     rgbbands[:,i-9]=rgbbands[:,i-9]+pcnbands
    # for i in range(0,12):
    #     pcn=color_eigvec[:,i]
    #     pcnbands=np.dot(color_std,pcn)
    #     pcvar=np.var(pcnbands)
    #     print('csv color index pc',i+1,'var=',pcvar)
    #     csvpcabands[:,i]=csvpcabands[:,i]+pcnbands
    # for i in range(12,15):
    #     pcn=rgb_eigvec[:,i-12]
    #     pcnbands=np.dot(rgb_std,pcn)
    #     csvpcabands[:,i]=csvpcabands[:,i]+pcnbands

    #


    '''save to csv'''
    # indexbands[:,0]=indexbands[:,0]+pcabands[:,2]
    # indexbands[:,1]=indexbands[:,1]+pcabands[:,3]
    # indexbands[:,2]=indexbands[:,2]+pcabands[:,4]
    # plot3d(indexbands)
    # np.savetxt('pcs.csv',pcabands,delimiter=',',fmt='%10.5f')
    # minpc=np.min(pcabands)
    #
    # meanpc=np.mean(pcabands)
    # stdpc=np.std(pcabands)
    # print('meanpc',meanpc,'stdpc',stdpc)
    # pcabands=pcabands-meanpc/stdpc
    # import matplotlib.pyplot as plt
    # minpc2=np.min(pcabands[:,13])
    # maxpc2=np.max(pcabands[:,13])
    # print(minpc2,maxpc2)
    # bins=range(int(minpc2),int(maxpc2),10)
    # plt.hist(pcabands[:,13],bins,range=(minpc2,maxpc2))
    # plt.show()
    # np.savetxt('pcs.csv',pcabands[:,3],delimiter=',',fmt='%10.5f')
    # header=['PC1','PC2','PC3',
    #         'PC4','PC5','PC6',
    #         'PC7','PC8','PC9',
    #         'PC10','PC11','PC12']
    # # displayfea_vector=np.concatenate((RGB_vector,colorindex_vector),axis=1)
    # pcatalbe=np.copy(pcabands)
    # with open('pcabands.csv','w') as f:
    #     writer=csv.writer(f)
    #     writer.writerow(header)
    #     for i in range(pcatalbe.shape[0]):
    #         writer.writerow(list(pcatalbe[i,:]))

    for i in range(12):
        perc=np.percentile(pcabands[:,i],1)
        print('perc',perc)
        pcabands[:,i]=np.where(pcabands[:,i]<perc,perc,pcabands[:,i])
        perc=np.percentile(pcabands[:,i],99)
        print('perc',perc)
        pcabands[:,i]=np.where(pcabands[:,i]>perc,perc,pcabands[:,i])

    # import matplotlib.pyplot as plt
    # fig,axs=plt.subplots(4,3)
    # for i in range(2,14):
    #     minpc2=np.min(pcabands[:,i])
    #     maxpc2=np.max(pcabands[:,i])
    #     print(minpc2,maxpc2)
    #     # bins=range(int(minpc2),int(maxpc2)+1,10)
    #     axs[int((i-2)/3),(i-2)%3].hist(pcabands[:,i],10,range=(minpc2,maxpc2))
    #     axs[int((i-2)/3),(i-2)%3].set_title('PC_'+str(i-2+1))
    #     # axs[i].hist(colorindex_vector[:,i],10,range=(minpc2,maxpc2))
    #     # axs[i].set_title('Colorindex_'+str(i+1))
    # # plt.hist(pcabands[:,13],bins,range=(minpc2,maxpc2))
    # plt.show()


    # header=['PAT_R','PAT_G','PAT_B',
    #         'DIF_R','DIF_G','DIF_B',
    #         'GLD_R','GLD_G','GLD_B',
    #         'R','G','B']
    # # displayfea_vector=np.concatenate((RGB_vector,colorindex_vector),axis=1)
    # displayfea_vector=np.copy(colorindex_vector)
    # with open('color-index.csv','w') as f:
    #     writer=csv.writer(f)
    #     writer.writerow(header)
    #     for i in range(displayfea_vector.shape[0]):
    #         writer.writerow(list(displayfea_vector[i,:]))


    # np.savetxt('color-index.csv',displayfea_vector,delimiter=',',fmt='%10.5f')

    # displayfea_vector=np.concatenate((RGB_vector,colorindex_vector),axis=1)
    displayfea_vector=np.copy(colorindex_vector)
    originpcabands.update({file:displayfea_vector})
    pcabandsdisplay=pcabands.reshape(displayfea_l,displayfea_w,featurechannel)
    tempdictdisplay={'LabOstu':pcabandsdisplay}
    displaybandarray.update({file:tempdictdisplay})
    originbandarray.update({file:originbands})

    # Red=displays['Band1']
    # Green=displays['Band2']
    # Blue=displays['Band3']

    # convimg=np.zeros((Red.shape[0],Red.shape[1],3))
    # convimg[:,:,0]=Red
    # convimg[:,:,1]=Green
    # convimg[:,:,2]=Blue
    # convimg=Image.fromarray(convimg.astype('uint8'))
    # convimg.save('convimg.png','PNG')

    need_w=int(450/3)
    need_h=int(400/4)
    # pcdisplay=[3,4,5,6,7,8,9,10,11,0,1,2]
    # for i in range(2,featurechannel):
    for i in range(featurechannel):
        band=np.copy(pcabandsdisplay[:,:,i])
        imgband=(band-band.min())*255/(band.max()-band.min())
        pcimg=Image.fromarray(imgband.astype('uint8'),'L')
        # pcimg.save('pc'+'_'+str(i)+'.png',"PNG")
        pcimg.thumbnail((need_w,need_h),Image.ANTIALIAS)
        # pcimg.save('pc'+'_'+str(i)+'.png',"PNG")
        # ratio=max(displayfea_l/need_h,displayfea_w/need_w)
        # print('origin band range',band.max(),band.min())
        # # band,cache=tkintercorestat.pool_forward(band,{"f":int(ratio),"stride":int(ratio)})
        # band=cv2.resize(band,(need_w,need_h),interpolation=cv2.INTER_LINEAR)
        # bandrange=band.max()-band.min()
        # print('band range',band.max(),band.min())
        # band=(band-band.min())/bandrange*255
        # print('button img range',band.max(),band.min())
        # buttonimg=Image.fromarray(band.astype('uint8'),'L')
        pcbuttons.append(ImageTk.PhotoImage(pcimg))


def colorindices_cal(file):
    global colorindicearray
    try:
        bands=Multiimagebands[file].bands
    except:
        return
    channel,fea_l,fea_w=bands.shape
    print('bandsize',fea_l,fea_w)
    if fea_l*fea_w>2000*2000:
        ratio=findratio([fea_l,fea_w],[2000,2000])
    else:
        ratio=1
    print('ratio',ratio)

    originbands={}
    displays={}
    # displaybands=cv2.resize(bands[0,:,:],(int(fea_w/ratio),int(fea_l/ratio)),interpolation=cv2.INTER_LINEAR)
    # displaybands=np.copy(bands[0,:,:])
    # displayfea_l,displayfea_w=displaybands.shape
    # displayfea_l,displayfea_w=fea_l,fea_w
    print(displayfea_l,displayfea_w)
    colorindex_vector=np.zeros((displayfea_l*displayfea_w,7))
    if channel==1:
        Red=bands[0,:,:]
        Green=bands[0,:,:]
        Blue=bands[0,:,:]
    else:
        Red=bands[0,:,:]
        Green=bands[1,:,:]
        Blue=bands[2,:,:]

    secondsmallest_R=np.partition(Red,1)[1][0]
    secondsmallest_G=np.partition(Green,1)[1][0]
    secondsmallest_B=np.partition(Blue,1)[1][0]

    Red=Red+secondsmallest_R
    Green=Green+secondsmallest_G
    Blue=Blue+secondsmallest_B

    NDI=128*((Green-Red)/(Green+Red)+1)
    VEG=Green/(np.power(Red,0.667)*np.power(Blue,(1-0.667)))
    Greenness=Green/(Green+Red+Blue)
    CIVE=0.44*Red+0.811*Green+0.385*Blue+18.7845
    MExG=1.262*Green-0.844*Red-0.311*Blue
    NDRB=(Red-Blue)/(Red+Blue)
    NGRDI=(Green-Red)/(Green+Red)

    fillbands(originbands,displays,colorindex_vector,0,'NDI',NDI)
    fillbands(originbands,displays,colorindex_vector,1,'VEG',VEG)
    fillbands(originbands,displays,colorindex_vector,2,'Greenness',Greenness)
    fillbands(originbands,displays,colorindex_vector,3,'CIVE',CIVE)
    fillbands(originbands,displays,colorindex_vector,4,'MExG',MExG)
    fillbands(originbands,displays,colorindex_vector,5,'NDRB',NDRB)
    fillbands(originbands,displays,colorindex_vector,6,'NGRDI',NGRDI)

    colorindicearray.update({file:originbands})


def singleband_oldversion(file):
    global displaybandarray,originbandarray,originpcabands,displayfea_l,displayfea_w
    global pcbuttons
    try:
        bands=Multigraybands[file].bands
    except:
        return
    pcbuttons=[]
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
    # fea_vector=np.zeros((fea_l*fea_w,3))

    pyplt.imsave('bands.png',bands)
    displaybands=cv2.resize(bands,(int(bandsize[1]/ratio),int(bandsize[0]/ratio)),interpolation=cv2.INTER_LINEAR)
    pyplt.imsave('displaybands.png',displaybands)
    displayfea_l,displayfea_w=displaybands.shape
    fea_vector=np.zeros((displayfea_l*displayfea_w,3))
    displayfea_vector=np.zeros((displayfea_l*displayfea_w,7))
    colorfea_vector=np.zeros((displayfea_l*displayfea_w,7))
    # originfea_vector=np.zeros((bandsize[0],bandsize[1],10))

    # saveimg=np.copy(bands).astype('uint8')
    # pyplt.imsave('ostuimg.png',saveimg)

    if 'LabOstu' not in originbands:
        originbands.update({'LabOstu':bands})
        fea_bands=bands.reshape(fea_l*fea_w,1)[:,0]
        # originfea_vector[:,9]=originfea_vector[:,0]+fea_bands
        displayfea_bands=displaybands.reshape((displayfea_l*displayfea_w),1)[:,0]
        # fea_vector[:,9]=fea_vector[:,0]+fea_bands
        displayfea_vector[:,6]=displayfea_vector[:,6]+displayfea_bands
        minv=displayfea_bands.min()
        maxv=displayfea_bands.max()
        fearange=maxv-minv
        colorfeabands=displayfea_bands-minv
        colorfeabands=colorfeabands/fearange*255
        colorfea_vector[:,6]=colorfea_vector[:,6]+colorfeabands
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
        # fea_vector[:,1]=fea_vector[:,1]+fea_bands
        displayfea_vector[:,1]=displayfea_vector[:,1]+displayfea_bands
        minv=displayfea_bands.min()
        maxv=displayfea_bands.max()
        fearange=maxv-minv
        colorfeabands=displayfea_bands-minv
        colorfeabands=colorfeabands/fearange*255
        colorfea_vector[:,1]=colorfea_vector[:,1]+colorfeabands
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
        # fea_bands=Red.reshape(fea_l*fea_w,1)[:,0]
        fea_bands=image.reshape((displayfea_l*displayfea_w),1)[:,0]
        # originfea_vector[:,2]=originfea_vector[:,2]+fea_bands
        displayfea_bands=image.reshape((displayfea_l*displayfea_w),1)[:,0]
        fea_vector[:,0]=fea_vector[:,0]+fea_bands
        # displayfea_vector[:,2]=displayfea_vector[:,2]+displayfea_bands
    tempdict={'Band2':Green}
    if 'Band2' not in originbands:
        originbands.update(tempdict)

        image=cv2.resize(Green,(int(bandsize[1]/ratio),int(bandsize[0]/ratio)),interpolation=cv2.INTER_LINEAR)
        displaydict={'Band2':image}
        displays.update(displaydict)
        # fea_bands=Green.reshape(fea_l*fea_w,1)[:,0]
        fea_bands=image.reshape((displayfea_l*displayfea_w),1)[:,0]
        # originfea_vector[:,3]=originfea_vector[:,3]+fea_bands
        displayfea_bands=image.reshape((displayfea_l*displayfea_w),1)[:,0]
        fea_vector[:,1]=fea_vector[:,1]+fea_bands
        # displayfea_vector[:,3]=displayfea_vector[:,3]+displayfea_bands
    tempdict={'Band3':Blue}
    if 'Band3' not in originbands:
        originbands.update(tempdict)
        # originfea_vector[:,4]=originfea_vector[:,4]+Blue
        image=cv2.resize(Blue,(int(bandsize[1]/ratio),int(bandsize[0]/ratio)),interpolation=cv2.INTER_LINEAR)
        displaydict={'Band3':image}
        displays.update(displaydict)
        # fea_bands=Blue.reshape(fea_l*fea_w,1)[:,0]
        fea_bands=image.reshape((displayfea_l*displayfea_w),1)[:,0]
        displayfea_bands=image.reshape((displayfea_l*displayfea_w),1)[:,0]
        fea_vector[:,2]=fea_vector[:,2]+fea_bands
        # displayfea_vector[:,4]=displayfea_vector[:,4]+displayfea_bands
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
        # fea_vector[:,5]=fea_vector[:,5]+fea_bands
        displayfea_vector[:,2]=displayfea_vector[:,2]+displayfea_bands
        minv=displayfea_bands.min()
        maxv=displayfea_bands.max()
        fearange=maxv-minv
        colorfeabands=displayfea_bands-minv
        colorfeabands=colorfeabands/fearange*255
        colorfea_vector[:,2]=colorfea_vector[:,2]+colorfeabands
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
        # fea_vector[:,6]=fea_vector[:,6]+fea_bands
        displayfea_vector[:,3]=displayfea_vector[:,3]+displayfea_bands
        minv=displayfea_bands.min()
        maxv=displayfea_bands.max()
        fearange=maxv-minv
        colorfeabands=displayfea_bands-minv
        colorfeabands=colorfeabands/fearange*255
        colorfea_vector[:,3]=colorfea_vector[:,3]+colorfeabands
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
        # fea_vector[:,7]=fea_vector[:,7]+fea_bands
        displayfea_vector[:,4]=displayfea_vector[:,4]+displayfea_bands
        minv=displayfea_bands.min()
        maxv=displayfea_bands.max()
        fearange=maxv-minv
        colorfeabands=displayfea_bands-minv
        colorfeabands=colorfeabands/fearange*255
        colorfea_vector[:,4]=colorfea_vector[:,4]+colorfeabands
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
        # fea_vector[:,8]=fea_vector[:,8]+fea_bands
        displayfea_vector[:,5]=displayfea_vector[:,5]+displayfea_bands
        minv=displayfea_bands.min()
        maxv=displayfea_bands.max()
        fearange=maxv-minv
        colorfeabands=displayfea_bands-minv
        colorfeabands=colorfeabands/fearange*255
        colorfea_vector[:,5]=colorfea_vector[:,5]+colorfeabands
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
        # fea_vector[:,0]=fea_vector[:,9]+fea_bands
        displayfea_vector[:,0]=displayfea_vector[:,0]+displayfea_bands
        minv=displayfea_bands.min()
        maxv=displayfea_bands.max()
        fearange=maxv-minv
        colorfeabands=displayfea_bands-minv
        colorfeabands=colorfeabands/fearange*255
        colorfea_vector[:,0]=colorfea_vector[:,0]+colorfeabands
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

    '''PCA part'''
    displayfea_vector=np.concatenate((fea_vector,displayfea_vector),axis=1)
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
    featurechannel=10
    # for i in range(len(eigvalues)):
    #     print('percentage',i,eigvalues[i]/sum(eigvalues))
    #     eigvalueperc.update({i:eigvalues[i]/sum(eigvalues)})
    #     #if eigvalues[i]>0:
    #     featurechannel+=1
    # o_eigenvalue,o_eigenvector=np.linalg.eig(OV)
    pcabands=np.zeros((displayfea_vector.shape[0],featurechannel))
    # o_pcabands=np.zeros((fea_vector.shape[0],featurechannel))
    pcavar={}
    # #
    # # # separate PCs
    # # for i in range(3):
    # #     pcn=o_eigenvector[:,i]
    # #     pcnbands=np.dot(O_stddisplayfea,pcn)
    # #     pcvar=np.var(pcnbands)
    # #     print('pc',i+1,' var=',pcvar)
    # #     pcabands[:,i]=pcabands[:,i]+pcnbands
    # # for i in range(7):
    # #     pcn=eigvectors[:,i]
    # #     pcnbands=np.dot(std_displayfea,pcn)
    # #     pcvar=np.var(pcnbands)
    # #     print('pc',i+1,' var=',pcvar)
    # #     temppcavar={i:pcvar}
    # #     pcavar.update(temppcavar)
    # #     pcabands[:,i+3]=pcabands[:,i+3]+pcnbands
    # #
    # #
    # combined PCs
    for i in range(featurechannel):
        pcn=eigvectors[:,i]
        # pcnbands=np.dot(std_displayfea,pcn)
        pcnbands=np.dot(C,pcn)
        pcvar=np.var(pcnbands)
        print('pc',i+1,' var=',pcvar)
        temppcavar={i:pcvar}
        pcavar.update(temppcavar)
        pcabands[:,i]=pcabands[:,i]+pcnbands

    # ''' NO PCA'''
    # colorfea_vector=np.concatenate((fea_vector,colorfea_vector),axis=1)
    # displayfea_vector=np.concatenate((fea_vector,displayfea_vector),axis=1)
    # M=np.mean(colorfea_vector.T,axis=1)
    # print('colorfea_vector M',M)
    # pcabands=np.copy(colorfea_vector)
    # featurechannel=10

    '''Export to CSV'''
    # np.savetxt('pcs.csv',pcabands,delimiter=',',fmt='%s')
    # np.savetxt('color-index.csv',displayfea_vector,delimiter=',',fmt='%s')

    #threedplot(pcabands)
    # originpcabands.update({file:o_pcabands})
    originpcabands.update({file:displayfea_vector})
    pcabandsdisplay=pcabands.reshape(displayfea_l,displayfea_w,featurechannel)
    #originbands={'LabOstu':pcabandsdisplay}
    tempdictdisplay={'LabOstu':pcabandsdisplay}
    #displaybandarray.update({file:displays})
    displaybandarray.update({file:tempdictdisplay})
    originbandarray.update({file:originbands})

    need_w=int(450/4)
    need_h=int(400/3)
    for i in range(featurechannel):
        band=np.copy(pcabandsdisplay[:,:,i])
        ratio=max(displayfea_l/need_h,displayfea_w/need_w)
        band,cache=tkintercorestat.pool_forward(band,{"f":int(ratio),"stride":int(ratio)})
        bandrange=band.max()-band.min()
        band=(band-band.min())/bandrange*255
        buttonimg=Image.fromarray(band.astype('uint8'),'L')
        pcbuttons.append(ImageTk.PhotoImage(buttonimg))
        # buttonimg.save('pcbutton_'+str(i)+'.png',"PNG")
        # print('saved')


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
    changekmeansbar('')
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
        ch=Checkbutton(checkboxframe,text=dictkey,variable=checkboxdict[dictkey],command=partial(changeclusterbox,''))#,command=partial(changecluster,''))
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

def generateimgplant(event):
    global currentlabels,changekmeans,colordicesband,originbinaryimg,pre_checkbox
    global selview,selareapos,binaryselareaspose
    global displaylabels
    try:
        selview=app.getselview()
        selareapos=app.getinfo(rects[1])
    except:
        pass
    if selview=='Color Deviation' and len(selareapos)>0 and selareapos!=[0,0,1,1]:
        binaryselareaspose=selareapos.copy()
    if binaryselareaspose!=[0,0,1,1] and len(binaryselareaspose)>0:
        npfilter = np.zeros((displayimg['Origin']['Size'][0], displayimg['Origin']['Size'][1]))
        print("npfilter shape:", npfilter.shape)
        filter = Image.fromarray(npfilter)
        draw = ImageDraw.Draw(filter)
        if app.getdrawpolygon() == False:
            draw.ellipse(binaryselareaspose, fill='red')
        else:
            draw.polygon(binaryselareaspose, fill='red')
        filter = np.array(filter)
        filter = np.divide(filter, np.max(filter))
        # filter = cv2.resize(filter, (displaylabels.shape[1], displaylabels.shape[0]), interpolation=cv2.INTER_LINEAR)
        # displaylabels = np.multiply(displaylabels, filter)




    colordicesband=np.copy(displaylabels)
    keys=checkboxdict.keys()
    plantchoice=[]
    pre_checkbox=[]
    for key in keys:
        plantchoice.append(checkboxdict[key].get())
        pre_checkbox.append(checkboxdict[key].get())
    try:
        origindisplaylabels=np.copy(displaybandarray[currentfilename]['LabOstu'])
        h, w, c = origindisplaylabels.shape
    except:
        return

    tempdisplayimg=np.zeros((h,w))
    colordivimg=np.zeros((h,w))
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
    try:
        filter = cv2.resize(filter, (tempdisplayimg.shape[1], tempdisplayimg.shape[0]), interpolation=cv2.INTER_LINEAR)
        tempdisplayimg = np.multiply(tempdisplayimg,filter)
    except:
        pass
    currentlabels=np.copy(tempdisplayimg)
    originbinaryimg=np.copy(tempdisplayimg)

    tempcolorimg=np.copy(displaylabels).astype('float32')

    colordivimg=np.copy(tempcolorimg)
    binaryimg=np.zeros((h,w,3))
    kvar=int(kmeans.get())
    locs=np.where(tempdisplayimg==1)
    binaryimg[locs]=[240,228,66]
    colordeimg=np.zeros((h,w,3))

    # binarypreview=cv2.resize(binaryimg,(int(previewshape[0]),int(previewshape[1])))
    binarypreview=np.copy(binaryimg)

    if kvar==1:
        if colordivimg.min()<0:
            # if abs(colordivimg.min())<colordivimg.max():
            colordivimg=colordivimg-colordivimg.min()
        colorrange=colordivimg.max()-colordivimg.min()
        colordivimg=colordivimg*255/colorrange
        grayimg=Image.fromarray(colordivimg.astype('uint8'),'L')
        grayimg=grayimg.resize((int(resizeshape[0]),int(resizeshape[1])))
        #grayimg.show()
        colordivdict={}
        colordivdict.update({'Size':[resizeshape[1],resizeshape[0]]})
        colordivdict.update({'Image':ImageTk.PhotoImage(grayimg)})
        displayimg['Color Deviation']=colordivdict

        colordivpreview={}
        # colordivpreimg=cv2.resize(colordivimg,(int(previewshape[0]),int(previewshape[1])))
        graypreviewimg=Image.fromarray(colordivimg.astype('uint8'),'L')
        graypreviewimg=graypreviewimg.resize((int(previewshape[0]),int(previewshape[1])))
        colordivpreview.update({'Size':[previewshape[1],previewshape[0]]})
        colordivpreview.update({'Image':ImageTk.PhotoImage(graypreviewimg)})
        previewimg['Color Deviation']=colordivpreview


        binaryimg=np.zeros((resizeshape[1],resizeshape[0],3))
        tempdict={}
        tempdict.update({'Size':[resizeshape[1],resizeshape[0]]})
        tempdict.update({'Image':ImageTk.PhotoImage(Image.fromarray(binaryimg.astype('uint8')))})
        displayimg['ColorIndices']=tempdict

        binarypreview=np.zeros((int(previewshape[1]),int(previewshape[0])))
        tempdict={}
        tempdict.update({'Size':binarypreview.shape})
        tempdict.update({'Image':ImageTk.PhotoImage(Image.fromarray(binarypreview.astype('uint8')))})
        previewimg['ColorIndices']=tempdict

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
        colordeimg=Image.fromarray(colordeimg.astype('uint8'))
        colordeimg.save('allcolorindex.png',"PNG")
        binaryimg=Image.fromarray(binaryimg.astype('uint8'))
        binaryimg.save('binaryimg.png',"PNG")
        binaryimg=binaryimg.resize((int(resizeshape[0]),int(resizeshape[1])))
        tempdict={}
        tempdict.update({'Size':[resizeshape[1],resizeshape[0]]})
        tempdict.update({'Image':ImageTk.PhotoImage(binaryimg)})
        displayimg['ColorIndices']=tempdict

        tempdict={}
        binaryimg=binaryimg.resize((int(previewshape[0]),int(previewshape[1])))
        tempdict.update({'Size':[previewshape[1],previewshape[0]]})
        tempdict.update({'Image':ImageTk.PhotoImage(binaryimg)})
        previewimg['ColorIndices']=tempdict


        #indimg=cv2.imread('allcolorindex.png')
        #tempdict.update({'Image':ImageTk.PhotoImage(Image.fromarray(indimg))})
        #
        # colorimg=cv2.imread('allcolorindex.png')
        # Image.fromarray((binaryimg.astype('uint8'))).save('binaryimg.png',"PNG")
        colordeimg=colordeimg.resize((resizeshape[0],resizeshape[1]))
        colordivdict={}
        colordivdict.update({'Size':[resizeshape[1],resizeshape[0]]})
        colordivdict.update({'Image':ImageTk.PhotoImage(colordeimg)})
        displayimg['Color Deviation']=colordivdict

        colordivdict={}
        # colordeimgpre=cv2.resize(colordeimg,(int(previewshape[0]),int(previewshape[1])))
        colordeimg=colordeimg.resize((previewshape[0],previewshape[1]))
        colordivdict.update({'Size':[previewshape[1],previewshape[0]]})
        colordivdict.update({'Image':ImageTk.PhotoImage(colordeimg)})
        previewimg['Color Deviation']=colordivdict

        # changedisplayimg(imageframe,'ColorIndices')
    # print('sel count',sel_count)
    if kvar>1:
        if sel_count==0:
            changedisplayimg(imageframe,'Color Deviation')
        else:
            changedisplayimg(imageframe,'ColorIndices')
    # changekmeans=True


#def kmeansclassify(choicelist,reshapedtif):
def kmeansclassify_oldversion():
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

def kmeansclassify():
    global clusterdisplay,displaylabels
    global selview,selareapos,kmeanselareapose,kmeanspolygon
    # global selareapos, pcselarea
    if int(kmeans.get())==0:
        return
    originpcabands=displaybandarray[currentfilename]['LabOstu']
    pcah,pcaw,pcac=originpcabands.shape
    print('pcah',pcah,'pcaw',pcaw,'pcac',pcac)
    pcpara=pc_combine_up.get()
    print(pcpara,type(pcpara))
    tempband=np.zeros((pcah,pcaw,1))
    # pcsel=buttonvar.get()+2
    pcsel=buttonvar.get()
    pcweights=pc_combine_up.get()-0.5
    if pcweights==0.0:
        tempband[:,:,0]=tempband[:,:,0]+originpcabands[:,:,pcsel]
    else:
        if pcweights<0.0:  #RGBPC1
            rgbpc=originpcabands[:,:,9]
        else:
            rgbpc=originpcabands[:,:,10]
        rgbpc=(rgbpc-rgbpc.min())*255/(rgbpc.max()-rgbpc.min())
        firstterm=abs(pcweights)*2*rgbpc
        colorpc=originpcabands[:,:,pcsel]
        colorpc=(colorpc-colorpc.min())*255/(colorpc.max()-colorpc.min())
        secondterm=(1-abs(pcweights)*2)*colorpc
        tempband[:,:,0]=tempband[:,:,0]+firstterm+secondterm
    if int(kmeans.get())==1:
        print('kmeans=1')
        displaylabels=np.mean(tempband,axis=2)
        pyplt.imsave('k=1.png',displaylabels)
    else:
        if int(kmeans.get())>1:
            h,w,c=tempband.shape
            print('shape',tempband.shape)
            reshapedtif=tempband.reshape(tempband.shape[0]*tempband.shape[1],c)
            selview = app.getselview()
            selareapos = app.getinfo(rects[1])
            if selview == 'PCs' and len(selareapos)>0 and selareapos!=[0,0,1,1]:
                kmeanselareapose=selareapos.copy()
                kmeanspolygon = app.getdrawpolygon()
            if kmeanselareapose!=[0,0,1,1] and len(kmeanselareapose)>0:
                npfilter=np.zeros((displayimg['Origin']['Size'][0],displayimg['Origin']['Size'][1]))
                print("npfilter shape:", npfilter.shape)
                filter = Image.fromarray(npfilter)
                draw = ImageDraw.Draw(filter)
                if kmeanspolygon == False:
                    draw.ellipse(kmeanselareapose, fill='red')
                else:
                    draw.polygon(kmeanselareapose, fill='red')
                filter = np.array(filter)
                filter = np.divide(filter, np.max(filter))
                filter = cv2.resize(filter, (displayfea_w, displayfea_l), interpolation=cv2.INTER_LINEAR)

                partialtempband=np.multiply(tempband[:,:,0],filter)
                partialshape=partialtempband.reshape(tempband.shape[0]*tempband.shape[1],c)
                nonzerovector = np.where(partialshape > 0)
                clf = KMeans(n_clusters=int(kmeans.get()), init='k-means++', n_init=10, random_state=0)
                tempdisplayimg = clf.fit(partialshape)
                # partialshape[nonzerovector,0]=np.add(tempdisplayimg.labels_,1)
                displaylabels=tempdisplayimg.labels_.reshape((displaybandarray[currentfilename]['LabOstu'].shape[0],
                                                  displaybandarray[currentfilename]['LabOstu'].shape[1]))
                # print(tempdisplayimg)
                clusterdict = {}
                displaylabels = displaylabels + 10
                for i in range(int(kmeans.get())):
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

            if partialpca==True:
                partialshape=reshapedtif[nonzero_vector]
                print('partial reshape',partialshape.shape)
                clf=KMeans(n_clusters=int(kmeans.get()),init='k-means++',n_init=10,random_state=0)
                tempdisplayimg=clf.fit(partialshape)
                reshapedtif[nonzero_vector,0]=np.add(tempdisplayimg.labels_,1)
                print(reshapedtif[nonzero_vector])
                displaylabels=reshapedtif.reshape((displaybandarray[currentfilename]['LabOstu'].shape[0],
                                                  displaybandarray[currentfilename]['LabOstu'].shape[1]))
            # reshapedtif=cv2.resize(reshapedtif,(c,resizeshape[0]*resizeshape[1]),cv2.INTER_LINEAR)
                clusterdict={}
                displaylabels=displaylabels+10
                for i in range(int(kmeans.get())):
                    locs=np.where(tempdisplayimg.labels_==i)
                    try:
                        maxval=partialshape[locs].max()
                    except:
                        print('kmeans',i)
                        messagebox.showerror('Cluster maximum value is ', i)
                        return displaylabels
                    print(maxval)
                    clusterdict.update({maxval:i+11})
                print(clusterdict)
                sortcluster=list(sorted(clusterdict))
                print(sortcluster)
                for i in range(len(sortcluster)):
                    cluster_num=clusterdict[sortcluster[i]]
                    displaylabels=np.where(displaylabels==cluster_num,i,displaylabels)
                return displaylabels
            else:
                print('reshape',reshapedtif.shape)

                clf=KMeans(n_clusters=int(kmeans.get()),init='k-means++',n_init=10,random_state=0)
                tempdisplayimg=clf.fit(reshapedtif)
                # print('label=0',np.any(tempdisplayimg==0))
                displaylabels=tempdisplayimg.labels_.reshape((displaybandarray[currentfilename]['LabOstu'].shape[0],
                                                      displaybandarray[currentfilename]['LabOstu'].shape[1]))
                    # displaylabels=tempdisplayimg.labels_.reshape((resizeshape[1],resizeshape[0]))
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

    # if kmeans.get() not in clusterdisplay:
    #     tempdict={kmeans.get():displaylabels}
    #     #clusterdisplay.update({''.join(choicelist):tempdict})
    #     clusterdisplay.update(tempdict)
    c1=np.where(displaylabels==0)
    c2=np.where(displaylabels==1)
    c3=np.where(displaylabels==2)
    c4=np.where(displaylabels==3)
    try:
        c1pix = tempband[c1]
        c1pix = np.reshape(c1pix,c1pix.shape[0])
        c2pix=tempband[c2]
        c2pix = np.reshape(c2pix,c2pix.shape[0])
        c3pix=tempband[c3]
        c4pix=tempband[c4]
        c3pix=np.reshape(c3pix,c3pix.shape[0])
        c4pix=np.reshape(c4pix,c4pix.shape[0])
        print('two cluster var:',np.var(c1pix),c1pix.shape,np.var(c2pix),c2pix.shape,np.var(c3pix),c3pix.shape,np.var(c4pix),c4pix.shape)
    except:
        pass
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
    global displayimg,displaypclabels
    originpcabands=displaybandarray[currentfilename]['LabOstu']
    pcah,pcaw,pcac=originpcabands.shape
    pcweights=pc_combine_up.get()-0.5
    tempband=np.zeros((pcah,pcaw))
    # pcsel=buttonvar.get()+2
    pcsel=buttonvar.get()
    print("pcsel",pcsel)
    if pcweights==0.0:
        tempband=tempband+originpcabands[:,:,pcsel]
    else:
        if pcweights<0.0:  #RGBPC1
            rgbpc=originpcabands[:,:,9]
        else:
            rgbpc=originpcabands[:,:,10]
        rgbpc=(rgbpc-rgbpc.min())*255/(rgbpc.max()-rgbpc.min())
        firstterm=abs(pcweights)*2*rgbpc
        colorpc=originpcabands[:,:,pcsel]
        colorpc=(colorpc-colorpc.min())*255/(colorpc.max()-colorpc.min())
        secondterm=(1-abs(pcweights)*2)*colorpc
        tempband=tempband+firstterm+secondterm
    displaypclabels=np.copy(tempband)
    displaylabels=np.copy(tempband)
    pyplt.imsave('k=1.png',displaylabels)
    colordivimg=np.copy(displaylabels)
    print('origin pc range',colordivimg.max(),colordivimg.min())
    # colordivimg=cv2.resize(tempcolorimg,(int(resizeshape[0]),int(resizeshape[1])))
    print('pc range',colordivimg.max(),colordivimg.min())
    if colordivimg.min()<0:
        colordivimg=colordivimg-colordivimg.min()
    colorrange=colordivimg.max()-colordivimg.min()
    colordivimg=(colordivimg)*255/colorrange
    colordivimg=Image.fromarray(colordivimg.astype('uint8'),'L')
    colordivimg=colordivimg.resize((int(resizeshape[0]),int(resizeshape[1])),Image.ANTIALIAS)
    displayimg['PCs']['Image']=ImageTk.PhotoImage(colordivimg)
    # displayimg['Color Deviation']['Image']=ImageTk.PhotoImage(colordivimg)

def getPCs_olcversion():
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
    ratio=findratio([originpcabands.shape[0],originpcabands.shape[1]],[screenstd,screenstd])
    tempcolorimg=np.copy(displaylabels)
    colordivimg=np.zeros((displaylabels.shape[0],
                          displaylabels.shape[1]))
    # if originpcabands.shape[0]*originpcabands.shape[1]<850*850:
    #     # tempdisplayimg=cv2.resize(originpcabands,(int(originpcabands.shape[1]*ratio),int(originpcabands.shape[0]*ratio)))
    #     colordivimg=cv2.resize(tempcolorimg,(int(colordivimg.shape[1]*ratio),int(colordivimg.shape[0]*ratio)))
    # else:
    #     # tempdisplayimg=cv2.resize(originpcabands,(int(originpcabands.shape[1]/ratio),int(originpcabands.shape[0]/ratio)))
    #     colordivimg=cv2.resize(tempcolorimg,(int(colordivimg.shape[1]/ratio),int(colordivimg.shape[0]/ratio)))
    # if colordivimg.min()<0:
    #     if abs(colordivimg.min())<colordivimg.max():
    #         colordivimg=colordivimg-colordivimg.min()
    colordivimg=cv2.resize(tempcolorimg,(int(resizeshape[0]),int(resizeshape[1])))

    if colordivimg.min()<0:
        colordivimg=colordivimg-colordivimg.min()
    colorrange=colordivimg.max()-colordivimg.min()
    colordivimg=colordivimg*255/colorrange
    colordivimg=colordivimg.astype('uint8')
    grayimg=Image.fromarray(colordivimg,'L')
    displayimg['PCs']['Image']=ImageTk.PhotoImage(grayimg)

def changepca(event):
    global clusterdisplay,colordicesband,oldpcachoice
    global displaylabels

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
    generateimgplant()
    return

def savePCAimg(path,originfile,file):
    originpcabands=displaybandarray[currentfilename]['LabOstu']
    pcah,pcaw,pcac=originpcabands.shape
    # pcacount={}
    # keys=list(pcaboxdict.keys())
    # for item in keys:
    #     if pcaboxdict[item].get()=='1':
    #         pcacount.update({item:pcaboxdict[item]})
    # pcakeys=list(pcacount.keys())
    # tempband=np.zeros((pcah,pcaw,len(pcakeys)))
    # for i in range(len(pcakeys)):
    #     channel=int(pcakeys[i])-1
    #     tempband[:,:,i]=tempband[:,:,i]+originpcabands[:,:,channel]
    # displaylabels=np.mean(tempband,axis=2)
    # generateimgplant(displaylabels)
    # grayimg=(((displaylabels-displaylabels.min())/(displaylabels.max()-displaylabels.min()))*255.9).astype(np.uint8)
    # pyplt.imsave('k=1.png',displaylabels.astype('uint8'))
    # pyplt.imsave('k=1.png',grayimg)
    pcweights=pc_combine_up.get()-0.5
    tempband=np.zeros((pcah,pcaw))
    # pcsel=buttonvar.get()+2
    pcsel=buttonvar.get()
    if pcweights==0.0:
        tempband=tempband+originpcabands[:,:,pcsel]
    else:
        if pcweights<0.0:  #RGBPC1
            rgbpc=originpcabands[:,:,9]
        else:
            rgbpc=originpcabands[:,:,10]
        rgbpc=(rgbpc-rgbpc.min())*255/(rgbpc.max()-rgbpc.min())
        firstterm=abs(pcweights)*2*rgbpc
        colorpc=originpcabands[:,:,pcsel]
        colorpc=(colorpc-colorpc.min())*255/(colorpc.max()-colorpc.min())
        secondterm=(1-abs(pcweights)*2)*colorpc
        tempband=tempband+firstterm+secondterm
    displaylabels=np.copy(tempband)
    if displaylabels.min()<0:
        # if abs(displaylabels.min())<displaylabels.max():
        displaylabels=displaylabels-displaylabels.min()
    colorrange=displaylabels.max()-displaylabels.min()
    displaylabels=displaylabels*255/colorrange
    grayimg=Image.fromarray(displaylabels.astype('uint8'),'L')
    originheight,originwidth=Multigraybands[file].size
    origingray=grayimg.resize([originwidth,originheight],resample=Image.BILINEAR)
    origingray.save(path+'/'+originfile+'-PCAimg.png',"PNG")
    # addcolorstrip()
    return


def changecluster(event):
    global havecolorstrip,pre_checkbox,displaylabels,needreclass
    originpcabands=displaybandarray[currentfilename]['LabOstu']
    pcah,pcaw,pcac=originpcabands.shape
    pcweights=pc_combine_up.get()-0.5
    tempband=np.zeros((pcah,pcaw,1))
    # pcsel=buttonvar.get()+2
    pcsel=buttonvar.get()
    if pcweights==0.0:
        tempband[:,:,0]=tempband[:,:,0]+originpcabands[:,:,pcsel]
    else:
        if pcweights<0.0:  #RGBPC1
            rgbpc=originpcabands[:,:,9]
        else:
            rgbpc=originpcabands[:,:,10]
        rgbpc=(rgbpc-rgbpc.min())*255/(rgbpc.max()-rgbpc.min())
        firstterm=abs(pcweights)*2*rgbpc
        colorpc=originpcabands[:,:,pcsel]
        colorpc=(colorpc-colorpc.min())*255/(colorpc.max()-colorpc.min())
        secondterm=(1-abs(pcweights)*2)*colorpc
        tempband[:,:,0]=tempband[:,:,0]+firstterm+secondterm
    if int(kmeans.get())==1:
        displaylabels=np.mean(tempband,axis=2)
        generateimgplant(displaylabels)
        print('max',displaylabels.max())
        print('min',displaylabels.min())
        if displaylabels.min()<0:
            # if abs(displaylabels.min())<displaylabels.max():
            displaylabels=displaylabels-displaylabels.min()
        colorrange=displaylabels.max()-displaylabels.min()
        displaylabels=displaylabels*255/colorrange
        grayimg=Image.fromarray(displaylabels.astype('uint8'),'L')
        print('max',displaylabels.max())
        print('min',displaylabels.min())
        # grayimg.thumbnail((int(resizeshape[0]),int(resizeshape[1])),Image.ANTIALIAS)
        grayimg.save('k=1.png',"PNG")
        addcolorstrip()
        return
    else:
        # if kmeans.get() in clusterdisplay:
        #     displaylabels=clusterdisplay[kmeans.get()]
        #
        # else:
        #     havecolorstrip=False
        #     # choicelist=[]
        #     #reshapemodified_tif=np.zeros((displaybandarray[currentfilename]['LabOstu'].shape[0]*displaybandarray[currentfilename]['LabOstu'].shape[1],len(choicelist)))
        #     #displaylabels=kmeansclassify(choicelist,reshapemodified_tif)
        #     displaylabels=kmeansclassify()
        displaylabels=kmeansclassify()
        # changedisplayimg(imageframe,'Color Deviation')
        global checkboxdict
        keys=checkboxdict.keys()
        for key in keys:
            checkboxdict[key].set('0')
        generateimgplant('')
        # pyplt.imsave('allcolorindex.png',displaylabels)
        #kmeanscanvas.update()
        addcolorstrip()
        return

def changecluster_oldversion(event):
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
        print('max',displaylabels.max())
        print('min',displaylabels.min())
        if displaylabels.min()<0:
            # if abs(displaylabels.min())<displaylabels.max():
            displaylabels=displaylabels-displaylabels.min()
        colorrange=displaylabels.max()-displaylabels.min()
        displaylabels=displaylabels*255/colorrange

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
        #colortable=labeldict[itervalue]['colortable']
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
    minsize=min(image.size[0],image.size[1])
    suggsize=int(minsize**0.5)
    # if suggsize>22:
    #     suggsize=22
    # if suggsize<14:
    #     suggsize=14
    #suggsize=8
    #print('fontsize',suggsize)
    # suggsize=22
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
            uni=colortable[uni]
            if uni=='Ref':
                pixelloc = np.where(labels == 65535)
            else:
                pixelloc = np.where(labels == uni)
            try:
                ulx = min(pixelloc[1])
            except:
                print('no pixellloc[1] on uni=',uni)
                print('pixelloc =',pixelloc)
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
                    # canvastext = 'No label'
                    canvastext=uni
                canvastext=str(canvastext)
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
    ratio=findratio([height,width],[screenstd,screenstd])
    #if labels.shape[0]*labels.shape[1]<850*850:
    #    disimage=image.resize([int(labels.shape[1]*ratio),int(labels.shape[0]*ratio)],resample=Image.BILINEAR)
    #else:
    #    disimage=image.resize([int(labels.shape[1]/ratio),int(labels.shape[0]/ratio)],resample=Image.BILINEAR)
    print('show counting ratio',ratio)
    if height*width<screenstd*screenstd:
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
                    tempuni=colortable[uni]
                    if tempuni=='Ref':
                        pixelloc=np.where(labels==65535)
                    else:
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
                    #linepoints=[(currborder[1][i],currborder[0][i]),(currborder[1][j],currborder[0][j])]
                    #draw.line(linepoints,fill='yellow')
                    #points=linepixels(currborder[1][i],currborder[0][i],currborder[1][j],currborder[0][j])

                    lengthpoints=cal_kernelsize.bresenhamline(x0,y0,x1,y1)  #x0,y0,x1,y1
                    for point in lengthpoints:
                        if imgtypevar.get()=='0':
                            draw.point([int(point[0]),int(point[1])],fill='yellow')
                    # abovecenter=[]
                    # lowercenter=[]
                    # for i in range(len(currborder[0])):
                    #     for j in range(len(lengthpoints)):
                    #         if currborder[0][i]<lengthpoints[j][1]:
                    #             lowercenter.append((currborder[1][i],currborder[0][i])) #append(x,y)
                    #             break
                    #     loc=(currborder[1][i],currborder[0][i])
                    #     if loc not in abovecenter and loc not in lowercenter:
                    #         abovecenter.append(loc)
                    othodict={}
                    # widthdict={}
                    for i in range(len(currborder[0])):
                        for j in range(i+1,len(currborder[0])):
                            wx0=currborder[1][i]
                            wy0=currborder[0][i]
                            wx1=currborder[1][j]
                            wy1=currborder[0][j]
                            u1=x1-x0
                            u2=y1-y0
                            v1=wx1-wx0
                            v2=wy1-wy0
                            otho=abs(u1*v1+u2*v2)/(((u1**2+u2**2)**0.5)*(v1**2+v2**2)**0.5)
                            wlength=float((wx0-wx1)**2+(wy0-wy1)**2)**0.5
                            if otho<=0.13:
                                othodict.update({(wx0,wy0,wx1,wy1):wlength})

                    sortedwidth=sorted(othodict,key=othodict.get,reverse=True)
                    try:
                        topwidth=sortedwidth[0]
                    except:
                        continue
                    widepoints=cal_kernelsize.bresenhamline(topwidth[0],topwidth[1],topwidth[2],topwidth[3])
                    for point in widepoints:
                        if imgtypevar.get()=='0':
                            draw.point([int(point[0]),int(point[1])],fill='black')
                    width=othodict[topwidth]
                    print('width',width,'length',kernellength)
                    print('kernelwidth='+str(width*pixelmmratio))
                    print('kernellength='+str(kernellength*pixelmmratio))
                    #print('kernelwidth='+str(kernelwidth*pixelmmratio))
                    tempdict.update({colortable[uni]:[kernellength,width,pixelmmratio**2*len(pixelloc[0]),kernellength*pixelmmratio,width*pixelmmratio]})
                    #if uni in colortable:
                    canvastext = str(colortable[uni])
                    #else:
                    #    canvastext = uni
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

def export_opts(iterver):
    if proc_mode[proc_name].get()=='1':
        batchprocess.batch_exportpath()
        return
    opt_window=Toplevel()
    opt_window.geometry('300x150')
    opt_window.title('Export options')
    segmentresult=IntVar()
    croppedimage=IntVar()
    hundred=IntVar()
    two_hundred=IntVar()
    originsize=IntVar()
    checkframe=Frame(opt_window)
    checkframe.pack()
    Checkbutton(checkframe,text='Export Segment outlook images (big size)',variable=segmentresult).pack(padx=10,pady=10)
    Checkbutton(checkframe,text='Export Cropped images for ML dataset',variable=croppedimage).pack(padx=10,pady=10)
    Checkbutton(checkframe,text='100x100',variable=hundred).pack(padx=10,pady=10)
    Checkbutton(checkframe,text='224x224',variable=two_hundred).pack(padx=10,pady=10)
    Checkbutton(checkframe,text='Origin',variable=originsize).pack(padx=10,pady=10)
    Button(checkframe,text='Export',command=partial(export_result,opt_window,segmentresult,croppedimage,hundred,two_hundred,originsize,iterver)).pack(padx=10,pady=10)
    opt_window.transient(root)
    opt_window.grab_set()

def removeisland(cropband,uin,cropimg):
    tempband = cropband - uin
    island = np.where((tempband!=0) & (tempband!= 0 -uin))
    islandminx,islandmaxx=np.min(island[1]),np.max(island[1])
    islandminy,islandmaxy=np.min(island[0]),np.max(island[0])
    islandarea=cropband[islandminy:islandmaxy,islandminx:islandmaxx]
    # islandarea=islandarea.astype(np.uint8)
    # islandcontour,_=cv2.findContours(cropimg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # print(islandcontour)
    background = np.where(cropband==0)
    # cimg = np.zeros_like(islandarea)
    # cv2.drawContours(cimg,islandcontour,0,color=255, thickness=-1)
    # dummyimg = np.zeros_like(cropband)
    # dummyimg[islandminy:islandmaxy,islandminx:islandmaxx]=cimg
    # island = np.where(dummyimg==255)
    background = cropimg[background]
    backgroundavg=np.mean(background,axis=1)

    # backgroundperc=np.percentile(backgroundavg,50)
    backgroundperc = np.percentile(backgroundavg,25)
    print(background, backgroundavg,backgroundperc)
    backgroundavg = np.where(backgroundavg<backgroundperc)
    print(backgroundavg,'axis 0 length',len(backgroundavg[0]))
    if len(backgroundavg[0])>0:
        background = background[backgroundavg[0][0]]
        cropimg[island] = background
    # cropimg[island] = [0,0,0]
    return cropimg

def checkisland(cropband,uin):
    tempband=cropband-uin
    uniqband=np.unique(tempband)
    if uniqband.shape[0]>2:
        return True
    return False
    # nonzero=np.where((tempband != 0) & (tempband != -uin))
    # print(nonzero)

def export_result(popup,segmentoutputopt,cropimageopt,hundredsize,two_hundredsize,originsize,iterver):
    global batch
    suggsize=8
    print('fontsize',suggsize)
    smallfont=ImageFont.truetype('cmb10.ttf',size=suggsize)
    files=multi_results.keys()
    path=filedialog.askdirectory()
    popup.destroy()
    root.update()
    if segmentoutputopt.get()>0:
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
        originimg_crop=cv2.imread(file)
        uniquelabels=list(colortable.keys())
        # originheight,originwidth=Multigraybands[file].size
        originheight, originwidth=currentlabels.shape
        ratio=int(findratio([512,512],[labels.shape[0],labels.shape[1]]))
        if labels.shape[0]!=originheight and labels.shape[1]!=originwidth:
            if labels.shape[0]<512:
                cache=(np.zeros((labels.shape[0]*ratio,labels.shape[1]*ratio)),{"f":int(ratio),"stride":int(ratio)})
                convband=tkintercorestat.pool_backward(labels,cache)
            else:
                if labels.shape[0]>512:
                    convband=cv2.resize(labels,(512,512),interpolation=cv2.INTER_LINEAR)
                else:
                    if labels.shape[0]==512:
                        convband=np.copy(labels)
        else:
            convband=np.copy(labels)
        locfilename=path+'/'+originfile+'-pixellocs.csv'
        with open(locfilename,mode='w') as f:
            csvwriter=csv.writer(f)
            rowcontent=['id','locs']
            csvwriter.writerow(rowcontent)
            for uni in uniquelabels:
                if uni!=0:
                    tempuni=colortable[uni]
                    if tempuni=='Ref':
                        pixelloc = np.where(convband == 65535)
                    else:
                        pixelloc = np.where(convband == float(uni))
                    rowcontent=[colortable[uni]]
                    rowcontent=rowcontent+list(pixelloc[0])
                    csvwriter.writerow(rowcontent)
                    rowcontent=[colortable[uni]]
                    rowcontent=rowcontent+list(pixelloc[1])
                    csvwriter.writerow(rowcontent)

            f.close()
        #from spectral import imshow, view_cube
        '''hyperspectral img process'''
        # import spectral.io.envi as envi
        '''For image crop version below'''
        lesszeroonefive = []
        if cropimageopt.get()>0:
            import cropimg_extraction
            thresholds = [cal_xvalue(linelocs[0]), cal_xvalue(linelocs[1])]
            minthres = min(thresholds)
            maxthres = max(thresholds)
            lwthresholds = [cal_yvalue(linelocs[2]), cal_yvalue(linelocs[3])]
            maxlw = max(lwthresholds)
            minlw = min(lwthresholds)
            print('thresholds',thresholds,'lwthresholds',lwthresholds)
            imgrsc = cv2.imread(file, flags=cv2.IMREAD_ANYCOLOR)
            imgrsc = cv2.resize(imgrsc, (originwidth,originheight),interpolation=cv2.INTER_LINEAR)
            cropratio=findratio((originheight,originwidth),(labels.shape[0],labels.shape[1]))
            if cropratio>1 and originheight*originwidth!=labels.shape[0]*labels.shape[1]:
                cache = (np.zeros((originheight,originwidth)),
                     {"f": int(cropratio), "stride": int(cropratio)})
                originconvband = tkintercorestat.pool_backward(labels, cache)
            else:
                originconvband=np.copy(labels)
            # cv2.imwrite(os.path.join(path, originfile + '_before.png'), originconvband)
            labelsegfile=os.path.join(path,originfile+'_cropimage_label.csv')
            with open(labelsegfile,mode='w') as f:
                csvwriter=csv.writer(f)
                # rowcontent=['id','locs']
                rowcontent=['index','i','j','filename','label']
                csvwriter.writerow(rowcontent)
            #     result_ref=envi.open(head_tail[0]+'/'+originfile+'/results/REFLECTANCE_'+originfile+'.hdr', head_tail[0]+'/'+originfile+'/results/REFLECTANCE_'+originfile+'.dat')
            #     result_nparr=np.array(result_ref.load())
            #     corrected_nparr=np.copy(result_nparr)
                index=1
                cropfilenames=[]
                kernelpixsize=[]
                for uni in uniquelabels:
                    if uni!=0:
                        tempuni=colortable[uni]
                        if tempuni=='Ref':
                            # pixelloc = np.where(convband == 65535)
                            originpixelloc=np.where(originconvband == 65535)
                            # labelpixelloc = np.where(labels == 65535)
                        else:
                            # pixelloc = np.where(convband == float(uni))
                            originpixelloc = np.where(originconvband == float(uni))
                            # labelpixelloc = np.where(labels == float(uni))
                        # kernelval=corrected_nparr[pixelloc]
                        # nirs=np.mean(kernelval,axis=0)
            #             print('nirs 170',nirs[170])
            #             if nirs[170]<0.15:
            #                 lesszeroonefive.append(uni)
                        try:
                            # ulx = min(pixelloc[1])
                            ulx = min(originpixelloc[1])
                            # labelulx = min(labelpixelloc[1])
                        except:
                            print('no pixellloc[1] on uni=', uni)
                            print('pixelloc =', originpixelloc)
                            continue
                        uly = min(originpixelloc[0])
                        # labeluly = min(labelpixelloc[0])
                        rlx = max(originpixelloc[1])
                        # labelrlx = max(labelpixelloc[1])
                        rly = max(originpixelloc[0])
                        # labelrly = max(labelpixelloc[0])
                        width=rlx-ulx+1
                        height=rly-uly+1
                        # labelwidth=labelrlx-labelulx+1
                        # labelheight=labelrly-labeluly+1
                        originbkgloc=np.where(originconvband==0)
                        blx=min(originbkgloc[1])
                        bly=min(originbkgloc[0])
                        # cropimage = imgrsc[uly:rly, ulx:rlx]
                        print('width,height',width,height,'pixelsize',len(originpixelloc[0]))
                        print('output to cropimg', path, originfile + '_crop_' + str(int(uni)) + '.png')
                        if max(height/width,width/height)>1.05:
                            # edgelen = max(height, width)
                            if height>width: #vertical
                                addlen=int((height-width)/2)
                                # labeladdlen=int((labelheight-labelwidth)/2)
                                newulx = (ulx - addlen) if (ulx-addlen)>0 else ulx
                                # cropband = labels[labeluly:labelrly,labelulx:(labelrlx+labeladdlen)]
                                # cropband = originconvband[uly:rly, ulx:(rlx + addlen)]
                                cropband = originconvband[uly:rly,ulx:rlx]
                                # cropimage = imgrsc[uly:rly,ulx:rlx]
                                cropimage = imgrsc[uly:rly, newulx:(rlx + addlen)]

                                '''add dummy background to cover other kernels
                                # if checkisland(cropband,uni)==True:
                                #     cropimage = removeisland(cropband,uni,cropimage)
                                #     # cropimgband=currentlabels[uly:rly, ulx:(rlx + addlen)]
                                #     # cropimg = cropimg_extraction.batch_cropimg(originfile+'_crop_'+str(int(uni))+'.png',
                                #     #                                            path,0,
                                #     #                                            len(originpixelloc[0]),cropimgband)
                                #     # cropimg.process()![](../Downloads/OneDrive_1_5-23-2022/gridfree/10_202-18C_crop_1.png)
                                #     # print(cropimg.batch_results)
                                #     # continue
                                #     # print('have other kernels, V')
                                # 
                                # dummyimg=np.zeros([height-1,height-1,3])
                                # dummyimg[:,addlen:addlen+width-1]=cropimage
                                # background=np.where(cropband==0)
                                # background=cropimage[background]
                                # backgroundavg=np.mean(background,axis=1)
                                # try:
                                #     backgroundproc=np.percentile(backgroundavg,25)
                                #     # backgroundproc = np.median(backgroundavg)
                                #     print(backgroundproc,backgroundavg)
                                #     backgroundavg=np.where(backgroundavg<backgroundproc)
                                #     if len(backgroundavg[0])>0:
                                #         background=background[backgroundavg[0][0]]
                                #         dummyimg[:,:addlen]=background
                                #         dummyimg[:,addlen+width-1:]=background
                                # except:
                                #     pass
                                # cropimage=np.copy(dummyimg)
                                '''

                            else:
                                addlen=int((width-height)/2)
                                newuly = (uly - addlen) if (uly - addlen)>0 else uly
                                # cropband = labels[labeluly:(labelrly+labeladdlen),labelulx:labelrlx]
                                # cropband = originconvband[uly:(rly+addlen),ulx:rlx]
                                cropimage = imgrsc[uly:(rly+addlen),ulx:rlx]
                                '''add dummy background to cover other kernels
                                cropband = originconvband[uly:rly, ulx:rlx]
                                cropimage = imgrsc[uly:rly, ulx:rlx]
                                if checkisland(cropband,uni)==True:
                                    # print('have other kernel, H')
                                    cropimage = removeisland(cropband, uni, cropimage)
                                    # cropimg = cropimg_extraction.batch_cropimg(
                                    #     originfile + '_crop_' + str(int(uni)) + '.png',
                                    #     path, len(originpixelloc[0]), len(originpixelloc[0] + 500))
                                    # cropimg.process()![](../germination_NIR/band_img_output/spring/band_01_germ_1.png)
                                    # continue

                                dummyimg = np.zeros([width - 1, width - 1, 3])
                                dummyimg[addlen:addlen+height-1, :] = cropimage
                                background = np.where(cropband == 0)
                                background = cropimage[background]
                                backgroundavg = np.mean(background, axis=1)
                                try:
                                    backgroundproc = np.percentile(backgroundavg,25)
                                    backgroundavg = np.where(backgroundavg <backgroundproc)
                                    if len(backgroundavg[0])>0:
                                        background = background[backgroundavg[0][0]]
                                        dummyimg[:addlen, :] = background
                                        dummyimg[addlen + height - 1:, :] = background
                                except:
                                    pass
                                cropimage = np.copy(dummyimg)
                                '''
                        else:
                            # cropband = labels[labeluly:labelrly, labelulx:labelrlx]
                            # cropband = originconvband[uly:rly, ulx:rlx]
                            cropimage = imgrsc[uly:rly, ulx:rlx]
                            # if checkisland(cropband, uni) == True:
                            #     cropimage = removeisland(cropband, uni, cropimage)
                                # cropimg = cropimg_extraction.batch_cropimg(
                                #     originfile + '_crop_' + str(int(uni)) + '.png',
                                #     path, len(originpixelloc[0]), len(originpixelloc[0] + 500))
                                # cropimg.process()
                                # continue
                        if hundredsize.get() > 0:
                            cropimage = cv2.resize(cropimage, (100, 100), interpolation=cv2.INTER_LINEAR)
                        if two_hundredsize.get() > 0:
                            cropimage = cv2.resize(cropimage, (224, 224), interpolation=cv2.INTER_LINEAR)
                        # cropimage = cv2.resize(cropimage,(224,224),interpolation=cv2.INTER_LINEAR)
                        cropimgoutput=os.path.join(path,originfile+'_crop_'+str(int(uni))+'.png')
                        cropfilenames.append(cropimgoutput)
                        kernelpixsize.append(len(originpixelloc[0]))
                        cv2.imwrite(cropimgoutput,cropimage)

                        rowcontent=[index,0,0,originfile+'_crop_'+str(int(uni))+'.png',0]
                        csvwriter.writerow(rowcontent)
                        index+=1
                        # rowcontent=[colortable[uni]]
                        # rowcontent=rowcontent+list(pixelloc[0])
                        # csvwriter.writerow(rowcontent)
                        # rowcontent=[colortable[uni]]
                        # rowcontent=rowcontent+list(pixelloc[1])
                        # csvwriter.writerow(rowcontent)

                print(cropfilenames)
                print(kernelpixsize)
                # import cropimg_extraction
                # minkernelpix=min(kernelpixsize)
                # maxkernelpix=max(kernelpixsize)
                # for filename in cropfilenames:
                #     cropapp=cropimg_extraction.batch_cropimg(filename,path,minkernelpix,maxkernelpix)
                #     cropapp.process()

                f.close()
        print(lesszeroonefive)
        '''end'''

        if segmentoutputopt.get()>0:
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
                        #uni=colortable[uni]
                        tempuni=colortable[uni]
                        if tempuni=='Ref':
                            pixelloc = np.where(labels == 65535)
                        else:
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
                        othodict={}
                        # widthdict={}
                        for i in range(len(currborder[0])):
                            for j in range(i+1,len(currborder[0])):
                                wx0=currborder[1][i]
                                wy0=currborder[0][i]
                                wx1=currborder[1][j]
                                wy1=currborder[0][j]
                                u1=x1-x0
                                u2=y1-y0
                                v1=wx1-wx0
                                v2=wy1-wy0
                                otho=abs(u1*v1+u2*v2)/(((u1**2+u2**2)**0.5)*(v1**2+v2**2)**0.5)
                                wlength=float((wx0-wx1)**2+(wy0-wy1)**2)**0.5
                                if otho<=0.13:
                                    othodict.update({(wx0,wy0,wx1,wy1):wlength})

                        sortedwidth=sorted(othodict,key=othodict.get,reverse=True)
                        try:
                            topwidth=sortedwidth[0]
                        except:
                            continue
                        widepoints=cal_kernelsize.bresenhamline(topwidth[0],topwidth[1],topwidth[2],topwidth[3])
                        for point in widepoints:
                            if imgtypevar.get()=='0':
                                draw.point([int(point[0]),int(point[1])],fill='black')
                        width=othodict[topwidth]

                        print('width',width,'length',kernellength)
                        print('kernelwidth='+str(width*pixelmmratio))
                        print('kernellength='+str(kernellength*pixelmmratio))
                        #print('kernelwidth='+str(kernelwidth*pixelmmratio))
                        tempdict.update({colortable[uni]:[kernellength,width,pixelmmratio**2*len(pixelloc[0]),kernellength*pixelmmratio,width*pixelmmratio]})
                        #if uni in colortable:
                        canvastext = str(colortable[uni])
                       # else:
                            # canvastext = 'No label'
                        #    canvastext = uni
                        if imgtypevar.get()=='0':
                            if uni in lesszeroonefive:
                                draw.text((midx-1, midy+1), text=canvastext, font=smallfont, fill='white')
                                draw.text((midx+1, midy+1), text=canvastext, font=smallfont, fill='white')
                                draw.text((midx-1, midy-1), text=canvastext, font=smallfont, fill='white')
                                draw.text((midx+1, midy-1), text=canvastext, font=smallfont, fill='white')
                                #draw.text((midx,midy),text=canvastext,font=font,fill=(141,2,31,0))
                                draw.text((midx,midy),text=canvastext,font=smallfont,fill='red')
                            else:
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
                # temppcabands=np.zeros((originpcabands[file].shape[0],len(batch['PCs'])))
                # temppcabands=np.zeros(originpcabands[file].shape[0],1)
                # for i in range(len(batch['PCs'])):
                #     temppcabands[:,i]=temppcabands[:,i]+originpcabands[file][:,batch['PCs'][i]-1]
                pcabands=np.copy(displaypclabels)
                # pcabands=pcabands.reshape((originheight,originwidth))
                # pcabands=pcabands.reshape(displayfea_l,displayfea_w)

                colorindices_cal(file)
                colorindicekeys=list(colorindicearray[file].keys())
                colorindicelist=[ 0 for i in range(len(colorindicekeys)*3)]


                datatable={}
                origindata={}
                for key in indicekeys:
                    data=originbandarray[file][key]
                    data=data.tolist()
                    tempdict={key:data}
                    origindata.update(tempdict)
                    print(key)

                for key in colorindicekeys:
                    data=colorindicearray[file][key]
                    data=data.tolist()
                    tempdict={key:data}
                    origindata.update(tempdict)
                    print(key)
                # for uni in colortable:
                print(uniquelabels)
                print('len uniquelabels',len(uniquelabels))
                for uni in uniquelabels:
                    print(uni,colortable[uni])
                    uni=colortable[uni]
                    if uni=='Ref':
                        uniloc=np.where(labels==65535)
                        smalluniloc=np.where(originrestoredband==65535)
                    else:
                        uniloc=np.where(labels==float(uni))
                        smalluniloc=np.where(originrestoredband==uni)
                    if len(uniloc)==0 or len(uniloc[1])==0:
                        print('no uniloc\n')
                        print(uniloc[0],uniloc[1])
                        continue

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
                    # tempdict={colortable[uni]:templist+indeclist+colorindicelist+pcalist}  #NIR,Redeyes,R,G,B,NDVI,area
                    tempdict={uni:templist+indeclist+colorindicelist+pcalist}  #NIR,Redeyes,R,G,B,NDVI,area
                    print(tempdict)
                    indicekeys=list(origindata.keys())
                    for ki in range(len(indicekeys)):
                        originNDVI=origindata[indicekeys[ki]]
                        print('originNDVI size',len(originNDVI),len(originNDVI[0]))
                        pixellist=[]
                        for k in range(len(uniloc[0])):
                            #print(uniloc[0][k],uniloc[1][k])
                            try:
                                # tempdict[colortable[uni]][6+ki*3]+=originNDVI[uniloc[0][k]][uniloc[1][k]]
                                tempdict[uni][6+ki*3]+=originNDVI[uniloc[0][k]][uniloc[1][k]]
                            except IndexError:
                                print(uniloc[0][k],uniloc[1][k])
                            # tempdict[colortable[uni]][7+ki*3]+=originNDVI[uniloc[0][k]][uniloc[1][k]]
                            tempdict[uni][7+ki*3]+=originNDVI[uniloc[0][k]][uniloc[1][k]]
                            pixellist.append(originNDVI[uniloc[0][k]][uniloc[1][k]])
                        # tempdict[colortable[uni]][ki*3+6]=tempdict[colortable[uni]][ki*3+6]/amount
                        # tempdict[colortable[uni]][ki*3+8]=np.std(pixellist)
                        tempdict[uni][ki*3+6]=tempdict[uni][ki*3+6]/amount
                        tempdict[uni][ki*3+8]=np.std(pixellist)
                    pixellist=[]
                    for k in range(len(uniloc[0])):
                        try:
                            # tempdict[colortable[uni]][-2]+=pcabands[uniloc[0][k]][uniloc[1][k]]
                            tempdict[uni][-2]+=pcabands[uniloc[0][k]][uniloc[1][k]]
                        except IndexError:
                            print(uniloc[0][k],uniloc[1][k])
                        # tempdict[colortable[uni]][-3]+=pcabands[uniloc[0][k]][uniloc[1][k]]
                        tempdict[uni][-3]+=pcabands[uniloc[0][k]][uniloc[1][k]]
                        pixellist.append(pcabands[uniloc[0][k]][uniloc[1][k]])
                        # tempdict[colortable[uni]][-3]=tempdict[colortable[uni]][-3]/amount
                        # tempdict[colortable[uni]][-1]=np.std(pixellist)
                        tempdict[uni][-3]=tempdict[uni][-3]/amount
                        tempdict[uni][-1]=np.std(pixellist)
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
    batch['drawpolygon'] = [int(drawpolygon)]
    batch['filtercoord'] = pcfilter.copy()
    batch['filterbackground']=[displayimg['Origin']['Size'][0],displayimg['Origin']['Size'][1]]
    batch['segmentoutputopt'] = [segmentoutputopt.get()]
    batch['cropimageopt'] = [cropimageopt.get()]

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




def resegment(thresholds=[],lwthresholds=[]):
    global loccanvas,maxx,minx,maxy,miny,linelocs,bins,ybins,reseglabels,figcanvas,refvar,refsubframe,panelA
    global labelplotmap,figdotlist,multi_results
    global batch
    global outputimgdict,outputimgbands
    figcanvas.unbind('<Any-Enter>')
    figcanvas.unbind('<Any-Leave>')
    figcanvas.unbind('<Button-1>')
    figcanvas.unbind('<B1-Motion>')
    figcanvas.unbind('<Shift-Button-1>')
    figcanvas.delete(ALL)
    #panelA.unbind('<Button-1>')
    #refvar.set('0')
    #for widget in refsubframe.winfo_children():
    #    widget.config(state=DISABLED)
    if len(thresholds)==0:
        thresholds=[cal_xvalue(linelocs[0]),cal_xvalue(linelocs[1])]
    minthres=min(thresholds)
    maxthres=max(thresholds)
    if len(lwthresholds)==0:
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
    if refarea is None:
        retrivearea=np.where(labels==65535)
        if len(retrivearea[1])>0:
            ulx,uly=min(retrivearea[1]),min(retrivearea[0])
            rlx,rly=max(retrivearea[1]),max(retrivearea[0])
            rtl=rly-uly
            rtw=rlx-ulx
            rtd=(rtl**2+rtw**2)**0.5
            rtarea=len(retrivearea[0])
            print('rtarea,rtl,rtw,rtd',rtarea,rtl,rtw,rtd)
            if rtarea>maxthres:
                maxthres=rtarea
            if rtd>maxlw:
                maxlw=rtd
            if rtarea<minthres:
                minthres=rtarea
            if rtd<minlw:
                minlw=rtd
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
        outputdisplay,outputimg,small_seg=showcounting(tup,False,True,True)
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
    # displayfig()

    # data=[]
    # uniquelabels=list(colortable.keys())
    # lenwid=[]
    # lenlist=[]
    # widlist=[]
    # labelplotmap={}
    # templabelplotmap={}
    # unitable=[]
    # for uni in uniquelabels:
    #     if uni!=0:
    #         pixelloc = np.where(reseglabels == uni)
    #         try:
    #             ulx = min(pixelloc[1])
    #         except:
    #             continue
    #         uly = min(pixelloc[0])
    #         rlx = max(pixelloc[1])
    #         rly = max(pixelloc[0])
    #         length=rly-uly
    #         width=rlx-ulx
    #         lenwid.append((length+width))
    #         lenlist.append(length)
    #         widlist.append(width)
    #         data.append(len(pixelloc[0]))
    #         unitable.append(uni)
    #         # templabelplotmap.update({(len(pixelloc[0]),length+width):uni})
    # residual,area=lm_method.lm_method(lenlist,widlist,data)
    # lenwid=list(residual)
    # data=list(area)
    # for i in range(len(unitable)):
    #     templabelplotmap.update({(data[i],lenwid[i]):unitable[i]})
    # miny=min(lenwid)
    # maxy=max(lenwid)
    # minx=min(data)
    # maxx=max(data)
    # binwidth=(maxx-minx)/10
    # ybinwidth=(maxy-miny)/10
    # bin_edges=[]
    # y_bins=[]
    # for i in range(0,11):
    #     bin_edges.append(minx+i*binwidth)
    # for i in range(0,11):
    #     y_bins.append(miny+i*ybinwidth)
    # #bin_edges.append(maxx)
    # #bin_edges.append(maxx+binwidth)
    # #y_bins.append(maxy)
    # #y_bins.append(maxy+ybinwidth)
    # plotdata=[]
    # for i in range(len(data)):
    #     plotdata.append((data[i],lenwid[i]))
    # scaledDatalist=[]
    # try:
    #     x_scalefactor=300/(maxx-minx)
    # except:
    #     return
    # y_scalefactor=250/(maxy-miny)
    # for (x,y) in plotdata:
    #     xval=50+(x-minx)*x_scalefactor+50
    #     yval=300-(y-miny)*y_scalefactor+25
    #     scaledDatalist.append((int(xval),int(yval)))
    # for key in templabelplotmap:
    #     x=key[0]
    #     y=key[1]
    #     xval=50+(x-minx)*x_scalefactor+50
    #     yval=300-(y-miny)*y_scalefactor+25
    #     unilabel=templabelplotmap[key]
    #     labelplotmap.update({(int(xval),int(yval)):unilabel})
    # figdotlist={}
    # axistest.drawdots(25+50,325+25,375+50,25+25,bin_edges,y_bins,scaledDatalist,figcanvas,figdotlist)
    #
    #
    # #loccanvas=figcanvas
    # #minx=25
    # #maxx=375
    # #maxy=325
    # #miny=25
    # #linelocs=[25+12,375-12,325-12,25+12]
    # #linelocs=[25+12,375-12,25+12,325-12]
    # linelocs=[75+12,425-12,350-12,50+12]
    # bins=bin_edges
    # ybins=y_bins
    #
    # figcanvas.bind('<Any-Enter>',item_enter)
    # figcanvas.bind('<Any-Leave>',item_leave)
    # figcanvas.bind('<Button-1>',item_start_drag)
    # figcanvas.bind('<B1-Motion>',item_drag)
    # #figcanvas.bind('<Shift-Button-1>',item_multiselect)
    # if refarea is not None:
    #     reseglabels[refarea]=65535
    #
    # pcasel=[]
    # pcakeys=list(pcaboxdict.keys())
    # for i in range(len(pcakeys)):
    #     currvar=pcaboxdict[pcakeys[i]].get()
    #     if currvar=='1':
    #         pcasel.append(i+1)
    # kchoice=[]
    # kchoicekeys=list(checkboxdict.keys())
    # for i in range(len(kchoicekeys)):
    #     currvar=checkboxdict[kchoicekeys[i]].get()
    #     if currvar=='1':
    #         kchoice.append(i+1)
    # batch['PCs']=pcasel.copy()
    # batch['Kmeans']=[int(kmeans.get())]
    # batch['Kmeans_sel']=kchoice.copy()
    # batch['Area_max']=[maxthres]
    # batch['Area_min']=[minthres]
    # # batch['L+W_max']=[maxlw]
    # # batch['L+W_min']=[minlw]
    # print(batch)


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
    global dotflash
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

def displayfig():
    global loccanvas,maxx,minx,maxy,miny,linelocs,bins,ybins,figcanvas
    global labelplotmap,resviewframe
    global figdotlist
    data=[]

    originlabeldict=multi_results[currentfilename][0]

    colortable=originlabeldict['iter0']['colortable']
    uniquelabels=list(colortable.keys())

    lenwid=[]
    lenlist=[]
    widlist=[]
    for widget in resviewframe.winfo_children():
        widget.pack_forget()
    figcanvas.pack()
    figcanvas.delete(ALL)
    labelplotmap={}

    templabelplotmap={}
    unitable=[]
    for uni in uniquelabels:
        if uni!=0:
            uni=colortable[uni]
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
    figcanvas.bind('<Shift-Button-1>',item_multiselect)
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
    # refvar.set('0')
    for widget in refsubframe.winfo_children():
        #widget.config(state=DISABLED)
        widget.config(state=NORMAL)
    outputbutton.config(state=NORMAL)
    #resegbutton.config(state=NORMAL)
    # pcasel=[]
    # pcakeys=list(pcaboxdict.keys())
    # for i in range(len(pcakeys)):
    #     currvar=pcaboxdict[pcakeys[i]].get()
    #     if currvar=='1':
    #         pcasel.append(i+1)
    kchoice=[]
    kchoicekeys=list(checkboxdict.keys())
    for i in range(len(kchoicekeys)):
        currvar=checkboxdict[kchoicekeys[i]].get()
        if currvar=='1':
            kchoice.append(i+1)
    pcasel=[]
    pcasel.append(pc_combine_up.get()-0.5)
    batch['PCweight']=pcasel.copy()
    batch['PCsel']=[buttonvar.get()+1]
    batch['Kmeans']=[int(kmeans.get())]
    batch['Kmeans_sel']=kchoice.copy()
    print(batch)


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
    print('nonzeroratio=',nonzeroratio)
    batch['nonzero']=[nonzeroratio]
    #nonzeroratio=float(nonzeros)/(currentlabels.shape[0]*currentlabels.shape[1])
    dealpixel=nonzeroratio*currentlabels.shape[0]*currentlabels.shape[1]
    ratio=1
    # if selarea.get()=='1':
    selareadim=app.getinfo(rects[1])
    global selareapos,originselarea,binaryselarea
    if selareadim!=[0,0,1,1] and selareadim!=[] and selareadim!=selareapos:
        selareapos=selareadim
    if selareapos!=[0,0,1,1] and len(selareapos)>0 and selview!='' and selview!='Color Deviation':
        # selareadim=app.getinfo(rects[1])
        npfilter=np.zeros((displayimg['Origin']['Size'][0],displayimg['Origin']['Size'][1]))
        print("npfilter shape:",npfilter.shape)
        filter=Image.fromarray(npfilter)
        draw=ImageDraw.Draw(filter)
        if drawpolygon==False:
            draw.ellipse(selareapos,fill='red')
        else:
            draw.polygon(selareapos,fill='red')
        filter=np.array(filter)


        # start=list(selareapos)[:2]
        # end=list(selareapos)[2:]
        # lx,ly,rx,ry=int(min(start[0],end[0])),int(min(start[1],end[1])),int(max(start[0],end[0])),int(max(start[1],end[1]))
        # filter[:,lx:rx+1]=1
        # for i in range(0,ly):
        #     filter[i,:]=0
        # for i in range(ry+1,displayimg['Origin']['Size'][0]):
        #     filter[i,:]=0
        filter=np.divide(filter,np.max(filter))
        originselarea=False
        binaryselarea=False
        # filter=np.where(filter==max(filter),1,0)
    else:
        filter=np.ones((displayimg['Origin']['Size'][0],displayimg['Origin']['Size'][1]))
    filter=cv2.resize(filter,(currentlabels.shape[1],currentlabels.shape[0]),interpolation=cv2.INTER_LINEAR)
    selareapos=[]

    print('deal pixel',dealpixel)
    if dealpixel<512000:
        workingimg=np.copy(currentlabels)
        # if selarea.get()=='1':
        workingimg=np.multiply(workingimg,filter)
    else:
        if nonzeroratio<=0.2:# and nonzeroratio>=0.1:
            ratio=findratio([currentlabels.shape[0],currentlabels.shape[1]],[1600,1600])
            print('ratio to wkimg',ratio)
            # if dealpixel<512000 or currentlabels.shape[0]*currentlabels.shape[1]<=1600*1600:
            #     workingimg=np.copy(currentlabels)
            # else:
                # if currentlabels.shape[0]*currentlabels.shape[1]>1600*1600:
            workingimg=cv2.resize(currentlabels,(int(currentlabels.shape[1]/ratio),int(currentlabels.shape[0]/ratio)),interpolation=cv2.INTER_LINEAR)
            # if selarea.get()=='1':
            filter=cv2.resize(filter,(int(currentlabels.shape[1]/ratio),int(currentlabels.shape[0]/ratio)),interpolation=cv2.INTER_LINEAR)
            workingimg=np.multiply(workingimg,filter)
                # else:
                #     #ratio=1
                #     #print('nonzeroratio',ratio)
                #     workingimg=np.copy(currentlabels)
            segmentratio=0
        else:
            # if dealpixel>512000:
            if currentlabels.shape[0]*currentlabels.shape[1]>screenstd*screenstd:
                segmentratio=findratio([currentlabels.shape[0],currentlabels.shape[1]],[screenstd,screenstd])
                if segmentratio<2:
                    segmentratio=2
                workingimg=cv2.resize(currentlabels,(int(currentlabels.shape[1]/segmentratio),int(currentlabels.shape[0]/segmentratio)),interpolation=cv2.INTER_LINEAR)
                # if selarea.get()=='1':
                filter=cv2.resize(filter,(int(currentlabels.shape[1]/segmentratio),int(currentlabels.shape[0]/segmentratio)),interpolation=cv2.INTER_LINEAR)
                # filter=cv2.resize(filter,workingimg.shape[1],workingimg.shape[2],interpolation=cv2.INTER_LINEAR)
                workingimg=np.multiply(workingimg,filter)

            # else:
            #     segmentratio=1
            #     #print('ratio',ratio)
            #     workingimg=np.copy(currentlabels)
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
        outputdisplay,outputimg,smallset=showcounting(tup,False,True,True)
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
    # panelA.bind('<Button-1>',lambda event,arg=processlabel:customcoin(event,processlabel,tempband))
    # panelA.bind('<Shift-Button-1>',customcoin_multi)
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
    global coinbox_list,minflash,coinbox
    global dotflash,figcanvas

    x=event.x
    y=event.y
    # multiselectitems=[]
    if len(minflash)>0:
        for i in range(len(minflash)):
            panelA.delete(minflash.pop(0))
    if len(dotflash)>0:
        for i in range(len(dotflash)):
            figcanvas.delete(dotflash.pop(0))
    panelA.delete(coinbox)
    tempband=np.copy(convband)
    print(tempband.shape)
    coinlabel=tempband[y,x]
    print('coinlabel',coinlabel,'x',x,'y',y)
    if coinlabel==0:
        multiselectitems=[]
        if len(coinbox_list)>0:
            for i in range(len(coinbox_list)):
                panelA.delete(coinbox_list.pop(0))
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
        coinbox_list.append(a)
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
        labelplotmapkeys=getkeys(labelplotmap)
        for mapkey in labelplotmapkeys:
            k=mapkey[0]
            v=mapkey[1]
            templabel=labelplotmap[mapkey]
            if templabel in multiselectitems:
                xval=k
                yval=v
                print('lw',yval,'area',xval)
                plotflash(yval,xval,'Orange','Orange')
                # break

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

def del_reflabel():
    global reseglabels,panelA,loccanvas,linelocs,bins,ybins,figcanvas,maxx,minx,maxy,miny,refvar,refsubframe
    global labelplotmap,multiselectitems,dotflash,minflash,coinbox_list,reflabel
    processlabel=np.copy(reseglabels)
    refarea=np.where(processlabel==reflabel)
    print('reflabel to delete',reflabel)
    reseglabels[refarea]=0
    reflabel=0
    delselarea=app.getinfo(delrects[1])

    # if len(minflash)>0:
    #     print('delete minflash')
    #     for i in range(len(minflash)):
    #         panelA.delete(minflash.pop(0))
    # if len(dotflash)>0:
    #     print('delete dotflash')
    #     for i in range(len(dotflash)):
    #         figcanvas.delete(dotflash.pop(0))
    # if len(coinbox_list)>0:
    #     print('del coinbox_list')
    #     for i in range(len(coinbox_list)):
    #         panelA.delete(coinbox_list.pop(0))
    if len(multiselectitems)>0:
        print('del multiselection items',len(multiselectitems))
        for i in range(len(multiselectitems)):
            refarea=np.where(processlabel==multiselectitems.pop(0))
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
    if len(outsizethreshold)>0:
        print('del outsizethreshold',len(outsizethreshold))
    for i in range(len(outsizethreshold)):
        deletlabel=outsizethreshold[i]
        refarea=np.where(processlabel==deletlabel)
        reseglabels[refarea]=0

    if delselarea!=[]:
        print('delselarea',delrects[1],delselarea)
        npfilter=np.zeros((displayimg['Origin']['Size'][0],displayimg['Origin']['Size'][1]))
        filter=Image.fromarray(npfilter)
        draw=ImageDraw.Draw(filter)
        draw.ellipse(delselarea,fill='red')
        # filter.save('deletefilter.tiff')
        filter=np.array(filter)
        filter=np.divide(filter,np.max(filter))
        filter=cv2.resize(filter,(reseglabels.shape[1],reseglabels.shape[0]),interpolation=cv2.INTER_LINEAR)
        indices_one=np.where(filter==1)
        reseglabels[indices_one]=0

    process()



    # gen_convband()
    # panelA.delete(coinbox)
    # reseglabels=tkintercorestat.renamelabels(reseglabels)
    # resegment([maxx,minx],[maxy,miny])
    # displayfig()

    # newcolortables=tkintercorestat.get_colortable(reseglabels)
    # newunique,newcounts=np.unique(reseglabels,return_counts=True)
    # tup=(reseglabels,newcounts,newcolortables,{},currentfilename)
    # outputdisplay,outputimg,smallset=showcounting(tup,False)
    # tempimgdict={}
    # tempimgbands={}
    # tempsmall={}
    # tempimgdict.update({'iter0':outputdisplay})
    # tempimgbands.update({'iter0':outputimg})
    # tempsmall.update({'iter0':smallset})
    # outputimgdict.update({currentfilename:tempimgdict})
    # outputimgbands.update({currentfilename:tempimgbands})
    # outputsegbands.update({currentfilename:tempsmall})
    # changeoutputimg(currentfilename,'1')
    # #update plot
    # print('done image')
    # copyplotmap=labelplotmap.copy()
    # for k,v in copyplotmap.items():
    #     if v==reflabel:
    #         figindex=figdotlist[k]
    #         figcanvas.delete(figindex)
    # if len(multiselectitems)>0:
    #     for k,v in copyplotmap.items():
    #         if v in multiselectitems and v!=reflabel:
    #             figindex=figdotlist[k]
    #             figcanvas.delete(figindex)
    #     if len(dotflash)>0:
    #         for i in range(len(dotflash)):
    #             figcanvas.delete(dotflash.pop(0))
    # #tup=list(figcanvas.find_all())
    # #figcanvas.delete(tup[-1])
    # multiselectitems=[]
    # if len(outsizethreshold)>0:
    #     for k,v in copyplotmap.items():
    #         if v in outsizethreshold and v!=reflabel:
    #             figindex=figdotlist[k]
    #             figcanvas.delete(figindex)
    # outsizethreshold=[]
    # displayfig()
    # labels=np.copy(reseglabels)
    # reseglabels,border,colortable,labeldict=tkintercorestat.resegmentinput(labels,minthres,maxthres,minlw,maxlw)
    # displayfig()

#     update plot



# def selareachoice(widget):
#     # global panelA,rects,selareapos,app
#     global rects,selareapos,app
#     app=sel_area.Application(widget)
#     rects=app.start()
#     # if selarea.get()=='1':
#     #     messagebox.showinfo('select AOI',message='Clike mouse at start point and drag on the image to define an area you want to segment.')
#     #     rects=app.start()
#     # else:
#     #     selareapos=app.getinfo(rects[1])
#     #     app.end(rects)




#def refchoice(refsubframe):
def refchoice():
    #global coinsize,sizeentry,coinbox,panelA,boundaryarea,coindict,convband
    global sizeentry,coinbox,panelA,boundaryarea,coindict,convband
    global refarea,reseglabels
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
        reseglabels[refarea]=65535
        refarea=None

def changekmeansbar(event):
    global kmeanschanged
    kmeanschanged=True

def changepcweight(event):
    global pcweightchanged,kmeanschanged
    # print('pca weight',pc_combine_up.get())
    pcweightchanged=True
    if kmeans.get()>1:
        kmeanschanged=True

def changeclusterbox(event):
    global clusterchanged,changekmeans
    clusterchanged=True
    changekmeans=True

def beforecluster(event):
    global kmeanschanged,pcweightchanged,imageframe
    if pcweightchanged==True:
        # app=''
        pcweightchanged=False
        pcweightupdate(imageframe)
    if kmeanschanged==True:
        # app=''
        kmeanschanged=False
        changecluster('')

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
mapbutton=Button(buttondisplay,text='Pilot',cursor='hand2',command=Open_Map)
mapbutton.pack(side=LEFT,padx=20,pady=5)

# disbuttonoption={'Origin':'1','PCs':'5','Color Deviation':'2','ColorIndices':'3','Output':'4'}
# buttonname={'Raw':'1','PCs':'5','Clusters':'2','Selected':'3','Output':'4'}
# #disbuttonoption={'Origin':'1','ColorIndices':'3','Output':'4'}
# for (text,v1),(name,v2) in zip(disbuttonoption.items(),buttonname.items()):
#     b=Radiobutton(buttondisplay,text=name,variable=displaybut_var,value=disbuttonoption[text],command=partial(changedisplayimg,imageframe,controlframe,text))
#     b.pack(side=LEFT,padx=20,pady=5)
#     b.configure(state=DISABLED)
#     if disbuttonoption[text]=='1':
#         b.invoke()
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
pcaframe=LabelFrame(kmeansgenframe,text='       By PCs',cursor='hand2',bd=0)

kmeansgenframe.pack()
pcaframe.pack()
# pcselframe=LabelFrame(kmeansgenframe)
# pcselframe.pack()
kmeanslabel=LabelFrame(kmeansgenframe,text='By Clusters',bd=0)
checkboxframe=LabelFrame(filter_fr,cursor='hand2',bd=0)#,text='Select classes',cursor='hand2')
kmeanslabel.pack()
pcaboxdict={}

pc1label=Label(pcaframe,text='PC1',bd=0)
pc1label.pack(side=LEFT)
pccombinebar_up=ttk.Scale(pcaframe,from_=0,to=1,length=350,orient=HORIZONTAL,variable=pc_combine_up,command=changepcweight)#,command=partial(pcweightupdate,'',imageframe))#,command=partial(print,pc_combine_up.get))
pc_combine_up.set(0.5)
pccombinebar_up.pack(side=LEFT)
pccombinebar_up.state(["disabled"])
pc2label=Label(pcaframe,text='PC2',bd=0)
pc2label.pack(side=LEFT)


# for i in range(10):
#     dictkey=str(i+1)
#     tempdict={dictkey:Variable()}
#     if i==0:
#         tempdict[dictkey].set('1')
#     else:
#         tempdict[dictkey].set('0')
#     pcaboxdict.update(tempdict)
#     ch=Checkbutton(pcselframe,text=dictkey,variable=pcaboxdict[dictkey])#,command=changepca)
#     ch.configure(state=DISABLED)
#     ch.pack(side=LEFT)

# pcaframe.config(state=DISABLED)
keys=pcaboxdict.keys()
oldpcachoice=[]
for key in keys:
    oldpcachoice.append(pcaboxdict[key].get())
kmeans.set(1)
#kmeansbar=Scale(kmeanslabel,from_=1,to=10,tickinterval=1,length=270,showvalue=0,orient=HORIZONTAL,variable=kmeans,command=partial(generatecheckbox,checkboxframe))
kmeansbar=ttk.Scale(kmeanslabel,from_=1,to=10,length=350,orient=HORIZONTAL,variable=kmeans,cursor='hand2',command=partial(generatecheckbox,checkboxframe))
kmeansbar.pack()
# kmeansbar.bind('<ButtonRelease-1>',changecluster)
kmeansbar.state(["disabled"])

# pcaframe.bind('<Leave>',lambda event,arg=imageframe:pcweightupdate(arg))
kmeansgenframe.bind('<Leave>',beforecluster)

checkboxframe.pack()
checkboxframe.bind('<Leave>',generateimgplant)
for i in range(10):
    dictkey=str(i+1)
    tempdict={dictkey:Variable()}
    # if i==0:
    #     tempdict[dictkey].set('1')
    # else:
    tempdict[dictkey].set('0')
    checkboxdict.update(tempdict)
    # ch=Checkbutton(checkboxframe,text=dictkey,variable=checkboxdict[dictkey],command=partial(generateimgplant,''))
    ch = Checkbutton(checkboxframe, text=dictkey, variable=checkboxdict[dictkey])
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
#figcanvas.grid(row=0,column=0)
resviewframe.pack()
#refframe=LabelFrame(control_fr,cursor='hand2',bd=0)
refframe=LabelFrame(bottomframe,cursor='hand2',bd=0)
refframe.pack(side=LEFT)

disbuttonoption={'Origin':'1','PCs':'5','Color Deviation':'2','ColorIndices':'3','Output':'4'}
buttonname={'Raw':'1','PCs':'5','Clusters':'2','Selected':'3','Output':'4'}
#disbuttonoption={'Origin':'1','ColorIndices':'3','Output':'4'}
for (text,v1),(name,v2) in zip(disbuttonoption.items(),buttonname.items()):
    b=Radiobutton(buttondisplay,text=name,variable=displaybut_var,value=disbuttonoption[text],command=partial(changedisplayimg,imageframe,text))
    b.pack(side=LEFT,padx=20,pady=5)
    b.configure(state=DISABLED)
    if disbuttonoption[text]=='1':
        b.invoke()

# selareabutton=Checkbutton(buttondisplay,text='SelArea',variable=selarea,command=selareachoice)
# selarea.set('0')
# selareabutton.pack(side=LEFT)
# selareabutton.configure(state=DISABLED)

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
sizeentry.pack(side=LEFT,padx=2)
sizeunit=Label(refsubframe,text='mm^2')
sizeunit.pack(side=LEFT)
delrefbutton=Button(refsubframe,text='Delete',command=del_reflabel)
delrefbutton.pack(side=LEFT,padx=40)

#delrefbutton.config(state=DISABLED)
#refbutton=Checkbutton(refsubframe,text='Ref',variable=refvar,command=partial(refchoice,refsubframe))

for widget in refsubframe.winfo_children():
    widget.config(state=DISABLED)
#extractbutton=Button(refframe,text='Process',command=partial(extraction))
extractbutton=Button(refframe,text='Segment',command=process)
extractbutton.configure(activebackground='blue',state=DISABLED)
extractbutton.pack(side=LEFT,padx=20,pady=5)
# outputbutton=Button(refframe,text='Export',command=partial(export_result,'0'))
outputbutton=Button(refframe,text='Export',command=partial(export_opts,'0'))
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

