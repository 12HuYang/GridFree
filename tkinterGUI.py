#!/usr/bin/env python3.6


#from tkinter import LabelFrame,Label,Button,Frame,Menu,Entry,LEFT,END,Listbox,Scrollbar,Variable,Canvas,Toplevel,Message,Text,DISABLED,NORMAL,Tk,Radiobutton,RIGHT,Y,CENTER,N,W
#import PIL as pil

import time
import os
from tkinter import *
from PIL import Image,ImageDraw,ImageFont
from PIL import ImageTk

from tkinter import ttk
from tkinter import messagebox
import tkinter.filedialog as filedialog
import cv2
import csv
import numpy as np
#import gdal
import matplotlib as plt
import matplotlib.pyplot as pyplt
plt.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
from matplotlib.figure import Figure
import rasterio
from rasterio.plot import show
from rasterio.enums import ColorInterp
#from rasterio.plot import show
#import os
from scipy import *
import tkintercore
import tkintercorestat
#import tkintersinglecore
#import dronprocess
from sklearn.cluster import KMeans
from functools import partial
import calculator
from skimage import filters
#import tkinterclustering

panelA=None
panelB=None
panelTree=None

RGB=None

Infrared=None
Height=None
NDVI=None
GRAY=None
RGBGRAY=None
Multiimage=None
coin=False
confirm=False
bandarrays={}
workbandarrays={}
tiflayers=[]

nodataposition=None
modified_tif=None
seg_tif=None
valid_tif=None

ratio=0

defaultfile=False
currentfilename=''

RGBlabel=None
Inflabel=None
Multilabel=None
Multigray=None
Multigrayrgb=None
Heightlabel=None
comboleft=None
comboright=None
out_png='tempNDVI.jpg'
out_temppng='pixeltempNDVI.jpg'
rotation=None
threshold=None
degreeentry=None
mapfile=''
temptif=None
labels=None
modified_boundary=None
boundarychoice=None
plantchoice=None
w=None  #canvas
erasercoord=[]
paintrecord=[]

addpixelcoord={}
removepixelcoord={}


border=None
colortable=None
greatareas=None
tinyareas=None
tipwindow = None
recborder=None

drawcontents=[]
modificationcontents=[]
pixelcontents=[]

pixelmmratio=0

caliberation={}
parameter={}

kernelsizes={}

multi_results={}

class img():
    def __init__(self,size,bands):
        self.size=size
        self.bands=bands

class CreateToolTip(object):

    def __init__(self, widget,text='widget info'):
        self.widget = widget
        self.text=text
        #self.id = None
        #self.x = self.y = 0
        self.widget.bind("<Enter>",self.enter)
        self.widget.bind("<Leave>",self.close)
        self.tw=None

    def enter(self, event):
        "Display text in tooltip window"
        x = y = 0
        #print(event.x,event.y)
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        # creates a toplevel window
        self.tw = Toplevel(self.widget)
        # Leaves only the label and removes the app window
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))
        label = Label(self.tw, text=self.text, justify='left',
                         background='yellow', relief='solid', borderwidth=1,
                         font=("times", "12", "normal"))
        label.pack(ipadx=1)

    def close(self,event):
        if self.tw:
            self.tw.destroy()
'''
def CreateToolTip(widget, text):
    #toolTip = ToolTip(widget)

    def enterarea(event):
        global tipwindow
        print(event.x,event.y)
        x, y, cx, cy = widget.bbox("insert")
        x = x + widget.winfo_rootx() + 57
        y = y + cy + widget.winfo_rooty() + 27
        tipwindow = Toplevel(widget)
        tipwindow.wm_overrideredirect(1)
        tipwindow.wm_geometry("+%d+%d" % (x, y))
        label = Label(tipwindow, text=text, justify=LEFT,
                      background="#ffffe0", relief=SOLID, borderwidth=1,
                      font=("tahoma", "12", "normal"))
        label.pack(ipadx=1)
        #toolTip.showtip(text)
    def leavearea(event):
        global tipwindow
        #toolTip.hidetip()
        #tw = tipwindow
        #tipwindow = None
        #if tw:
        #    tw.destroy()
        if tipwindow is not None:
            tipwindow.destory()
    widget.bind('<Enter>', enterarea)
    widget.bind('<Leave>', leavearea)
'''
def extrareset(canvas):
    global drawcontents
    if len(drawcontents)>0:
        for i in range(len(drawcontents)):
            canvas.delete(drawcontents[i])
    drawcontents=[]

def cleanoutliers(labels,colortable):
    unilabels=list(colortable.keys())
    nooutlier=False
    while nooutlier==False:
        nooutlier=True
        for uni in unilabels:
            pixelloc=np.where(labels==uni)
            ulx = min(pixelloc[1])
            uly = min(pixelloc[0])
            rlx = max(pixelloc[1])
            rly = max(pixelloc[0])
            midx = ulx + int((rlx - ulx) / 2)
            midy = uly + int((rly - uly) / 2)
            if labels[midy,midx]!=uni:
                ulxdist=RGB.size[1]
                for i in range(len(pixelloc[1])):
                    tempdist=pixelloc[1][i]-ulx
                    if tempdist<ulxdist:
                        ulxdist=tempdist
                rlxdist=RGB.size[1]
                for i in range(len(pixelloc[1])):
                    tempdist=rlx-pixelloc[1][i]
                    if tempdist<rlxdist:
                        rlxdist=tempdist
                ulydist=RGB.size[0]
                for i in range(len(pixelloc[0])):
                    tempdist=pixelloc[0][i]-uly
                    if tempdist<ulydist:
                        ulydist=tempdist
                rlydist=RGB.size[0]
                for i in range(len(pixelloc[0])):
                    tempdist=rly-pixelloc[0][i]
                    if tempdist<rlydist:
                        rlydist=tempdist
                if ulydist>rlydist:
                    indy=pixelloc[0].tolist().index(uly)
                else:
                    indy=pixelloc[0].tolist().index(rly)
                labels[pixelloc[0][indy],pixelloc[1][indy]]=0
                if ulxdist>rlxdist:
                    indx=pixelloc[1].tolist().index(ulx)
                else:
                    indx=pixelloc[1].tolist().index(rlx)
                labels[pixelloc[0][indx],pixelloc[1][indx]]=0
                nooutlier=False



    '''
    x = [0, -1, -1, -1, 0, 1, 1, 1]
    y = [1, 1, 0, -1, -1, -1, 0, 1]
    edges=tkintercore.get_boundary(labels)
    labels=np.where(edges==1,0,labels)
    edges=edges+tkintercore.get_boundary(labels)
    labels=np.where(edges==1,0,labels)
    edges=edges+tkintercore.get_boundary(labels)
    labels=np.where(edges==1,0,labels)
    edges=edges+tkintercore.get_boundary(labels)
    labels=np.where(edges==1,0,labels)

    for k in range(4):
        for uni in unilabels:
            boundary=tkintercore.get_boundaryloc(labels,uni)
            for i in range(len(boundary[0])):
                for j in range(len(y)):
                    if edges[boundary[0][i]+y[j],boundary[1][i]+x[j]]==1:
                        labels[boundary[0][i]+y[j],boundary[1][i]+x[j]]=uni

    #edges=tkintercore.get_boundary(labels)
    #labels=np.where(edges==1,0,labels)
    '''
    return labels

def linepixels(x0,y0,x1,y1):
    """Uses Xiaolin Wu's line algorithm to interpolate all of the pixels along a
    straight line, given two points (x0, y0) and (x1, y1)

    Wikipedia article containing pseudo code that function was based off of:
        http://en.wikipedia.org/wiki/Xiaolin_Wu's_line_algorithm
    """
    pixels = []
    steep = abs(y1 - y0) > abs(x1 - x0)

    # Ensure that the path to be interpolated is shallow and from left to right
    if steep:
        t = x0
        x0 = y0
        y0 = t

        t = x1
        x1 = y1
        y1 = t

    if x0 > x1:
        t = x0
        x0 = x1
        x1 = t

        t = y0
        y0 = y1
        y1 = t

    dx = x1 - x0
    dy = y1 - y0
    gradient = dy / dx  # slope

    # Get the first given coordinate and add it to the return list
    x_end = round(x0)
    y_end = y0 + (gradient * (x_end - x0))
    xpxl0 = x_end
    ypxl0 = round(y_end)
    if steep:
        pixels.extend([(ypxl0, xpxl0), (ypxl0 + 1, xpxl0)])
    else:
        pixels.extend([(xpxl0, ypxl0), (xpxl0, ypxl0 + 1)])

    interpolated_y = y_end + gradient

    # Get the second given coordinate to give the main loop a range
    x_end = round(x1)
    y_end = y1 + (gradient * (x_end - x1))
    xpxl1 = x_end
    ypxl1 = round(y_end)

    # Loop between the first x coordinate and the second x coordinate, interpolating the y coordinates
    for x in range(xpxl0 + 1, xpxl1):
        if steep:
            pixels.extend([(math.floor(interpolated_y), x), (math.floor(interpolated_y) + 1, x)])

        else:
            pixels.extend([(x, math.floor(interpolated_y)), (x, math.floor(interpolated_y) + 1)])

        interpolated_y += gradient

    # Add the second given coordinate to the given list
    if steep:
        pixels.extend([(ypxl1, xpxl1), (ypxl1 + 1, xpxl1)])
    else:
        pixels.extend([(xpxl1, ypxl1), (xpxl1, ypxl1 + 1)])

    return pixels

def plotlow(x0,y0,x1,y1):
    points=[]
    dx=x1-x0
    dy=y1-y0
    yi=1
    if dy<0:
        yi=-1
        dy=-dy
    D=2*dy-dx
    y=y0

    for x in range(x0,x1+1):
        points.append((x,y))
        if D>0:
            y=y+yi
            D=D-2*dx
        D=D+2*dy
    return points

def plothigh(x0,y0,x1,y1):
    points=[]
    dx=x1-x0
    dy=y1-y0
    xi=1
    if dx<0:
        xi=-1
        dx=-dx
    D=2*dx-dy
    x=x0

    for y in range(y0,y1+1):
        points.append((x,y))
        if D>0:
            x=x+xi
            D=D-2*dy
        D=D+2*dx
    return points


def bresenhamline(x0,y0,x1,y1):
    if abs(y1-y0)<abs(x1-x0):
        if x0>x1:
            return plotlow(x1,y1,x0,y0)
        else:
            return plotlow(x0,y0,x1,y1)
    else:
        if y0>y1:
            return plothigh(x1,y1,x0,y0)
        else:
            return plothigh(x0,y0,x1,y1)

def tengentsub(x0,y0,x1,y1,ulx,uly,labels,uni):
    if abs(y1-y0)<abs(x1-x0):
        if x0>x1:
            if y0<y1:
                y1=y1-1
                y0=y0-1
                while(y1>=uly):
                    points=plotlow(x1,y1,x0,y0)
                    tengentnum=0
                    for point in points:
                        if labels[int(point[1]),int(point[0])]==uni:
                            tengentnum+=1
                    if tengentnum==0:
                        y0=y0+1
                        y1=y1+1
                        points=plotlow(x1,y1,x0,y0)
                        break
                    if tengentnum==1:
                        break
                    else:
                        y0=y0-1
                        y1=y1-1
                return points
            else:
                y1=y1-1
                y0=y0-1
                while(y0>=uly):
                    points=plotlow(x1,y1,x0,y0)
                    tengentnum=0
                    for point in points:
                        if labels[int(point[1]),int(point[0])]==uni:
                            tengentnum+=1
                    if tengentnum==0:
                        y0=y0+1
                        y1=y1+1
                        points=plotlow(x1,y1,x0,y0)
                        break
                    if tengentnum==1:
                        break
                    else:
                        y0=y0-1
                        y1=y1-1
                return points
        else:
            if y0<y1:
                y1=y1-1
                y0=y0-1
                while(y1>=uly):
                    points=plotlow(x0,y0,x1,y1)
                    tengentnum=0
                    for point in points:
                        if labels[int(point[1]),int(point[0])]==uni:
                            tengentnum+=1
                    if tengentnum==0:
                        y0=y0+1
                        y1=y1+1
                        points=plotlow(x0,y0,x1,y1)
                        break
                    if tengentnum==1:
                        break
                    else:
                        y0=y0-1
                        y1=y1-1
                return points
            else:
                y1=y1-1
                y0=y0-1
                while(y0>=uly):
                    points=plotlow(x0,y0,x1,y1)
                    tengentnum=0
                    for point in points:
                        if labels[int(point[1]),int(point[0])]==uni:
                            tengentnum+=1
                    if tengentnum==0:
                        y0=y0+1
                        y1=y1+1
                        points=plotlow(x0,y0,x1,y1)
                        break
                    if tengentnum==1:
                        break
                    else:
                        y0=y0-1
                        y1=y1-1
                try:
                    return points
                except:
                    return []
    else:
        if y0>y1:
            if x0<x1:
                x0=x0-1
                x1=x1-1
                while(x1>=ulx):
                    points=plothigh(x1,y1,x0,y0)
                    tengentnum=0
                    for point in points:
                        if labels[int(point[1]),int(point[0])]==uni:
                            tengentnum+=1
                    if tengentnum==0:
                        x0=x0+1
                        x1=x1+1
                        points=plothigh(x1,y1,x0,y0)
                        break
                    if tengentnum==1:
                        break
                    else:
                        x0=x0-1
                        x1=x1-1
                return points
            else:
                x0=x0+1
                x1=x1+1
                while(x0>=ulx):
                    points=plothigh(x1,y1,x0,y0)
                    tengentnum=0
                    for point in points:
                        if labels[int(point[1]),int(point[0])]==uni:
                            tengentnum+=1
                    if tengentnum==0:
                        x0=x0+1
                        x1=x1+1
                        points=plothigh(x1,y1,x0,y0)
                        break
                    if tengentnum==1:
                        break
                    else:
                        x0=x0-1
                        x1=x1-1
                return points
        else:
            if x0<x1:
                x0=x0-1
                x1=x1-1
                while(x1>=ulx):
                    points=plothigh(x0,y0,x1,y1)
                    tengentnum=0
                    for point in points:
                        if labels[int(point[1]),int(point[0])]==uni:
                            tengentnum+=1
                    if tengentnum==0:
                        x0=x0+1
                        x1=x1+1
                        points=plothigh(x0,y0,x1,y1)
                        break
                    if tengentnum==1:
                        break
                    else:
                        x0=x0-1
                        x1=x1-1
                return points
            else:
                x0=x0+1
                x1=x1+1
                while(x0>=ulx):
                    points=plothigh(x0,y0,x1,y1)
                    tengentnum=0
                    for point in points:
                        if int(point[1])<labels.shape[0] and int(point[0])<labels.shape[1]:
                            if labels[int(point[1]),int(point[0])]==uni:
                                tengentnum+=1
                    if tengentnum==0:
                        x0=x0+1
                        x1=x1+1
                        points=plothigh(x0,y0,x1,y1)
                        break
                    if tengentnum==1:
                        break
                    else:
                        x0=x0-1
                        x1=x1-1
                return points


def tengentadd(x0,y0,x1,y1,rlx,rly,labels,uni):
    if abs(y1-y0)<abs(x1-x0):
        if x0>x1:   #low
            if y0<y1:
                y0=y0+1
                y1=y1+1
                while(y0<=rly):
                    points=plotlow(x1,y1,x0,y0)
                    tengentnum=0
                    for point in points:
                        if int(point[1])<labels.shape[0] and int(point[0])<labels.shape[1]:
                            if labels[int(point[1]),int(point[0])]==uni:
                                tengentnum+=1
                    if tengentnum==0:
                        y0=y0-1
                        y1=y1-1
                        points=plotlow(x1,y1,x0,y0)
                        break
                    if tengentnum==1:
                        break
                    else:
                        y0=y0+1
                        y1=y1+1
                return points
            else:
                y0=y0+1
                y1=y1+1
                while(y1<=rly):
                    points=plotlow(x1,y1,x0,y0)
                    tengentnum=0
                    for point in points:
                        if labels[int(point[1]),int(point[0])]==uni:
                            tengentnum+=1
                    if tengentnum==0:
                        y0=y0-1
                        y1=y1-1
                        points=plotlow(x1,y1,x0,y0)
                        break
                    if tengentnum==1:
                        break
                    else:
                        y0=y0+1
                        y1=y1+1
                return points
        else:
            if y0<y1:
                y0=y0+1
                y1=y1+1
                while(y0<=rly):
                    points=plotlow(x0,y0,x1,y1)
                    tengentnum=0
                    for point in points:
                        if int(point[1])<labels.shape[0] and int(point[0])<labels.shape[1]:
                            if labels[int(point[1]),int(point[0])]==uni:
                                tengentnum+=1
                    if tengentnum==0:
                        y0=y0-1
                        y1=y1-1
                        points=plotlow(x0,y0,x1,y1)
                        break
                    if tengentnum==1:
                        break
                    else:
                        y0=y0+1
                        y1=y1+1
                return points
            else:
                y0=y0+1
                y1=y1+1
                while(y1<=rly):
                    points=plotlow(x0,y0,x1,y1)
                    tengentnum=0
                    for point in points:
                        if labels[int(point[1]),int(point[0])]==uni:
                            tengentnum+=1
                    if tengentnum==0:
                        y0=y0-1
                        y1=y1-1
                        points=plotlow(x0,y0,x1,y1)
                        break
                    if tengentnum==1:
                        break
                    else:
                        y0=y0+1
                        y1=y1+1
                try:
                    return points
                except:
                    return []
    else:
        if y0>y1:
            if x0<x1:
                x0=x0+1
                x1=x1+1
                while(x0<rlx):
                    points=plothigh(x1,y1,x0,y0)
                    tengentnum=0
                    for point in points:
                        if labels[int(point[1]),int(point[0])]==uni:
                            tengentnum+=1
                    if tengentnum==0:
                        x0=x0-1
                        x1=x1-1
                        points=plothigh(x1,y1,x0,y0)
                        break
                    if tengentnum==1:
                        break
                    else:
                        x0=x0+1
                        x1=x1+1
                try:
                    return points
                except:
                    return []
            else:
                x0=x0+1
                x1=x1+1
                while(x1<rlx):
                    points=plothigh(x1,y1,x0,y0)
                    tengentnum=0
                    for point in points:
                        if labels[int(point[1]),int(point[0])]==uni:
                            tengentnum+=1
                    if tengentnum==0:
                        x0=x0-1
                        x1=x1-1
                        points=plothigh(x1,y1,x0,y0)
                        break
                    if tengentnum==1:
                        break
                    else:
                        x0=x0+1
                        x1=x1+1
                return points
        else:
            if x0<x1:
                x0=x0+1
                x1=x1+1
                while(x0<=rlx):
                    points=plothigh(x0,y0,x1,y1)
                    tengentnum=0
                    for point in points:
                        if int(point[1])<labels.shape[0] and int(point[0])<labels.shape[1]:
                            if labels[int(point[1]),int(point[0])]==uni:
                                tengentnum+=1
                    if tengentnum==0:
                        x0=x0-1
                        x1=x1-1
                        points=plothigh(x0,y0,x1,y1)
                        break
                    if tengentnum==1:
                        break
                    else:
                        x0=x0+1
                        x1=x1+1
                return points
            else:
                x0=x0+1
                x1=x1+1
                while(x1<=rlx):
                    points=plothigh(x0,y0,x1,y1)
                    tengentnum=0
                    for point in points:
                        if int(point[1])<labels.shape[0] and int(point[0])<labels.shape[1]:
                            if labels[int(point[1]),int(point[0])]==uni:
                                tengentnum+=1
                    if tengentnum==0:
                        x0=x0-1
                        x1=x1-1
                        points=plothigh(x0,y0,x1,y1)
                        break
                    if tengentnum==1:
                        break
                    else:
                        x0=x0+1
                        x1=x1+1
                try:
                    return points
                except:
                    return []

def showcountingstat(tupinput,usertype):
    global drawcontents,kernelsizes,pixelmmratio
    kerneldict={}
    filename=tupinput[0]
    coinparts=tupinput[1]
    labeldict=tupinput[2]
    labelkeys=list(labeldict.keys())
    if filename in multi_results:
        return
    multi_results.update({filename:labeldict})
    originfile,extension=os.path.splitext(filename)
    canvascontainter={}
    for key in labelkeys:
        tempdict={}
        iterreturn=labeldict[key]
        labels=iterreturn['labels']
        #labels=labels.astype(int)
        colortable=iterreturn['colortable']
        counts=iterreturn['counts']
        if usertype==1:
            img=Image.open(filename)
        else:
            if extension=='.tif':
                img=Image.open('tempcroptif.png')
            else:
                img=Image.open(filename)
        img=img.resize([labels.shape[1],labels.shape[0]],resample=Image.BILINEAR)
        draw=ImageDraw.Draw(img)
        font=ImageFont.load_default()
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
        uniquelabels=list(colortable.keys())
        for uni in uniquelabels:
            if uni !=0:
                pixelloc = np.where(labels == uni)
                if len(pixelloc[0])>0:
                    ulx = min(pixelloc[1])
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

                    lengthpoints=bresenhamline(x0,y0,x1,y1)  #x0,y0,x1,y1
                    for point in lengthpoints:
                        draw.point([int(point[0]),int(point[1])],fill='yellow')
                    tengentaddpoints=tengentadd(x0,y0,x1,y1,rlx,rly,labels,uni)
                    #for point in tengentaddpoints:
                        #if int(point[0])>=ulx and int(point[0])<=rlx and int(point[1])>=uly and int(point[1])<=rly:
                    #    draw.point([int(point[0]),int(point[1])],fill='green')
                    tengentsubpoints=tengentsub(x0,y0,x1,y1,ulx,uly,labels,uni)
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
                        draw.text([(midx,midy)],text=canvastext,fill='black',font=font)
                    tempdict.update({uni:[kernellength,width,kernellength*pixelmmratio,width*pixelmmratio]})
        kerneldict.update({key:tempdict})
        print(key,tempdict)
        content='item count:'+str(len(uniquelabels))+' ID:'+key+' '+filename
        draw.text((10,10),text=content,fill='black',font=font)
        #c=Toplevel()
        #canvasframe=Label(c)
        #canvas=Canvas(canvasframe,width=labels.shape[1],height=labels.shape[0])
        #img.save('templabelimg.png')
        #statimage=cv2.imread('templabelimg.png')
        #statimage=Image.fromarray(statimage)
        #statimage=ImageTk.PhotoImage(statimage)
        #canvas.create_image(0,0,image=statimage,anchor=NW)
        #canvasframe.image=statimage
        #canvascontainter.update({key:c})
        #canvasframe.pack()
        #canvas.pack()
        img.show()
        #pyplt.imshow(img)
    kernelsizes.update({filename:kerneldict})


def showcounting(tupinput,usertype):
    global drawcontents,kernelsizes,pixelmmratio
    tempdict={}
    labels=tupinput[0]
    border=tupinput[1]
    colortable=tupinput[2]
    filename=tupinput[3]
    coinparts=tupinput[4]
    multi_results.update({filename:(labels,border,colortable)})
    uniquelabels=list(colortable.keys())
    originfile,extension=os.path.splitext(filename)
    if usertype==1:
        img=Image.open(filename)
    else:
        if extension=='.tif':
            img=Image.open('tempcroptif.png')
        else:
            img=Image.open(filename)
    img=img.resize([labels.shape[1],labels.shape[0]],resample=Image.BILINEAR)
    if usertype==1:
        img.save(originfile+'-training'+extension,"JPEG")
    draw=ImageDraw.Draw(img)
    font=ImageFont.load_default()
    trainingdataset=[]
    trainingdataset.append(['image_name','type','xmin','xmax','ymin','ymax'])
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
            ulx = min(pixelloc[1])
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

            lengthpoints=bresenhamline(x0,y0,x1,y1)  #x0,y0,x1,y1
            for point in lengthpoints:
                draw.point([int(point[0]),int(point[1])],fill='yellow')
            tengentaddpoints=tengentadd(x0,y0,x1,y1,rlx,rly,labels,uni)
            #for point in tengentaddpoints:
                #if int(point[0])>=ulx and int(point[0])<=rlx and int(point[1])>=uly and int(point[1])<=rly:
            #    draw.point([int(point[0]),int(point[1])],fill='green')
            tengentsubpoints=tengentsub(x0,y0,x1,y1,ulx,uly,labels,uni)
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
                draw.text([(midx,midy)],text=canvastext,fill='black',font=font)
                trainingdataset.append([originfile+'-training'+extension,'wheat',str(ulx),str(rlx),str(uly),str(rly)])
    kernelsizes.update({filename:tempdict})
    content='item count:'+str(len(uniquelabels))+'\n File: '+filename
    contentlength=len(content)+50
    #rectext=canvas.create_text(10,10,fill='black',font='Times 16',text=content,anchor=NW)
    if usertype==1:
        draw.text((10,10),text=content,fill='black',font=font)
        img.save(originfile+'-countresult'+extension,"JPEG")
    if usertype==2:
        img=img.resize([RGB.size[1],RGB.size[0]],resample=Image.BILINEAR)
        img.save(originfile+'-countresult'+'.png',"PNG")
    img.show()


    return trainingdataset

def extractimg(event,tupinput):
    global drawcontents
    labels=tupinput[0]
    #ratio=int(max(RGB.size[0],RGB.size[1])/450)
    #labels=labels.astype('float32')
    #labels=cv2.resize(src=labels,dsize=(RGB.size[1],RGB.size[0]),interpolation=cv2.INTER_LINEAR)
    canvas=tupinput[1]
    colortable=tupinput[2]
    #labels=cleanoutliers(labels,colortable)
    viewselection=tupinput[3]
    textselection=tupinput[4]
    uniquelabels=list(colortable.keys())
    global recborder
    print(event.x,event.y)
    if viewselection.get()==2: #show all bounding box with labels
        #uniqlabel=np.unique(labels)
        for uni in uniquelabels:
            if uni !=0:
                pixelloc = np.where(labels == uni)
                ulx = min(pixelloc[1])
                uly = min(pixelloc[0])
                rlx = max(pixelloc[1])
                rly = max(pixelloc[0])
                print(event.x, event.y, labels[event.x, event.y], ulx, uly, rlx, rly)
                midx = ulx + int((rlx - ulx) / 2)
                midy = uly + int((rly - uly) / 2)
                recborder = canvas.create_rectangle(ulx, uly, rlx, rly, outline='red')
                drawcontents.append(recborder)
                if uni in colortable:
                    canvastext = colortable[uni]
                else:
                    canvastext = 'No label'
                rectext = canvas.create_text(midx, midy, fill='black', font='Times 8', text=canvastext)
                drawcontents.append(rectext)
        rectext=canvas.create_text(60,10,fill='black',font='Times 16',text='item count:'+str(len(uniquelabels)))
        drawcontents.append(rectext)
        return
    if viewselection.get()==3: #show all filled pixels with labels
        #uniqlabel = np.unique(labels)
        for uni in uniquelabels:
            if uni != 0:
                pixelloc = np.where(labels == uni)
                ulx = min(pixelloc[1])
                uly = min(pixelloc[0])
                rlx = max(pixelloc[1])
                rly = max(pixelloc[0])
                print(event.x, event.y, labels[event.x, event.y], ulx, uly, rlx, rly)
                midx = ulx + int((rlx - ulx) / 2)
                midy = uly + int((rly - uly) / 2)
                for i in range(len(pixelloc[0])):
                    recborder = canvas.create_oval(pixelloc[1][i], pixelloc[0][i], pixelloc[1][i], pixelloc[0][i],
                                                   width=0, fill='red')
                    drawcontents.append(recborder)
                if uni in colortable:
                    canvastext = colortable[uni]
                else:
                    canvastext = 'No label'
                rectext = canvas.create_text(midx, midy, fill='black', font='Times 8', text=canvastext)
                drawcontents.append(rectext)
        rectext = canvas.create_text(60, 10, fill='black', font='Times 16',
                                     text='item count:' + str(len(uniquelabels)))
        drawcontents.append(rectext)
        return
    if event.y>=0 and event.y<450 and event.x>=0 and event.x<450:
        if labels[event.y,event.x]!=0:
            pixelloc=np.where(labels==labels[event.y,event.x])
            ulx=min(pixelloc[1])
            uly=min(pixelloc[0])
            rlx=max(pixelloc[1])
            rly=max(pixelloc[0])
            print(event.x, event.y,labels[event.x,event.y],ulx,uly,rlx,rly)
            midx = ulx + int((rlx - ulx) / 2)
            midy = uly + int((rly - uly) / 2)
            viewtype=viewselection.get()
            texttype=textselection.get()
            if viewtype==1:
                recborder=canvas.create_rectangle(ulx,uly,rlx,rly,outline='red')
                drawcontents.append(recborder)
            if viewtype==0:
                for i in range(len(pixelloc[0])):
                    recborder=canvas.create_oval(pixelloc[1][i],pixelloc[0][i],pixelloc[1][i],pixelloc[0][i],width=0,fill='red')
                    drawcontents.append(recborder)
            if texttype==1:
                if labels[event.y,event.x] in colortable:
                    canvastext=colortable[labels[event.y,event.x]]
                else:
                    canvastext='No label'
                rectext=canvas.create_text(midx,midy,fill='black',font='Times 8',text=canvastext)
                drawcontents.append(rectext)
        else:
            recborder=None

def removeandhighlight(args,selection):
    global modificationcontents
    canvas=args[0]
    locallables=args[1]
    colortable=args[2]
    spaceindex=selection.find(' ')
    selectedlabel=selection[spaceindex+1:]
    if len(modificationcontents)>0:
        for i in range(len(modificationcontents)):
            canvas.delete(modificationcontents[i])
    labelkeys=list(colortable.keys())
    valuekeys=list(colortable.values())
    if selectedlabel!='':
        labelindex=valuekeys.index(selectedlabel)
        currentlabel=labelkeys[labelindex]
        pixelloc=np.where(locallables==currentlabel)
        for i in range(len(pixelloc[0])):
            dot=canvas.create_oval(pixelloc[1][i],pixelloc[0][i],pixelloc[1][i],pixelloc[0][i],width=0,fill='red')
            modificationcontents.append(dot)


    pass

def erasepixel(e,tupinput):
    global removepixelcoord,pixelcontents
    selection=tupinput[0]
    selection=selection.get()
    spaceindex = selection.find(' ')
    selection=selection[spaceindex+1:]
    canvas=tupinput[1]
    if selection=='':
        messagebox.showerror('No selected label',message='Plaese choose label you want to remove pixel form.')
        return

    mousex = e.x
    mousey = e.y
   
    content=canvas.create_rectangle(mousex, mousey, mousex + 3, mousey + 3, fill='black',
                       outline="")
    pixelcontents.append(content)
    # w.old_drawcoords=mousex,mousey
    if selection not in removepixelcoord:
        temptup=([],[])
        for i in range(4):
            temptup[0].append(e.y)
            temptup[0].append(e.y+i)
            temptup[0].append(e.y)
            temptup[0].append(e.y+i)
            temptup[1].append(e.x)
            temptup[1].append(e.x)
            temptup[1].append(e.x+i)
            temptup[1].append(e.x+i)
            print('draw at', e.x + i, e.y + i)
        tempdict={selection:temptup}
        removepixelcoord.update(tempdict)
    else:
        for i in range(4):
            removepixelcoord[selection][0].append(e.y)
            removepixelcoord[selection][0].append(e.y + i)
            removepixelcoord[selection][0].append(e.y)
            removepixelcoord[selection][0].append(e.y + i)
            removepixelcoord[selection][1].append(e.x)
            removepixelcoord[selection][1].append(e.x)
            removepixelcoord[selection][1].append(e.x + i)
            removepixelcoord[selection][1].append(e.x + i)
            print('draw at', e.x + i, e.y + i)
    pass

def drawpixel(e,tupinput):
    global addpixelcoord,pixelcontents
    selection = tupinput[0]
    selection=selection.get()
    spaceindex = selection.find(' ')
    selection = selection[spaceindex + 1:]
    canvas = tupinput[1]
    if selection == '':
        messagebox.showerror('No selected label', message='Plaese choose label you want to add pixel form.')
        return

    mousex = e.x
    mousey = e.y

    content=canvas.create_rectangle(mousex, mousey, mousex + 3, mousey + 3, fill='yellow',
                            outline="")
    pixelcontents.append(content)

    if selection not in addpixelcoord:
        temptup=([],[])
        for i in range(4):
            temptup[0].append(e.y)
            temptup[0].append(e.y+i)
            temptup[0].append(e.y)
            temptup[0].append(e.y+i)
            temptup[1].append(e.x)
            temptup[1].append(e.x)
            temptup[1].append(e.x+i)
            temptup[1].append(e.x+i)
            print('draw at', e.x + i, e.y + i)
        tempdict={selection:temptup}
        addpixelcoord.update(tempdict)
    else:
        for i in range(4):
            addpixelcoord[selection][0].append(e.y)
            addpixelcoord[selection][0].append(e.y + i)
            addpixelcoord[selection][0].append(e.y)
            addpixelcoord[selection][0].append(e.y + i)
            addpixelcoord[selection][1].append(e.x)
            addpixelcoord[selection][1].append(e.x)
            addpixelcoord[selection][1].append(e.x + i)
            addpixelcoord[selection][1].append(e.x + i)
            print('draw at', e.x + i, e.y + i)
    pass

def resetpixel(canvas,localimage):
    global addpixelcoord,removepixelcoord,pixelcontents
    addpixelcoord.clear()
    removepixelcoord.clear()
    for i in range(len(pixelcontents)):
        canvas.delete(pixelcontents[i])
    canvas.create_image(0,0,image=localimage,anchor=NW)

    pass

def removepixel(canvas,selection):
    canvas.config(cursor='cross')
    print(selection.get())
    tup=(selection,canvas)
    canvas.bind("<B1-Motion>",lambda event,arg=tup:erasepixel(event,tup))
    pass

def addpixel(canvas,selection):
    canvas.config(cursor='pencil')
    print(selection.get())
    tup = (selection, canvas)
    canvas.bind("<B1-Motion>", lambda event, arg=tup:drawpixel(event, tup))
    pass

def pixelpreview(tupinput):
    locallabel=tupinput[0]
    colortable=tupinput[1]
    addpixelkeys=list(addpixelcoord.keys())
    erasepixelkeys=list(removepixelcoord.keys())
    colorkeys=list(colortable.keys())
    colorvalues=list(colortable.values())
    for i in range(len(addpixelkeys)):
        currkey=addpixelkeys[i]
        keyloc=addpixelcoord[currkey]
        for j in range(len(keyloc[0])):
            if modified_tif[keyloc[0][j],keyloc[1][j]]==1:
                colorindex=colorvalues.index(currkey)
                locallabel[keyloc[0][j],keyloc[1][j]]=colorkeys[colorindex]
    for i in range(len(erasepixelkeys)):
        currkey=erasepixelkeys[i]
        keyloc=removepixelcoord[currkey]
        for j in range(len(keyloc[0])):
            locallabel[keyloc[0][j],keyloc[1][j]]=0
    tempborder=tkintercore.makeboundary(locallabel)

    tempbands = np.zeros((450, 450, 3), np.uint8)
    tempbands[:, :, 0] = tempborder
    tempbands[:, :, 1] = modified_tif * 150
    tempbands[:, :, 2] = tempborder
    pyplt.imsave(out_temppng, tempbands)
    image = cv2.imread(out_temppng)
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)

    panelA = Label(ctr_left)
    panelA.image = image
    panelA.grid(row=1, column=0)

    panelE=Toplevel()
    frame=LabelFrame(panelE)
    viewcanvas=Canvas(frame,width=450,height=450,bg='white')
    viewcanvas.create_image(0,0,image=image,anchor=NW)

    frame.pack()
    viewcanvas.pack()

def export_modifiedresults(tupinput):
    locallabel=tupinput[0]
    colortable=tupinput[1]
    addpixelkeys = list(addpixelcoord.keys())
    erasepixelkeys = list(removepixelcoord.keys())
    colorkeys = list(colortable.keys())
    colorvalues = list(colortable.values())
    for i in range(len(addpixelkeys)):
        currkey = addpixelkeys[i]
        keyloc = addpixelcoord[currkey]
        for j in range(len(keyloc[0])):
            if modified_tif[keyloc[0][j], keyloc[1][j]] == 1:
                colorindex = colorvalues.index(currkey)
                locallabel[keyloc[0][j], keyloc[1][j]] = colorkeys[colorindex]
    for i in range(len(erasepixelkeys)):
        currkey = erasepixelkeys[i]
        keyloc = removepixelcoord[currkey]
        for j in range(len(keyloc[0])):
            if keyloc[0][j]>=0 and keyloc[0][j]<450 and keyloc[1][j]<450 and keyloc[0][j]>=0:
                locallabel[keyloc[0][j], keyloc[1][j]] = 0
    tempborder = tkintercore.makeboundary(locallabel)

    path = filedialog.askdirectory()
    if type(rotation) != type(None):
        if rotation != 0:
            center = (225, 225)
            a = rotation * -1.0
            print(a)
            M = cv2.getRotationMatrix2D(center, a, 1.0)
            tempborder = cv2.warpAffine(tempborder.astype('float32'), M, dsize=(450, 450), flags=cv2.INTER_LINEAR)
            locallabel = cv2.warpAffine(locallabel.astype('float32'), M, dsize=(450, 450), flags=cv2.INTER_LINEAR)
    if len(path) > 0:
        messagebox.showinfo('Save process', message='Program is saving results to' + path)
        tempborder = tempborder.astype('float32')
        print(tempborder)
        realborder = cv2.resize(src=tempborder, dsize=(RGB.size[1], RGB.size[0]), interpolation=cv2.INTER_LINEAR)
        #out_img = path + '/Output_Modified_RGBwithBorder.tif'
        band1 = RGB.bands[0] + realborder * 255
        band2 = RGB.bands[1]
        band3 = RGB.bands[2]

        '''
        gtiffdriver = gdal.GetDriverByName('GTiff')
        out_ds = gtiffdriver.Create(out_img, RGB.size[1], RGB.size[0], 3, 3)
        # out_ds.SetGeoTransform(in_gt)
        # out_ds.SetProjection(dataproj)
        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(band1)
        out_band = out_ds.GetRasterBand(2)
        out_band.WriteArray(band2)
        out_band = out_ds.GetRasterBand(3)
        out_band.WriteArray(band3)
        out_ds.FlushCache()
        out_img = path + '/Modified_Labeleddata.tif'
        '''
        outputimg = np.zeros((RGB.size[0], RGB.size[1], 3))
        outputimg[:, :, 0] = band1
        outputimg[:, :, 1] = band2
        outputimg[:, :, 2] = band3
        outputimg = outputimg.astype('uint8')
        outputimg = np.where(outputimg == 0, 255, outputimg)
        pyplt.imsave(path + '/Modified-OutputRGB-Border.png', outputimg)
        floatlabels = locallabel.astype('float32')
        floatlabels = cv2.resize(src=floatlabels, dsize=(RGB.size[1], RGB.size[0]), interpolation=cv2.INTER_LINEAR)
        lastone = np.zeros(floatlabels.shape, dtype='float32')
        unikeys = list(colortable.keys())
        # for uni in colortable:
        for i in range(len(unikeys)):
            lastone = np.where(floatlabels == float(unikeys[i]), i, lastone)

        band1 = lastone
        '''
        out_ds = gtiffdriver.Create(out_img, RGB.size[1], RGB.size[0], 1, 3)
        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(band1)
        out_ds.FlushCache()
        '''
        band1 = band1.astype('uint8')
        band1 = np.where(band1 == 0, 255, band1)
        pyplt.imsave(path + '/Modified-Labeleddata.png', band1)

        indicekeys = list(bandarrays.keys())
        indeclist = [0 for i in range(len(indicekeys) * 3)]
        #originrestoredband = np.multiply(locallabel, modified_tif)
        originrestoredband=locallabel
        restoredband = originrestoredband.astype('float32')
        restoredband = cv2.resize(src=restoredband, dsize=(RGB.size[1], RGB.size[0]), interpolation=cv2.INTER_LINEAR)

        datatable = {}
        origindata = {}
        for key in indicekeys:
            data = bandarrays[key]
            # data=workbandarrays[key]
            data = data.tolist()
            tempdict = {key: data}
            origindata.update(tempdict)
        for uni in colortable:
            print(uni, colortable[uni])
            uniloc = np.where(restoredband == float(uni))
            if len(uniloc) == 0 or len(uniloc[1]) == 0:
                continue
            # width=max(uniloc[0])-min(uniloc[0])
            # length=max(uniloc[1])-min(uniloc[1])

            # subarea=restoredband[min(uniloc[0]):max(uniloc[0])+1,min(uniloc[1]):max(uniloc[1])+1]
            # findcircle(subarea)
            smalluniloc = np.where(originrestoredband == uni)
            ulx, uly = min(smalluniloc[1]), min(smalluniloc[0])
            rlx, rly = max(smalluniloc[1]), max(smalluniloc[0])
            width = rlx - ulx
            length = rly - uly
            print(width, length)
            subarea = restoredband[uly:rly + 1, ulx:rlx + 1]
            subarea = subarea.tolist()
            amount = len(uniloc[0])
            print(amount)
            templist = [amount, length, width]
            tempdict = {colortable[uni]: templist + indeclist}  # NIR,Redeyes,R,G,B,NDVI,area
            for ki in range(len(indicekeys)):
                # originNDVI=bandarrays[indicekeys[ki]]
                # originNDVI=originNDVI.tolist()
                originNDVI = origindata[indicekeys[ki]]
                pixellist = []
                for k in range(len(uniloc[0])):
                    # tempdict[colortable[uni]][5]+=databand[uniloc[0][k]][uniloc[1][k]]
                    # tempdict[colortable[uni]][0]+=infrbands[0][uniloc[0][k]][uniloc[1][k]]
                    # tempdict[colortable[uni]][1]+=infrbands[2][uniloc[0][k]][uniloc[1][k]]
                    # tempdict[colortable[uni]][2]+=rgbbands[1][uniloc[0][k]][uniloc[1][k]]
                    # tempdict[colortable[uni]][3]+=rgbbands[0][uniloc[0][k]][uniloc[1][k]]
                    # tempdict[colortable[uni]][4]+=rgbbands[2][uniloc[0][k]][uniloc[1][k]]
                    # tempdict[colortable[uni]][6]+=NDVI[uniloc[0][k]][uniloc[1][k]]
                    tempdict[colortable[uni]][3 + ki * 3] += originNDVI[uniloc[0][k]][uniloc[1][k]]
                    tempdict[colortable[uni]][4 + ki * 3] += originNDVI[uniloc[0][k]][uniloc[1][k]]
                    pixellist.append(originNDVI[uniloc[0][k]][uniloc[1][k]])
                # for i in range(7):
                tempdict[colortable[uni]][ki * 3 + 3] = tempdict[colortable[uni]][ki * 3 + 3] / amount
                tempdict[colortable[uni]][ki * 3 + 5] = np.std(pixellist)
            datatable.update(tempdict)

        filename = path + '/Modified_NDVIdata.csv'
        with open(filename, mode='w') as f:
            csvwriter = csv.writer(f)
            rowcontent = ['Index', 'Plot', 'Area(#pixel)', 'Length(#pixel)', 'Width(#pixel)']
            for key in indicekeys:
                rowcontent.append('avg-' + str(key))
                rowcontent.append('sum-' + str(key))
                rowcontent.append('std-' + str(key))
            # csvwriter.writerow(['ID','NIR','Red Edge','Red','Green','Blue','NIRv.s.Green','LabOstu','area(#of pixel)'])
            # csvwriter.writerow(['Index','Plot','Area(#pixels)','avg-NDVI','sum-NDVI','std-NDVI','Length(#pixel)','Width(#pixel)'])#,'#holes'])
            csvwriter.writerow(rowcontent)
            i = 0
            for uni in datatable:
                row = [i, uni]
                for j in range(len(datatable[uni])):
                    row.append(datatable[uni][j])
                # row=[i,uni,datatable[uni][0],datatable[uni][1],datatable[uni][2],datatable[uni][5],datatable[uni][3],datatable[uni][4]]#,
                # datatable[uni][5]]
                i += 1
                print(row)
                csvwriter.writerow(row)
        messagebox.showinfo('Saved', message='Results are saved to ' + path)

def modifyextra(inputtup):
    locallabels=inputtup[0]
    localimage=inputtup[1]
    colortable=inputtup[2]
    triggerbutton=inputtup[3].winfo_children()[1]
    #triggerbutton.config(state='disabled')

    panelD=Toplevel()
    extraframe=LabelFrame(panelD)
    modifyframe = LabelFrame(extraframe, text='Modification')
    labeldes = Label(modifyframe, text='Select a label')
    labelvariable=StringVar(modifyframe)
    OPTIONS=list(colortable.values())
    for i in range(len(OPTIONS)):
        OPTIONS[i]='['+str(i+1)+'] '+str(OPTIONS[i])

    canvasframe=LabelFrame(extraframe)
    canvasw=Canvas(canvasframe,width=450,height=450,bg='white')
    canvasw.create_image(0,0,image=localimage,anchor=NW)

    tup=(locallabels,colortable)

    buttonframe=LabelFrame(extraframe)
    resetbutton=Button(buttonframe,text='reset',command=partial(resetpixel,canvasw,localimage))
    previewbutton=Button(buttonframe,text='Preview',command=partial(pixelpreview,tup))
    confirmbutton=Button(buttonframe,text='confirm & export',command=partial(export_modifiedresults,tup))

    tup=(canvasw,locallabels,colortable)

    labelname = OptionMenu(modifyframe, labelvariable, *OPTIONS,
                           command=partial(removeandhighlight,tup))

    removepixelradiobutton = Radiobutton(modifyframe, text='Remove pixel', value=0,command=partial(removepixel,canvasw,labelvariable))
    addpixelradiobutton = Radiobutton(modifyframe, text='Add pixel', value=1,command=partial(addpixel,canvasw,labelvariable))

    extraframe.pack()
    modifyframe.pack()
    canvasframe.pack()
    buttonframe.pack()

    labeldes.grid(row=0, column=0)
    labelname.grid(row=0, column=1)
    removepixelradiobutton.grid(row=1, column=0)
    addpixelradiobutton.grid(row=1, column=1)

    canvasw.pack()

    resetbutton.grid(row=0,column=0,padx=3,pady=3)
    previewbutton.grid(row=0,column=1,padx=3,pady=3)
    confirmbutton.grid(row=0,column=2,padx=3,pady=3)
    pass

def modifyresult(inputtup):
    labels,image,colortable,buttonframe=inputtup
    items,count=np.unique(labels,return_counts=True)
    hist=dict(zip(items[1:],count[1:]))
    mean=sum(count[1:])/len(count[1:])
    sigma=np.std(np.asarray(count[1:]))
    uprange=mean+3*sigma
    for item in items[1:]:
        if hist[item]>uprange:
            temppanel=Toplevel()

    pass




def startbatchextract(usertype):
    global panelA,currentfilename
    if type(Multiimage)==type(None):
        return
    multifiles=Multiimage.keys()
    for key in multifiles:
        rgbimg=Multiimage[key]
        grayimg=Multigray[key]
        grayrgb=Multigrayrgb[key]
        keybandarrays,keyworkbandarrays=singleimg_bandcal(rgbimg,grayimg,grayrgb)
        workimg=single_KMeans(keyworkbandarrays,parameter['bandchoice'],parameter['classnum'],parameter['classpick'])
        workimg=singleremove(parameter['removevalue'],workimg)
        itertime=10
        #coin=True
        #labels,border,colortable,coinparts=tkintersinglecore.init(workimg,caliberation,itertime,coin)
        labels,border,colortable,greatareas,tinyareas,coinparts,labeldict=tkintercorestat.init(workimg,workimg,'',None,itertime,coin)
        print(colortable)
        #tempbands=np.zeros((int(rgbimg.size[0]/ratio),int(rgbimg.size[1]/ratio),3),np.uint8)
        #tempbands[:,:,0]=border
        #tempbands[:,:,1]=workimg*150
        #tempbands[:,:,2]=border
        #tempbands=np.where(tempbands==0,255,tempbands)
        #tempbands=tempbands.astype('float32')
        #originsizetemp=cv2.resize(src=tempbands,dsize=(RGB.size[1],RGB.size[0]),interpolation=cv2.INTER_LINEAR)
        #originsizetemp=originsizetemp.astype('uint8')
        #pyplt.imsave(out_png,tempbands)
        #image=cv2.imread(out_png)
        #image=Image.fromarray(image)
        #image=ImageTk.PhotoImage(image)
        image=Image.open(key)
        image=image.resize((int(rgbimg.size[1]/ratio),int(rgbimg.size[0]/ratio)),Image.BILINEAR)
        image=ImageTk.PhotoImage(image)
        if panelA is not None:
            panelA.destroy()
        panelA=Label(ctr_left)
        panelA.image=image
        #panelA.grid(row=1,column=0)
        #panelA.bind('<Motion>',extractimg)
        #panelC=Toplevel()
        #extractframe = Label(panelC)
        #canvasframe=Label(extractframe)

        extractw = Canvas(panelA, width=int(rgbimg.size[1]/ratio), height=int(rgbimg.size[0]/ratio),bg='systemTransparent')
        #extractw.create_image(0, 0, image=image, anchor=NW)
        viewselection=IntVar()
        textselection=IntVar()
        tup=(labels,border,colortable,key,coinparts)

        #extractframe.pack()
        #canvasframe.pack()

        #extractw.pack()
        tup=(key,coinparts,labeldict)
        showcountingstat(tup,usertype)
        time.sleep(5)
        #x=extractw.winfo_rootx()+extractw.winfo_x()
        #y=extractw.winfo_rooty()+extractw.winfo_y()
        #x1=x+int(rgbimg.size[1]/ratio)
        #y1=y+int(rgbimg.size[0]/ratio)
        #canvasbox=(x,y,x1,y1)
        #file_origin,file_ext=os.path.splitext(key)
        #print(file_origin+'_countingresult_'+file_ext)
        #img=ImageGrab.grab(bbox=canvasbox)
        #img=img.convert("RGB")
        #img.save(file_origin+'_countingresult_'+file_ext)
        #panelC.destroy()
        #img=Image.open(file_origin+'_countingresult_'+file_ext)
        #img.show()




    #process images using parameters
    #run image one at a time call startsingleextract
    pass

#def startextract(modified_tif,mapfile,tiflayers,iteration):
def startsingleextract(modified_tif,mapfile,tiflayers,usertype):
    global panelA,labels,border,colortable,greatareas,tinyareas,caliberation,kernelsizes,multi_results
    kernelsizes.clear()
    multi_results.clear()
    #itertime=int(iteration.get())
    itertime=10
    #tkinterclustering.init(modified_tif,mapfile,tiflayers,itertime)
    #coin=False
    labels,border,colortable,greatareas,tinyareas,coinparts,labeldict=tkintercorestat.init(seg_tif,valid_tif,mapfile,tiflayers,itertime,coin)
    unique,count=np.unique(labels,return_counts=True)
    meanpixel=sum(count[1:])/len(count[1:])
    sigma=np.std(count[1:])
    maxpixel=max(count[1:])
    minpixel=min(count[1:])
    caliberation.update({'mean':meanpixel})
    caliberation.update({'max':maxpixel})
    caliberation.update({'min':minpixel})
    caliberation.update({'sigma':sigma})
    print(colortable)
    tempbands=np.zeros((int(RGB.size[0]/ratio),int(RGB.size[1]/ratio),3),np.uint8)
    tempbands[:,:,0]=border
    tempbands[:,:,1]=modified_tif*150
    tempbands[:,:,2]=border
    tempbands=np.where(tempbands==0,255,tempbands)
    #tempbands=tempbands.astype('float32')
    #originsizetemp=cv2.resize(src=tempbands,dsize=(RGB.size[1],RGB.size[0]),interpolation=cv2.INTER_LINEAR)
    #originsizetemp=originsizetemp.astype('uint8')
    #pyplt.imsave(out_png,tempbands)
    #image=cv2.imread(out_png)
    #image=Image.fromarray(image)
    originfile,extension=os.path.splitext(currentfilename)
    if usertype==1:
        image=Image.open(currentfilename)
        image=image.resize((int(RGB.size[1]/ratio),int(RGB.size[0]/ratio)),Image.BILINEAR)
    if usertype==2:
        pngbands=np.zeros((RGB.size[0],RGB.size[1]))
        if extension=='.tif':
            pngbands[:,:]=bandarrays['Greenness']
            print(pngbands)
            pyplt.imsave('tempcroptif.png',pngbands)
            image=cv2.imread('tempcroptif.png')
            image=Image.fromarray(image)
        else:
            image=Image.open(currentfilename)

        #pngbands=np.where(pngbands==1e-6,0,pngbands)
        #pngbands=pngbands.astype('uint8')

        image=image.resize((int(RGB.size[1]/ratio),int(RGB.size[0]/ratio)),Image.BILINEAR)
    image=ImageTk.PhotoImage(image)
    if panelA is not None:
        panelA.destroy()
    panelA=Label(ctr_left)
    panelA.image=image

    #panelA.grid(row=1,column=0)
    #panelA.bind('<Motion>',extractimg)
    #panelC=Toplevel()
    #extractframe = LabelFrame(panelC)
    #viewselectionframe=LabelFrame(extractframe,text='View Results')
    viewselection=IntVar()
    textselection=IntVar()
    #pixelview=Radiobutton(viewselectionframe,text='pixelview',value=0,variable=viewselection)
    #squareview=Radiobutton(viewselectionframe,text='squareview',value=1,variable=viewselection)
    #allsquareview=Radiobutton(viewselectionframe,text='all-boundingbox',value=2,variable=viewselection)
    #allpixelview=Radiobutton(viewselectionframe,text='all-highlighted',value=3,variable=viewselection)
    #textview=Checkbutton(viewselectionframe,text='label',variable=textselection)
    #canvasframe=LabelFrame(extractframe)
    extractw = Canvas(panelA, width=int(RGB.size[1]/ratio)+50, height=int(RGB.size[0]/ratio)+50, bg='white')
    tup=(labels,border,colortable,currentfilename,coinparts)
    #extractw.bind('<Button-1>',lambda event,arg=tup:extractimg(event,tup))
    #extractw.create_image(0, 0, image=image, anchor=NW)
    #buttonframe=LabelFrame(extractframe)
    #extrabutton=Button(buttonframe,text='reset',command=partial(extrareset,extractw))
    #modifytup=(labels,image,colortable,buttonframe)
    #modifybutton=Button(buttonframe,text='Start Modify',command=partial(modifyresult,modifytup))
    #recompubutton=Button(buttonframe,text='Re-Compute')
    #okeybutton = Button(panelC, text='OK', command=partial(canvasok, panelC, manultoolframe))
    #restartbutton = Button(panelC, text='Reset', command=partial(canvasreset, inputimage))

    #toolselectionframe.grid(row=0, column=0, columnspan=2)
    #eraserbutton.pack(side=LEFT)
    #paintbutton.pack(side=LEFT)
    #extractframe.pack()
    #viewselectionframe.pack()
    #canvasframe.pack()
    #buttonframe.pack()

    #pixelview.grid(row=0,column=0)
    #squareview.grid(row=0,column=1)
    #allsquareview.grid(row=0,column=2)
    #allpixelview.grid(row=0,column=3)
    #textview.grid(row=0,column=4)
    
    #extractw.pack()


    '''
    trainingdataset=showcounting(tup,usertype)

    with open('trainingdataset.csv','w') as f:
        writer=csv.writer(f)
        for item in trainingdataset:
            writer.writerow(item)
    f.close()
    '''
    startbatchextract(usertype)


    #extrabutton.grid(row=0,column=0,padx=5,pady=3)
    #modifybutton.grid(row=0,column=1,padx=5,pady=3)
    #recompubutton.grid(row=0,column=2,padx=5,pady=3)

    tup=(currentfilename,coinparts,labeldict)
    showcountingstat(tup,usertype)






def drawboundary(usertype):
    global panelB,treelist
    if usertype>0:
        #inserttop(4)
        #insertbottom(4)
        inserttop(2,usertype)
        insertbottom(2,usertype)
    panelTree.grid_forget()
    #if mapfile=='':
    #    messagebox.showerror('Map File',message='No map is loaded, pixels will be labeled numerically.')
        #return
    #else:
    #    if threshold is None:
    #        modified_tif=np.where(modified_tif<0,0,1)
        #modified_tif=fillholes(modified_tif)
    if confirm==False:
        messagebox.showerror('Invalid',message='Need to confirm settings in last step.')
        return
    if len(tiflayers)==0:
        messagebox.showerror('No image',message='Need to load image.')
        return
    if type(seg_tif)==type(None):
        messagebox.showerror('No Seg map',message='Need to set segmentation map.')
        return
    if type(valid_tif)==type(None):
        messagebox.showerror('No Valid map',message='Need to set Validation map.')
        return
    '''
    out_fn='tempNDVI400x400.tif'
    gtiffdriver=gdal.GetDriverByName('GTiff')
    out_ds=gtiffdriver.Create(out_fn,modified_tif.shape[1],modified_tif.shape[0],1,3)
    out_band=out_ds.GetRasterBand(1)
    #out_band.WriteArray(modified_tif)
    out_band.WriteArray(seg_tif)
    out_ds.FlushCache()
    '''

    '''
    for i in range(len(tiflayers)):
        out_fn='layersclass'+str(i)+'.tif'
        gtiffdriver=gdal.GetDriverByName('GTiff')
        out_ds=gtiffdriver.Create(out_fn,tiflayers[i].shape[1],tiflayers[i].shape[0],1,3)
        out_band=out_ds.GetRasterBand(1)
        out_band.WriteArray(tiflayers[i])
        out_ds.FlushCache()
    '''
    if panelB is not None:
        panelB.destroy()
    panelB=LabelFrame(ctr_right)
    panelB.grid(row=0,column=0)
    #panelA.image=modified_tif
    start=LabelFrame(panelB)#,text='Set # iteration')
    start.grid(row=0,column=0)
    iterdes=Label(start,text='Iteration')
    #iterdes.pack(side=LEFT)
    iterentry=Entry(start)
    #iterentry.pack(side=LEFT)
    iterentry.insert(END,10)
    #iterbutton=Button(start,text='Start',command=partial(startextract,modified_tif,mapfile,tiflayers,iterentry))
    iterbutton=Button(start,text='Start !',command=partial(startsingleextract,modified_tif,mapfile,tiflayers,usertype))
    iterbutton.pack(side=LEFT)
    #manual=LabelFrame(panelB,text='Combine and Divide (OPTIONAL)')
    #manual.grid(row=1,column=0)
    #manualdes=Label(manual,text='If there are borders placed incorrectly, click the button to modify that.')
    #manualdes.grid(row=0,column=0,columnspan=1)
    #manualbutton=Button(manual,text='Modify')
    #manualbutton.grid(row=1,column=1)
    #images=None
    #batchbutton=Button(start,text='Start (batch image)',command=startbatchextract)
    #batchbutton.pack(side=LEFT)



    #tkintercore.init(modified_tif,mapfile,tiflayers)
    #if rotation!=0:
    #    smalltif=rotat(rotation,newNDVI)
    #else:
    #    smalltif=cv2.resize(newNDVI,(450,450),interpolation=cv2.INTER_NEAREST)



def setthreshold():
    global threshold,panelA,modified_tif
    threshold=float(thresholdentry.get())
    modified_tif=np.where(modified_tif<threshold,0,modified_tif)
    pyplt.imsave(out_png,modified_tif)
    image=cv2.imread(out_png)
    image=Image.fromarray(image)
    image=ImageTk.PhotoImage(image)
    if panelA is not None:
        panelA.destroy()
    panelA=Label(ctr_left,image=image)
    panelA.image=image
    panelA.grid(row=1,column=0)


def resetrotate():
    global modified_tif,panelA,rotation,workbandarrays
    rotation=0
    if len(bandarrays.keys())==1:
        bandname='LabOstu'
    else:
        try:
            bandname=treelist.selection_get()
        except:
            bandname='LabOstu'
    band=bandarrays[bandname]
    keys=workbandarrays.keys()
    for key in keys:
        workbandarrays[key]=cv2.resize(bandarrays[key],(450,450),interpolation=cv2.INTER_LINEAR)
    modified_tif=cv2.resize(band,(450,450),interpolation=cv2.INTER_NEAREST)
    pyplt.imsave(out_png,modified_tif)
    image=cv2.imread(out_png)
    image=Image.fromarray(image)
    image=ImageTk.PhotoImage(image)
    if panelA is not None:
        panelA.destroy()
    panelA=Label(ctr_left,image=image)
    panelA.image=image
    panelA.grid(row=1,column=0)

def antirotatecv(degreeentry):
    global modified_tif,panelA,rotation,workbandarrays
    try:
        if type(rotation)==type(None):
            rotation=float(degreeentry.get())
        else:
            rotation+=float(degreeentry.get())
        localrotate=float(degreeentry.get())
    except ValueError:
        messagebox.showerror('Error',message='No Degree entry!')
        return
    center=(225,225)
    #center=(RGB.size[0]/2,RGB.size[1]/2)
    M=cv2.getRotationMatrix2D(center,localrotate,1.0)
    keys=workbandarrays.keys()
    for key in keys:
        workbandarrays[key]=cv2.warpAffine(workbandarrays[key],M,dsize=(450,450),flags=cv2.INTER_LINEAR)
    #modified_tif=cv2.warpAffine(modified_tif,M,dsize=(450,450),flags=cv2.INTER_LINEAR)
    try:
        key=treelist.selection_get()
        #modified_tif=cv2.resize(workbandarrays[key],(450,450),interpolation=cv2.INTER_LINEAR)
        modified_tif=workbandarrays[key]
    except:
        modified_tif=workbandarrays['LabOstu']
        #modified_tif=cv2.resize(workbandarrays['LabOstu'],(450,450),interpolation=cv2.INTER_LINEAR)
    pyplt.imsave(out_png,modified_tif)
    image=cv2.imread(out_png)
    image=Image.fromarray(image)
    image=ImageTk.PhotoImage(image)
    if panelA is not None:
        panelA.destroy()
    panelA=Label(ctr_left,image=image)
    panelA.image=image
    panelA.grid(row=1,column=0)

def rotationpanel(usertype):
    global panelB,panelTree,root
    if usertype>0:
        inserttop(2)
        insertbottom(2)
    if len(bandarrays)==0:
        messagebox.showerror('No 2D tif',message='No 2D tif is loaded.')
        return
    if panelB is not None:
        panelB.destroy()
    panelTree.grid_forget()

    panelB=Label(ctr_right)
    panelB.grid(row=0,column=0)
    anticlockrotdes=Label(master=panelB,text='Rotation Degree',justify=LEFT)
    anticlockrotdes.grid(row=1,column=0)
    degreeentry=Entry(master=panelB)
    degreeentry.grid(row=1,column=1)
    anticlockrotbutton=Button(master=panelB,text='Rotate',command=partial(antirotatecv,degreeentry))
    anticlockrotbutton.grid(row=1,column=2)
    #cloclrotdes=Label(master=panelB,text='Clockwise Rotation')
    #cloclrotdes.grid(row=2,column=0)
    #clockrotbutton=Button(master=panelB,text='+15 degree',command=rotate)
    #clockrotbutton.grid(row=2,column=1)
    notedes=Label(master=panelB,text='Note: Rotations lose information of your image.\n Please rotate to the extent you need.'
                                     '\n Once the last plot of your top-down first line\n is higher than the first plot of your second line,\n it is fine',width=35,justify=LEFT,anchor=W)
    notedes.grid(row=2,column=0,columnspan=1)
    resetdes=Label(master=panelB,text='Reset Image(if needed)')
    resetdes.grid(row=3,column=0)
    resetbutton=Button(master=panelB,text='Reset',command=resetrotate)
    resetbutton.grid(row=3,column=1)

def widthinterpreter(width):
    if width=='1':
        return 1
    if width=='2':
        return 2
    if width=='3':
        return 3
    if width=='4':
        return 4
    if width=='5':
        return 5
def release(e):
    global w
    #print('click at',e.x,e.y)
    w.old_coords=None
    w.old_drawcoords=None

def drag(e,widthchoice):
    global w
    #global mousex,mousey
    mousex=e.x
    mousey=e.y
    #if w.old_coords:
        #x1,y1=w.old_coords
        #w.create_line(mousex,mousey,x1,y1,width=widthchoice.get())
    erasrect=w.create_rectangle(mousex,mousey,mousex+widthchoice.get(),mousey+widthchoice.get(),fill='black',outline="")
    w.contents.append(erasrect)
   # w.old_coords=mousex,mousey
    for i in range(widthchoice.get()+1):
        erasercoord.append([e.x,e.y])
        erasercoord.append([e.x,e.y+i])
        erasercoord.append([e.x+i,e.y])
        erasercoord.append([e.x+i,e.y+i])
        print('click at',e.x+i,e.y+i)
    w.laststep = 4 * (widthchoice.get() + 1)
    w.lastaction='erase'

def drawdrag(e,widthchoice):
    global w
    mousex=e.x
    mousey=e.y
    #if w.old_drawcoords:
        #x1,y1=w.old_drawcoords
        #w.create_line(mousex,mousey,x1,y1,fill='green',width=widthchoice.get())
    drawrect=w.create_rectangle(mousex,mousey,mousex+widthchoice.get(),mousey+widthchoice.get(),fill='yellow',outline="")
    w.contents.append(drawrect)
    #w.old_drawcoords=mousex,mousey
    for i in range(widthchoice.get()+1):
        paintrecord.append([e.x,e.y])
        paintrecord.append([e.x,e.y+i])
        paintrecord.append([e.x+i,e.y])
        paintrecord.append([e.x+i,e.y+i])
        print('draw at',e.x+i,e.y+i)
    w.laststep=4*(widthchoice.get()+1)
    w.lastaction='draw'

def eraser(panelC):
    global w
    w.config(cursor='cross')

    widthframe=LabelFrame(panelC,text='Width Selection')
    widthframe.grid(row=1,columnspan=4)
    widthchoice=IntVar()
    widthchoice.set(1)
    for i in range(5):
        widthbutton=Radiobutton(widthframe,text=str(i+1),value=i,variable=widthchoice)
        widthbutton.pack(side=LEFT)
        if i==0:
            widthbutton.select()
    w.bind('<ButtonRelease-1>',release)
    w.bind('<B1-Motion>',lambda event,arg=widthchoice:drag(event,widthchoice))

def paint(panelC):
    global w
    w.config(cursor='pencil')

    widthframe=LabelFrame(panelC,text='Width Selection')
    widthframe.grid(row=1,columnspan=4)
    widthchoice=IntVar()
    for i in range(5):
        widthbutton=Radiobutton(widthframe,text=str(i+1),value=i,variable=widthchoice)
        widthbutton.pack(side=LEFT)
        if i==0:
            widthbutton.select()
    w.bind('<ButtonRelease-1>',release)
    w.bind('<B1-Motion>',lambda event,arg=widthchoice:drawdrag(event,widthchoice))

def canvasreset(inputimage):
    global w,erasercoord,paintrecord
    w.old_coords=None
    w.create_image(0,0,image=inputimage,anchor=NW)
    erasercoord=[]
    paintrecord=[]

def canvaslaststep():
    global w
    w.delete(w.contents.pop(-1))
    if w.lastaction=='draw':
        for i in range(4*w.laststep):
            paintrecord.pop(-1)
    if w.lastaction=='erase':
        for i in range(4*w.laststep):
            erasercoord.pop(-1)


def canvasok(window,manualtoolframe):
    global panelA,modified_tif
    if len(erasercoord)>0:
        for i in range(len(erasercoord)):
            x=erasercoord[i][0]
            y=erasercoord[i][1]
            if x>=0 and x<RGB.size[1] and y>=0 and y<RGB.size[0]:
                modified_tif[y,x]=0
    if len(paintrecord)>0:
        for i in range(len(paintrecord)):
            x=paintrecord[i][0]
            y=paintrecord[i][1]
            if x>=0 and x<RGB.size[1] and y>=0 and y<RGB.size[0]:
                modified_tif[y,x]=1
    pyplt.imsave(out_png,modified_tif)
    image=cv2.imread(out_png)
    image=Image.fromarray(image)
    image=ImageTk.PhotoImage(image)
    if panelA is not None:
        panelA.destroy()
    panelA=Label(ctr_left,image=image)
    panelA.image=image
    panelA.grid(row=0,column=0)
    manualtoolframe.grid_forget()
    window.destroy()


def manualeraser(inputimage,manultoolframe):
    global temptif,panelA,modified_tif,plantchoice,tiflayers,w
    panelC=Toplevel()
    toolselectionframe=LabelFrame(panelC,text='Tools')
    toolchoice=None
    eraserbutton=Radiobutton(toolselectionframe,text='Eraser',value=0,command=partial(eraser,panelC),variable=toolchoice)
    paintbutton=Radiobutton(toolselectionframe,text='Pen',value=1,command=partial(paint,panelC),variable=toolchoice)
    canvasframe=LabelFrame(panelC,text='Canvas')
    w=Canvas(canvasframe,width=RGB.size[1],height=RGB.size[0],bg='white')
    #w.old_coords=None
    #w.old_drawcoords=None
    #w.bind('<Enter>',w.config(cursor="cross"))
    image = cv2.imread(out_png)
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)

    #w.bind('<Leave>',cursor="arrow")
    w.create_image(0,0,image=image,anchor=NW)
    okeybutton=Button(panelC,text='OK',command=partial(canvasok,panelC,manultoolframe))
    restartbutton=Button(panelC,text='Reset',command=partial(canvasreset,image))
    #gobackbutton=Button(panelC,text='GoBack',command=canvaslaststep)

    toolselectionframe.grid(row=0,column=0,columnspan=3)
    eraserbutton.pack(side=LEFT)
    paintbutton.pack(side=LEFT)

    canvasframe.grid(row=2,column=0,columnspan=3)
    w.grid(row=0,column=0)
    w.contents=[]
    w.laststep=0
    w.lastaction=''

    #gobackbutton.grid(row=3,column=1)
    okeybutton.grid(row=3,column=2)
    restartbutton.grid(row=3,column=0)
    #CreateToolTip(canvasframe,'Modify areas here')
    #CreateToolTip(okeybutton,'Save modification and exit')
    #CreateToolTip(restartbutton,'Reset to origin image')

def setsegmap(plantchoice):
    global seg_tif,tiflayers
    seg_tif=None
    tiflayers = []
    for i in range(len(plantchoice)):
        tup=plantchoice[i]
        if '1' in tup:
            zerograph = np.zeros(temptif.shape)
            zerograph = np.where(temptif == i, 1, zerograph)
            tiflayers.append(zerograph)
    seg_tif=modified_tif

def setvalidmap():
    global valid_tif
    valid_tif=None
    valid_tif=modified_tif

def scalarbarvalue(blurvalue):
    global panelA
    print(int(blurvalue))
    tifcopy=modified_tif.astype('float32')
    blurtif=np.zeros(tifcopy.shape,dtype='float32')

    blurtif=cv2.bilateralFilter(tifcopy,int(blurvalue),75,75)
    #modified_tif=blurtif
    blurtif=np.where(blurtif>0,1,0)
    pyplt.imsave(out_png, blurtif)
    image = cv2.imread(out_png)
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)
    if panelA is not None:
        panelA.destroy()
    panelA = Label(ctr_left, image=image)
    panelA.image = image
    panelA.grid(row=0, column=0)

def resetblur():
    global panelA
    pyplt.imsave(out_png, modified_tif)
    image = cv2.imread(out_png)
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)
    if panelA is not None:
        panelA.destroy()
    panelA = Label(ctr_left, image=image)
    panelA.image = image
    panelA.grid(row=0, column=0)

def confirmblur(blurvalue):
    global panelA,modified_tif
    tifcopy = modified_tif.astype('float32')
    blurtif = np.zeros(tifcopy.shape, dtype='float32')
    blurtif = cv2.bilateralFilter(tifcopy, blurvalue.get(), 75, 75)
    blurtif=np.where(blurtif>0,1,0)
    modified_tif=blurtif
    pyplt.imsave(out_png, blurtif)
    image = cv2.imread(out_png)
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)
    if panelA is not None:
        panelA.destroy()
    panelA = Label(ctr_left, image=image)
    panelA.image = image
    panelA.grid(row=0, column=0)

def validwidget(widgets):
    global panelB
    manualtoolframe,blurframe,scalarbar,scalarbutto,scalarconfirm,mapsetframe,manualtoolbutton,segmapsetbutton,validmapsetbutton=widgets
    manualtoolframe.grid(row=7,columnspan=3)
    blurframe.grid(row=8,columnspan=3)
    scalarbar.pack(side=LEFT,padx=5,pady=5)
    scalarbutto.pack(side=LEFT,padx=5,pady=5)
    scalarconfirm.pack(side=LEFT,padx=5,pady=5)
    mapsetframe.grid(row=9,columnspan=3)
    manualtoolbutton.pack()
    segmapsetbutton.pack(side=LEFT,padx=5,pady=5)
    validmapsetbutton.pack(side=LEFT,padx=5,pady=5)

    pass

def forgetadvanceview(tupofwidgets):
    global panelB
    manualtoolframe,blurframe,scalarbar,scalarbutto,scalarconfirm,mapsetframe,manualtoolbutton,segmapsetbutton,validmapsetbutton=tupofwidgets
    manualtoolframe.grid_forget()
    blurframe.grid_forget()
    scalarbar.pack_forget()
    scalarbutto.pack_forget()
    scalarconfirm.pack_forget()
    mapsetframe.grid_forget()
    manualtoolbutton.pack_forget()
    segmapsetbutton.pack_forget()
    validmapsetbutton.pack_forget()
    pass

def singleremove(removevalue,workimg):
    size=workimg.shape
    copyimg=workimg
    for i in range(int(removevalue)):
        copyimg[i,:]=0  #up
        copyimg[:,i]=0  #left
        copyimg[:,size[1]-1-i]=0 #right
        copyimg[size[0]-1-i,:]=0
    return copyimg

def confirmremove(refop,edgeop,plantchoice,choicelist,classpick,usertype):
    global panelA,modified_tif,parameter,coin,confirm
    confirm=True
    copytif=modified_tif
    size=copytif.shape
    #for i in range(int(removevalue.get())):
    if usertype==1:
        ref=refop.get()
        edge=edgeop.get()
        if ref=='1':
            coin=True
        if ref=='0':
            coin=False
        if edge=='1':
            removevalue=30
        if edge=='0':
            removevalue=0
        for i in range(removevalue):
            copytif[i,:]=0  #up
            copytif[:,i]=0  #left
            copytif[:,size[1]-1-i]=0 #right
            copytif[size[0]-1-i,:]=0
    if usertype==2:
        removevalue=0
        coin=False
    modified_tif=copytif
    pyplt.imsave(out_png,copytif)
    image = cv2.imread(out_png)
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)
    if panelA is not None:
        panelA.destroy()
    panelA = Label(ctr_left, image=image)
    panelA.image = image
    panelA.grid(row=0, column=0)
    setsegmap(plantchoice)
    setvalidmap()
    parameter.update({'bandchoice':choicelist})
    parameter.update({'classnum':len(plantchoice)})
    parameter.update({'classpick':classpick})
    parameter.update({'removevalue':removevalue})
    #tempband=workbandarrays['Greenness(seeds)']
    #tempband=workbandarrays['LabOstu(wheat,grains)']
    #reshapeimg=tempband.reshape(tempband.shape[0]*tempband.shape[1],1)
    #clf=KMeans(n_clusters=len(plantchoice),init='k-means++',n_init=10,random_state=0)
    #clf=KMeans(n_clusters=2,init='k-means++',n_init=10,random_state=0)
    #res=clf.fit_predict(reshapeimg)
    #classband=res.reshape(tempband.shape[0],tempband.shape[1])
    #selectedclass=plantchoice.index('1')
    #tempband=np.where(classband==selectedclass,tempband,0)
    #robert_edge=filters.roberts(tempband)
    #robert_edge=tkintercore.get_boundary(robert_edge)
    #print(maxvalue)
    #robert_edge=np.where(robert_edge<maxvalue,0,1)
    #pyplt.imsave('greenedge.png',robert_edge)
    #edgeimg=Image.open('greenedge.png')
    #edgeimg.show()

    return



def removesquareedge(removeedgevalue):
    global panelA
    print(int(removeedgevalue))
    copytif=np.copy(modified_tif)
    size=copytif.shape
    for i in range(int(removeedgevalue)):
        copytif[i,:]=0  #up
        copytif[:,i]=0  #left
        copytif[:,size[1]-1-i]=0 #right
        copytif[size[0]-1-i,:]=0
    pyplt.imsave(out_png,copytif)
    image = cv2.imread(out_png)
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)
    if panelA is not None:
        panelA.destroy()
    panelA = Label(ctr_left, image=image)
    panelA.image = image
    panelA.grid(row=0, column=0)

def generateplant(checkboxdict,choicelist,usertype):
    global temptif,panelA,modified_tif,plantchoice,panelB,erasercoord,paintrecord,confirm
    if usertype==1:
        confirm=False
    if usertype==2:
        confirm=True
    plantchoice=[]
    keys=checkboxdict.keys()
    for key in keys:
        plantchoice.append(checkboxdict[key].get())
    modified_tif=np.zeros(temptif.shape)
    miniclass=0
    arearank={}
    for i in range(len(plantchoice)):
        tup=plantchoice[i]
        if '1' in tup:
            modified_tif=np.where(temptif==i,1,modified_tif)
            miniclass=i
        pixelloc=np.where(temptif==i)
        pixelnum=len(pixelloc[0])
        temparea=float(pixelnum/(temptif.shape[0]*temptif.shape[1]))
        arearank.update({i:temparea})
    sortarearank=sorted(arearank,key=arearank.get)
    i=0
    for ele in sortarearank:
        if ele==miniclass:
            classpick=i
        else:
            i+=1



    #resizemodifiedtif=cv2.resize(modified_tif,(RGB.size[1],RGB.size[0]),interpolation=cv2.INTER_LINEAR)
    #resizemodifiedtif=np.array(bandarrays['LabOstu'])*resizemodifiedtif
    #resizemodifiedtif=np.where(resizemodifiedtif<=float(threshold.get()),0,1)
    #modified_tif=cv2.resize(resizemodifiedtif,(450,450),interpolation=cv2.INTER_LINEAR)
    #baseimg=cv2.resize(np.array(bandarrays['LabOstu']),(450,450),interpolation=cv2.INTER_LINEAR)
    #baseimg=np.where(baseimg<=float(threshold.get()),0,baseimg)
    #modified_tif=np.where(modified_tif<=float(threshold.get()),0,1)


    pyplt.imsave(out_png,modified_tif)
    image=cv2.imread(out_png)
    image=Image.fromarray(image)
    image=ImageTk.PhotoImage(image)
    if panelA is not None:
        panelA.destroy()
    panelA=Label(ctr_left,image=image)
    panelA.image=image
    panelA.grid(row=0,column=0)

    #for key in keys:
    #    checkboxdict[key]=Variable()

    erasercoord=[]
    paintrecord=[]
    print(temptif)
    pixelnum=np.count_nonzero(modified_tif)
    pixelarea=float(pixelnum/(modified_tif.shape[0]*modified_tif.shape[1]))
    print('pixelarea='+str(pixelarea))
    setsegmap(plantchoice)
    setvalidmap()
    v=StringVar()
    edge=StringVar()
    #removeedge=LabelFrame(panelB,text='Remove Square Edge')
    if usertype==1:
        removeedge=LabelFrame(panelB,text='Settings')
        removeedge.grid(row=6,columnspan=4)
        refframe=LabelFrame(removeedge)
        refframe.pack()
        refoption=[('Coin as Ref','1'),('A4 paper as Ref','0')]

        v.set('1')
        for text,mode in refoption:
            b=Radiobutton(refframe,text=text,variable=v,value=mode)
            b.pack(side=LEFT)
        edgeframe=LabelFrame(removeedge)
        edgeframe.pack()
        edgeoption=[('Remove edge','1'),('Keep same','0')]

        edge.set('0')
        for text,mode in edgeoption:
            b=Radiobutton(edgeframe,text=text,variable=edge,value=mode)
            b.pack(side=LEFT)
    #removeedgevalue=IntVar()
    #removeedgebar=Scale(removeedge,from_=0,to=25,tickinterval=1,length=450,orient=HORIZONTAL,variable=removeedgevalue,command=removesquareedge)
    #removeedgebar.pack(side=LEFT)

        removeedgeconfirm=Button(removeedge,text='Confirm',command=partial(confirmremove,v,edge,plantchoice,choicelist,classpick,usertype))
        removeedgeconfirm.pack()
    advanceframe=LabelFrame(panelB,text='More Advanced Functions')
    #advanceframe.grid(row=6,columnspan=3)

    manualtoolframe=LabelFrame(panelB,text='Manually modify pre-processed image (OPTIONAL)')

    manualtoolbutton=Button(manualtoolframe,text='Modify image',command=partial(manualeraser,image,manualtoolframe))
    print(plantchoice)
    #manualeraser(image)

    blurframe=LabelFrame(panelB,text='Blur Control')

    blurvalue=IntVar()
    scalarbar=Scale(blurframe,from_=1,to=10,tickinterval=1,length=150,orient=HORIZONTAL,variable=blurvalue,command=scalarbarvalue)
    scalarbar.set(0)

    scalarbutto=Button(blurframe,text='Reset',command=resetblur)

    scalarconfirm=Button(blurframe,text='Confirm',command=partial(confirmblur,blurvalue))


    mapsetframe=LabelFrame(panelB,text='Map setting')

    segmapsetbutton=Button(mapsetframe,text='Set as Seg Map',command=partial(setsegmap,plantchoice))
    validmapsetbutton=Button(mapsetframe,text='Set as Validation Map',command=setvalidmap)


    advancedtup=(manualtoolframe,blurframe,scalarbar,scalarbutto,scalarconfirm,mapsetframe,manualtoolbutton,segmapsetbutton,validmapsetbutton)
    choice=IntVar()
    currentview=Radiobutton(advanceframe,text='Simplified',value=0,command=partial(forgetadvanceview,advancedtup),variable=choice)
    advancedview=Radiobutton(advanceframe,text='Advanced functions',command=partial(validwidget,advancedtup),variable=choice)
    #currentview.pack(side=LEFT,padx=5,pady=5)
    #advancedview.pack(side=LEFT,padx=5,pady=5)

def single_KMeans(singleworkbandarrays,bandchoice,classnum,classpick):
    numindec=len(bandchoice)
    reshapeworkimg=np.zeros((singleworkbandarrays[bandchoice[0]].shape[0]*singleworkbandarrays[bandchoice[0]].shape[1],numindec))
    for i in range(numindec):
        tempband=singleworkbandarrays[bandchoice[i]]
        reshapeworkimg[:,i]=tempband.reshape(tempband.shape[0]*tempband.shape[1],1)[:,0]
    clusternumber=classnum
    clf=KMeans(n_clusters=clusternumber,init='k-means++',n_init=10,random_state=0)
    labels=clf.fit_predict(reshapeworkimg)
    temptif = labels.reshape(singleworkbandarrays[bandchoice[0]].shape[0], singleworkbandarrays[bandchoice[0]].shape[1])
    arearank={}
    for i in range(clusternumber):
        pixelloc=np.where(labels==i)
        pixelnum=len(pixelloc[0])
        temparea=float(pixelnum/(temptif.shape[0]*temptif.shape[1]))
        arearank.update({i:temparea})
    sortarearank=sorted(arearank,key=arearank.get)
    miniclass=sortarearank[classpick]
    #i=0
    #for ele in sortarearank:
    #    if i==classpick:
    #        miniclass=ele
    #    else:
    #        i+=1

    workimg=np.zeros(temptif.shape)
    workimg=np.where(temptif==miniclass,1,workimg)
    return workimg


def kmeansclassify(bandchoice,classentry,usertype):
    global modified_tif,temptif,panelB,panelA
    wchildren=panelB.winfo_children()
    #for i in range(len(wchildren)):
    #    if wchildren[i].name=='kmeansresult':
    #        wchildren[i].grid_forget()

    keys=bandchoice.keys()
    numindec=0
    choicelist=[]
    for key in keys:
        tup=bandchoice[key].get()
        if '1' in tup:
            numindec+=1
            choicelist.append(key)
    reshapemodified_tif=np.zeros((modified_tif.shape[0]*modified_tif.shape[1],numindec))
    for i in range(numindec):
        #tempband=bandarrays[choicelist[i]]
        tempband=workbandarrays[choicelist[i]]
        #tempband=cv2.resize(tempband,(450,450),interpolation=cv2.INTER_LINEAR)
        reshapemodified_tif[:,i]=tempband.reshape(tempband.shape[0]*tempband.shape[1],1)[:,0]




    #reshapemodified_tif=modified_tif.reshape(modified_tif.shape[0]*modified_tif.shape[1],1)
    clusternumber=int(classentry.get())
    clf=KMeans(n_clusters=clusternumber,init='k-means++',n_init=10,random_state=0)
    if len(choicelist)==0:
        messagebox.showerror('No Indices is selected',message='Please select indicies to do KMeans Classification.')
        return
    labels=clf.fit_predict(reshapemodified_tif)
    temptif = labels.reshape(modified_tif.shape[0], modified_tif.shape[1])
    '''
    out_fn = 'kmeansclassify.tif'
    gtiffdriver = gdal.GetDriverByName('GTiff')
    out_ds = gtiffdriver.Create(out_fn, modified_tif.shape[0], modified_tif.shape[1], 1, 3)
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(temptif)
    out_ds.FlushCache()
    '''
    #temptif=labels.reshape(modified_tif.shape[0],modified_tif.shape[1])

    checkboxframe=LabelFrame(panelB,text='Select classes (check/empty boxes to see results)',name='kmeansresult')
    checkboxframe.grid(row=5,columnspan=3)
    #checkboxframe.grid_propagate(False)
    checkboxdict={}
    pixelarea=1.0
    minipixelareaclass=0
    for i in range(clusternumber):
        pixelloc=np.where(labels==i)
        pixelnum=len(pixelloc[0])
        temparea=float(pixelnum/(modified_tif.shape[0]*modified_tif.shape[1]))
        if temparea<pixelarea:
            minipixelareaclass=i
            pixelarea=temparea

    miniclass='class '+str(minipixelareaclass+1)
    #checkboxdict[miniclass]=Variable(value='1')
    #generateplant(checkboxdict)

    for i in range(clusternumber):
        dictkey='class '+str(i+1)
        #if i==1 or i==2 or i==4:
        #    tempdict={dictkey:Variable(value='1')}
        #else:
        tempdict={dictkey:Variable()}
        checkboxdict.update(tempdict)
        ch=ttk.Checkbutton(checkboxframe,text=dictkey,command=partial(generateplant,checkboxdict,choicelist,usertype),variable=checkboxdict[dictkey])
        if usertype==2 and defaultfile:
            if dictkey=='class 1' or dictkey=='class 5':
                ch.invoke()
        else:
            if dictkey==miniclass:
                ch.invoke()
        ch.grid(row=int(i/5),column=int(i%5))
    #generateplantargs=partial(generateplant,ch0,ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8,ch9)
    #genchbutton=Button(checkboxframe,text='Generate plant',command=partial(generateplant,checkboxdict))
    #genchbutton.grid(row=0,column=5)
    #gendes=Label(checkboxframe,text='(Light color represents plant)')
    #gendes.grid(row=1,column=5)
    #CreateToolTip(checkboxframe,'Select classes to check results of single class or combined classes')

    #modified_tif=np.where((temptif==3) | (temptif==2),1,0)
    #plantmap=LabelFrame(panelB,text='map plant scheme to plants (no boundary)')
    #plantmap.grid(row=7,column=1)
    #mapplant=Button(plantmap,text='Map planting shceme',command=drawboundary)
    #mapplant.grid(row=0,column=0)
    generateplant(checkboxdict,choicelist,usertype)

    return checkboxframe


def medianblurplus(entbox):
    global modified_tif,panelA
    val=int(entbox.get("1.0",END))
    entbox.insert(str(val))
    modified_tif=cv2.medianBlur(modified_tif,val+1)
    panelA.configure(image=modified_tif)

def onFrameConfigure(inputcanvas):
    '''Reset the scroll region to encompass the inner frame'''
    inputcanvas.configure(scrollregion=inputcanvas.bbox(ALL))

def autosetclassnumber(entbox,number):
    choice=[]
    keys=number.keys()
    for key in keys:
        choice.append(number[key].get())
    #print(choice)
    if '0' not in choice and '' not in choice:
        entbox.delete(0,END)
        entbox.insert(END,5)
        return
    if '1' in choice:
        if choice.index('1')==0:
            entbox.delete(0,END)
            entbox.insert(END,2)
            return
        if choice.index('1')==1:
            entbox.delete(0,END)
            entbox.insert(END,2)
            return
    if '1' not in choice:
        entbox.delete(0,END)
        return

    '''
    if 'wheat' in number:
        entbox.delete(0,END)
        entbox.insert(END,2)
    if 'seed' in number :
        entbox.delete(0,END)
        entbox.insert(END,5)
    '''

def KMeansPanel(usertype):
    global panelB,panelTree
    if usertype>0:
        #inserttop(3)
        #insertbottom(3)
        inserttop(1,usertype)
        insertbottom(1,usertype)
    if len(bandarrays)==0:
        #messagebox.showerror('No 2D layers',message='No 2D layer is loaded')
        #return
        bands_calculation(usertype)
    if panelB is not None:
        panelB.destroy()
    panelB=Label(ctr_right)
    panelB.grid(row=0,column=0)
    #panelTree.grid_forget()
    chframe=LabelFrame(panelB,text='Pick indicies below')
    chframe.grid(row=0,column=0,columnspan=2)
    chcanvas=Canvas(chframe,width=200,height=150,scrollregion=(0,0,400,400))
    chcanvas.pack(side=LEFT)
    chscroller=Scrollbar(chframe,orient=VERTICAL)
    chscroller.pack(side=RIGHT,fill=Y,expand=True)
    chcanvas.config(yscrollcommand=chscroller.set)
    chscroller.config(command=chcanvas.yview)
    contentframe=LabelFrame(chcanvas)
    chcanvas.create_window((4,4),window=contentframe,anchor=NW)
    contentframe.bind("<Configure>",lambda event,arg=chcanvas:onFrameConfigure(arg))
    bandkeys=bandarrays.keys()
    bandchoice={}
    clusternumberdes=Label(panelB,text='Set # of classes (wheat=2,seeds=5, wheat+seeds=5)')
    clusternumberdes.grid(row=3,column=0)
    clusternumberentry=Entry(panelB)
    if usertype==2 and defaultfile:
        clusternumberentry.delete(0,END)
        clusternumberentry.insert(END,5)
    if usertype==1 and defaultfile:
        clusternumberentry.delete(0,END)
        clusternumberentry.insert(END,2)
    for key in bandkeys:
        tempdict={key:Variable()}
        bandchoice.update(tempdict)
        ch=ttk.Checkbutton(contentframe,text=key,variable=bandchoice[key])#,command=partial(autosetclassnumber,clusternumberentry,bandchoice))
        if usertype==2 and defaultfile:
            if key=='MExG' or key=='NDVI':
                ch.invoke()
        if usertype==1 and defaultfile:
            if key=='NDI':
                ch.invoke()
        ch.pack(fill=X)


    #clusternumberentry.insert(END,2)
    clusternumberentry.grid(row=3,column=1)
    kmeansdes=Label(master=panelB,text='KMeans Clustering',justify=LEFT)
    kmeansdes.grid(row=4,column=0)
    kmeansbutton=Button(master=panelB,text='Classify',command=partial(kmeansclassify,bandchoice,clusternumberentry,usertype))
    kmeansbutton.grid(row=4,column=1)


def Generate_NDVI():
    global panelA,NDVI,panelB,modified_tif,thresholdentry,degreeentry
    if Infrared is not None and RGB is not None and Infrared.size==RGB.size:
        upper=Infrared.bands[0,:,:]-RGB.bands[0,:,:]
        lower=Infrared.bands[0,:,:]+RGB.bands[0,:,:]
        lower=np.where(lower==0,1,lower)
        NDVI=upper/lower
        '''
        out_fn='tempNDVI.tif'
        gtiffdriver=gdal.GetDriverByName('GTiff')
        out_ds=gtiffdriver.Create(out_fn,upper.shape[1],upper.shape[0],1,3)
        out_band=out_ds.GetRasterBand(1)
        out_band.WriteArray(NDVI)
        out_ds.FlushCache()
        '''

        modified_tif=cv2.resize(NDVI,(450,450),interpolation=cv2.INTER_NEAREST)
        pyplt.imsave(out_png,modified_tif)
        image=cv2.imread(out_png)
        image=Image.fromarray(image)
        image=ImageTk.PhotoImage(image)
        if panelA is not None:
            panelA.destroy()
        panelA=Label(ctr_left,image=image)
        panelA.image=image
        panelA.grid(row=1,column=0)
        #if panelB is not None:
        #    panelB.destory()

        #thresholddes=Label(master=panelB,text='Step 1. Set Threshold (0,1)',justify=LEFT)
        #thresholddes.grid(row=1,column=0)
        #thresholdentry=Entry(master=panelB)
        #thresholdentry.grid(row=1,column=1)
        #thresholdbutton=Button(master=panelB,text='Set',command=setthreshold)
        #thresholdbutton.grid(row=1,column=2)
        kmeansdes=Label(master=panelB,text='Step 2. Classify pixels (10 categories)',justify=LEFT)
        kmeansdes.grid(row=4,column=0)
        kmeansbutton=Button(master=panelB,text='Classify',command=kmeansclassify)
        kmeansbutton.grid(row=4,column=1)

        #boundarybutton=Button(master=panelB,text='draw boundary',command=drawboundary)
        #boundarybutton.grid(row=8,column=1)


def Open_HEIGHTfile(entbox):
    global Height,Heightlabel
    entbox.config(stat=NORMAL)
    entbox.delete("1.0",END)
    entbox.tag_config("wrap",wrap=CHAR)
    QGISNIRFILE=filedialog.askopenfilename()
    if len(QGISNIRFILE)>0:
        if QGISNIRFILE.endswith('.tif') is False:
            messagebox.showerror('Wrong file formate',message='Open tif formate file')
            return
        #messagebox.showinfo(title='Open Height file',message='open Height GeoTiff file:'+QGISNIRFILE)
        entbox.insert(END,QGISNIRFILE,"wrap")
        entbox.config(stat=DISABLED)
        '''
        NIRrsc=gdal.Open(QGISNIRFILE)
        NIRsize=(NIRrsc.RasterYSize,NIRrsc.RasterXSize)
        bands=[]
        bandrank={}
        for j in range(1):
            band=NIRrsc.GetRasterBand(j+1)
            stats = band.GetStatistics( True, True )
            print("[ STATS ] =  Minimum=%.3f, Maximum=%.3f, Mean=%.3f, StdDev=%.3f" % (
                    stats[0], stats[1], stats[2], stats[3] ))
            tempdict={j:stats[1]}
            bandrank.update(tempdict)
            nodata=band.GetNoDataValue()
            if type(nodata)==type(None):
                nodata=0
            if (nodata < stats[0] or nodata > stats[1]) and nodata != -10000 and nodata != -9999:
                nodata=0
            band=band.ReadAsArray()
            band=np.where(band==nodata,1e-6,band)
            bands.append(band)
        bands=np.array(bands)
        '''
        NIRrsc = rasterio.open(QGISNIRFILE)
        height = NIRrsc.height
        width = NIRrsc.width
        channel = NIRrsc.count
        print(NIRrsc)
        NIRbands = np.zeros((channel, height, width))
        NIRsize = (height, width)
        for j in range(5,6):
            band = NIRrsc.read(j + 1)
            band = np.where((band == 0) | (band == -10000) | (band == -9999), 1e-6, band)
            NIRbands[channel - (j + 1), :, :] = band
        Height=img(NIRsize,NIRbands)
        print(NIRbands)
        Heightlabel.text='Open file: '+QGISNIRFILE
    #Generate_NDVI()


def Open_Multifile(entbox,usertype):
    global Multiimage,Multigray,Multilabel,Multigrayrgb
    MULTIFILES=filedialog.askopenfilenames()
    Multiimage={}
    Multigray={}
    Multigrayrgb={}
    if len(MULTIFILES)>0:
        entbox.config(stat=NORMAL)
        entbox.delete("1.0",END)
        #entbox.tag_config("wrap",wrap=CHAR)
        for i in range(len(MULTIFILES)):
            try:
                Filersc=cv2.imread(MULTIFILES[i])
                #if MULTIFILES[i].endswith('.jpg') is True or MULTIFILES[i].endswith('.jpeg') is True or MULTIFILES[i].endswith('.png') is True or MULTIFILES[i].endswith('.JPG') is True:
                entbox.insert(END, MULTIFILES[i]+'\n', "wrap")
                Filersccopy=cv2.imread(MULTIFILES[i])
                height,width,channel=np.shape(Filersc)
                Filesize=(height,width)
                Filersc=cv2.cvtColor(Filersc,cv2.COLOR_BGR2RGB)
                Filersc=cv2.GaussianBlur(Filersc,(5,5),cv2.BORDER_DEFAULT)
                Filelab=cv2.cvtColor(Filersccopy,cv2.COLOR_BGR2Lab)
                Filelab=cv2.cvtColor(Filelab,cv2.COLOR_BGR2GRAY)
                Filelab=cv2.GaussianBlur(Filelab,(5,5),cv2.BORDER_DEFAULT)
                ostu=filters.threshold_otsu(Filelab)
                Filelab=Filelab.astype('float32')
                Filelab=Filelab/ostu
                Grayimg=img(Filesize,Filelab)
                Filergbgray=cv2.cvtColor(Filersccopy,cv2.COLOR_BGR2RGB)
                Filergbgray=cv2.cvtColor(Filergbgray,cv2.COLOR_BGR2GRAY)
                Filergbgray=cv2.GaussianBlur(Filergbgray,(5,5),cv2.BORDER_DEFAULT)
                ostu=filters.threshold_otsu(Filergbgray)
                Filergbgray=Filergbgray.astype('float32')
                Filergbgray=Filergbgray/ostu
                Grayrgbimg=img(Filesize,Filergbgray)
                #NIRrsc=cv2.cvtColor(NIRrsc,cv2.COLOR_BGR2LAB)
                #NIRrsc=cv2.cvtColor(NIRrsc,cv2.COLOR_BGR2RGB)
                Filebands=np.zeros((channel,height,width))
                for j in range(channel):
                    band=Filersc[:,:,j]
                    band=np.where(band==0,1e-6,band)
                    ostu=filters.threshold_otsu(band)
                    band=band/ostu
                    Filebands[j,:,:]=band
                Fileimg=img(Filesize,Filebands)

                #labelconteont=MULTIFILES[i]+","
                #Multilabel.text=labelconteont
                tempdict={MULTIFILES[i]:Fileimg}
                Multiimage.update(tempdict)
                tempdict={MULTIFILES[i]:Grayimg}
                Multigray.update(tempdict)
                tempdict={MULTIFILES[i]:Grayrgbimg}
                Multigrayrgb.update(tempdict)
                print(Filebands)
            except:
                try:
                    entbox.insert(END,MULTIFILES[i],"wrap")
                    entbox.config(stat=DISABLED)
                    Filersc=rasterio.open(MULTIFILES[i])
                    height=Filersc.height
                    width=Filersc.width
                    channel=Filersc.count
                    print(Filersc)
                    Filebands=np.zeros((channel,height,width))
                    Filesize=(height,width)
                    for j in range(channel):
                        band=Filersc.read(j+1)
                        band = np.where((band == 0)|(band==-10000) | (band==-9999), 1e-6, band)
                        Filebands[channel-(j+1),:,:]=band
                    Fileimg=img(Filesize,Filebands)
                    tempdict={MULTIFILES[i]:Fileimg}
                    Multiimage.update(tempdict)
                    print(Filebands)
                except:
                    messagebox.showerror('Invalid Filename','Cannot open '+MULTIFILES[i])
                    continue
        entbox.config(stat=DISABLED)

def Open_NIRfile(entbox):
    global Infrared,Inflabel
    QGISNIRFILE=filedialog.askopenfilename()
    if len(QGISNIRFILE)>0:
        entbox.config(stat=NORMAL)
        entbox.delete("1.0",END)
        entbox.tag_config("wrap",wrap=CHAR)
        if QGISNIRFILE.endswith('.tif') is True:
            #messagebox.showerror('Wrong file formate',message='Open tif formate file')
            #return
            entbox.insert(END,QGISNIRFILE,"wrap")
            entbox.config(stat=DISABLED)
            #messagebox.showinfo(title='Open NIR file',message='open NIR GeoTiff file:'+QGISNIRFILE)
            '''
            NIRrsc=gdal.Open(QGISNIRFILE)
            NIRsize=(NIRrsc.RasterYSize,NIRrsc.RasterXSize)
            bands=[]
            bandrank={}
            for j in range(3):
                band=NIRrsc.GetRasterBand(j+1)
                stats = band.GetStatistics( True, True )
                print("[ STATS ] =  Minimum=%.3f, Maximum=%.3f, Mean=%.3f, StdDev=%.3f" % (
                        stats[0], stats[1], stats[2], stats[3] ))
                tempdict={j:stats[1]}
                bandrank.update(tempdict)
                nodata=band.GetNoDataValue()
                if type(nodata)==type(None):
                    nodata=0
                if (nodata<stats[0] or nodata>stats[1]) and nodata!=-10000 and nodata!=-9999:
                    nodata=0
                band=band.ReadAsArray()
                band=np.where(band==nodata,1e-6,band)
                bands.append(band)
            bands=np.array(bands)
            NIRbands=np.zeros(bands.shape)
            i=0
            for e in sorted(bandrank,key=bandrank.get,reverse=True):
                NIRbands[i,:,:]=bands[e,:,:]
                i=i+1
            '''
            NIRrsc=rasterio.open(QGISNIRFILE)
            height=NIRrsc.height
            width=NIRrsc.width
            channel=NIRrsc.count
            print(NIRrsc)
            NIRbands=np.zeros((channel,height,width))
            NIRsize=(height,width)
            #for j in range(channel):
            for j in range(3,5):
                band = NIRrsc.read(j + 1)
                band = np.where((band == 0) | (band == -10000) | (band == -9999), 1e-6, band)
                NIRbands[channel - (j + 1), :, :] = band
            Infrared=img(NIRsize,NIRbands)
            Inflabel.text='Open file: '+QGISNIRFILE
            print(NIRbands)
        if QGISNIRFILE.endswith('.jpg') is True or QGISNIRFILE.endswith('.jpeg') is True or QGISNIRFILE.endswith('.png') is True:
            entbox.insert(END, QGISNIRFILE, "wrap")
            entbox.config(stat=DISABLED)
            NIRrsc=cv2.imread(QGISNIRFILE)
            height,width,channel=np.shape(NIRrsc)
            NIRsize=(height,width)
            NIRrsc=cv2.cvtColor(NIRrsc,cv2.COLOR_BGR2HSV)
            #NIRrsc=cv2.cvtColor(NIRrsc,cv2.COLOR_BGR2LAB)
            #NIRrsc=cv2.cvtColor(NIRrsc,cv2.COLOR_BGR2RGB)
            NIRbands=np.zeros((channel,height,width))
            for j in range(channel):
                band=NIRrsc[:,:,j]
                band=np.where(band==0,1e-6,band)
                NIRbands[j,:,:]=band
            Infrared=img(NIRsize,NIRbands)
            Inflabel.text = 'Open file: ' + QGISNIRFILE
            print(NIRbands)
    #Generate_NDVI()


def Open_defaultRGBfile(QGISRGBFILE):
    global RGB,RGBlabel,nodataposition,GRAY,currentfilename,defaultfile,RGBGRAY
    defaultfile=False
    currentfilename=QGISRGBFILE
    try:
        #entbox.insert(END, QGISRGBFILE, "wrap")
        #entbox.config(stat=DISABLED)
        RGBrsc=cv2.imread(QGISRGBFILE,flags=cv2.IMREAD_COLOR)
        #RGBrsc=cv2.imread(QGISRGBFILE,flags=cv2.IMREAD_COLOR)
        RGBrsccopy=cv2.imread(QGISRGBFILE,flags=cv2.IMREAD_COLOR)
        height,width,channel=np.shape(RGBrsc)
        RGBsize=(height,width)
        RGBrsc=cv2.cvtColor(RGBrsc,cv2.COLOR_BGR2RGB)   #white background bright kernels
        RGBrsc=cv2.GaussianBlur(RGBrsc,(5,5),cv2.BORDER_DEFAULT)
        #GRAYrsc=cv2.cvtColor(RGBrsccopy,cv2.COLOR_BGR2Lab)  #black background bright kernels
        GRAYrsc=cv2.cvtColor(RGBrsccopy,cv2.COLOR_BGR2Lab)   #white background black kernels
        GRAYrsc=cv2.cvtColor(GRAYrsc,cv2.COLOR_BGR2GRAY)
        GRAYrsc=cv2.GaussianBlur(GRAYrsc,(5,5),cv2.BORDER_DEFAULT)
        ostu=filters.threshold_otsu(GRAYrsc)
        GRAYrsc=GRAYrsc.astype('float32')
        GRAYrsc=GRAYrsc/ostu
        GRAYrgb=cv2.cvtColor(RGBrsccopy,cv2.COLOR_BGR2RGB)
        GRAYrgb=cv2.cvtColor(GRAYrgb,cv2.COLOR_BGR2GRAY)
        GRAYrgb=cv2.GaussianBlur(GRAYrgb,(5,5),cv2.BORDER_DEFAULT)
        ostu=filters.threshold_otsu(GRAYrgb)
        GRAYrgb=GRAYrgb.astype('float32')
        GRAYrgb=GRAYrgb/ostu

        RGBbands=np.zeros((channel,height,width))
        for j in range(channel):
            band=RGBrsc[:,:,j]
            band=np.where((band==0)|(band<0),1e-6,band)
            ostu=filters.threshold_otsu(band)
            band=band/ostu
            RGBbands[j,:,:]=band
        RGB=img(RGBsize,RGBbands)
        GRAY=img(RGBsize,GRAYrsc)
        RGBGRAY=img(RGBsize,GRAYrgb)
        #RGBlabel.text = 'Open file: ' + QGISRGBFILE
        print(RGBbands)
        defaultfile=True
    except:
        try:
            #entbox.insert(END,QGISRGBFILE,"wrap")
            #entbox.config(stat=DISABLED)
            RGBrsc=rasterio.open(QGISRGBFILE)
            #GBrsc.colorinterp=(ColorInterp.red,ColorInterp.green,ColorInterp.blue)
            height=RGBrsc.height
            width=RGBrsc.width
            channel=RGBrsc.count
            print(RGBrsc)
            #RGBrsc = cv2.cvtColor(RGBrsc, cv2.COLOR_BGR2RGB)
            RGBbands=np.zeros((channel,height,width))
            #GRAY=np.zeros((1,height,width))
            RGBsize=(height,width)
            #for j in range(channel):
            for j in range(0,channel):
                band = RGBrsc.read(j+1)
                band = np.where((band == 0)|(band==-10000) | (band==-9999), 1e-6, band)
                RGBbands[channel-(j+1), :, :] = band
            RGB=img(RGBsize,RGBbands)
            RGBlabel.text='Open file: '+QGISRGBFILE
            print(RGBbands)
            defaultfile=True
        except:
            messagebox.showerror('Wrong File',message='Cannot open '+QGISRGBFILE)

def Open_RGBfile(entbox,usertype):
    global RGB,RGBlabel,nodataposition,GRAY,currentfilename,RGBGRAY
    RGB=None
    RGBlabel=None
    GRAY=None
    QGISRGBFILE=filedialog.askopenfilename()
    currentfilename=QGISRGBFILE
    if len(QGISRGBFILE)>0:
        entbox.config(stat=NORMAL)
        entbox.delete("1.0",END)
        entbox.tag_config("wrap",wrap=CHAR)
        try:
            entbox.insert(END, QGISRGBFILE, "wrap")
            entbox.config(stat=DISABLED)
            RGBrsc=cv2.imread(QGISRGBFILE,flags=cv2.IMREAD_COLOR)
            #RGBrsc=cv2.imread(QGISRGBFILE,flags=cv2.IMREAD_COLOR)
            RGBrsccopy=cv2.imread(QGISRGBFILE,flags=cv2.IMREAD_COLOR)
            #cv2.imshow('singlefile',RGBrsccopy)
            #pyplt.imshow(RGBrsccopy)
            height,width,channel=np.shape(RGBrsc)
            RGBsize=(height,width)
            RGBrsc=cv2.cvtColor(RGBrsc,cv2.COLOR_BGR2RGB)   #white background bright kernels
            RGBrsc=cv2.GaussianBlur(RGBrsc,(5,5),cv2.BORDER_DEFAULT)
            GRAYrsc=cv2.cvtColor(RGBrsccopy,cv2.COLOR_BGR2Lab)  #black background bright kernels
            #GRAYrsc=cv2.cvtColor(RGBrsccopy,cv2.COLOR_BGR2RGB)   #white background black kernels
            GRAYrsc=cv2.cvtColor(GRAYrsc,cv2.COLOR_BGR2GRAY)
            GRAYrsc=cv2.GaussianBlur(GRAYrsc,(5,5),cv2.BORDER_DEFAULT)
            ostu=filters.threshold_otsu(GRAYrsc)
            GRAYrsc=GRAYrsc.astype('float32')
            GRAYrsc=GRAYrsc/ostu
            GRAYrgb=cv2.cvtColor(RGBrsccopy,cv2.COLOR_BGR2RGB)
            GRAYrgb=cv2.cvtColor(GRAYrgb,cv2.COLOR_BGR2GRAY)
            GRAYrgb=cv2.GaussianBlur(GRAYrgb,(5,5),cv2.BORDER_DEFAULT)
            ostu=filters.threshold_otsu(GRAYrgb)
            GRAYrgb=GRAYrgb.astype('float32')
            GRAYrgb=GRAYrgb/ostu

            RGBbands=np.zeros((channel,height,width))
            for j in range(channel):
                band=RGBrsc[:,:,j]
                band=np.where((band==0)|(band<0),1e-6,band)
                ostu=filters.threshold_otsu(band)
                band=band/ostu
                RGBbands[j,:,:]=band
            RGB=img(RGBsize,RGBbands)
            GRAY=img(RGBsize,GRAYrsc)
            RGBGRAY=img(RGBsize,GRAYrgb)
            #RGBlabel.text = 'Open file: ' + QGISRGBFILE
            print(RGBbands)
        except:
            try:
                entbox.insert(END,QGISRGBFILE,"wrap")
                entbox.config(stat=DISABLED)
                RGBrsc=rasterio.open(QGISRGBFILE)
                show(RGBrsc)
                #GBrsc.colorinterp=(ColorInterp.red,ColorInterp.green,ColorInterp.blue)
                height=RGBrsc.height
                width=RGBrsc.width
                channel=RGBrsc.count
                print(RGBrsc)
                #RGBrsc = cv2.cvtColor(RGBrsc, cv2.COLOR_BGR2RGB)
                RGBbands=np.zeros((channel,height,width))
                #GRAY=np.zeros((1,height,width))
                RGBsize=(height,width)
                #for j in range(channel):
                for j in range(0,channel):
                    band = RGBrsc.read(j+1)
                    band = np.where((band == 0)|(band==-10000) | (band==-9999), 1e-6, band)
                    RGBbands[channel-(j+1), :, :] = band
                RGB=img(RGBsize,RGBbands)
                RGBlabel.text='Open file: '+QGISRGBFILE
                print(RGBbands)
            except:
                messagebox.showerror('Wrong File',message='Cannot open '+QGISRGBFILE)

        #if QGISRGBFILE.endswith('.tif') is True:
            #messagebox.showerror('Wrong file formate',message='Open tif formate file')
            #return
        #messagebox.showinfo(title='Open RGB file',message='open RGB GeoTiff file:'+QGISRGBFILE)

            #RGBrsc=cv2.imread(QGISRGBFILE,flags=cv2.IMREAD_LOAD_GDAL)

            '''
            #height,width,channel=np.shape(RGBrsc)
            RGBrsc=gdal.Open(QGISRGBFILE)
            RGBsize=(RGBrsc.RasterYSize,RGBrsc.RasterXSize)
            bands=[]
            bandrank={}
            for j in range(3):
                band=RGBrsc.GetRasterBand(j+1)
                stats = band.GetStatistics( True, True )
                print("[ STATS ] =  Minimum=%.3f, Maximum=%.3f, Mean=%.3f, StdDev=%.3f" % (
                        stats[0], stats[1], stats[2], stats[3]))
                tempdict={j:stats[1]}
                bandrank.update(tempdict)
                nodata=band.GetNoDataValue()
                if type(nodata)==type(None):
                    nodata=0
                if (nodata<stats[0] or nodata>stats[1]) and nodata!=-10000 and nodata!=-9999:
                    nodata=0

                band=band.ReadAsArray()
                nodataposition=np.where(band==nodata)
                band=np.where(band==nodata,1e-6,band)
                bands.append(band)
            bands=np.array(bands)
            RGBbands=np.zeros(bands.shape)
            i=0
            for e in sorted(bandrank,key=bandrank.get,reverse=True):
                RGBbands[i,:,:]=bands[e,:,:]
                i=i+1
            '''

        '''
        if QGISRGBFILE.endswith('.jpg') is True or QGISRGBFILE.endswith('.jpeg') is True or QGISRGBFILE.endswith('.png') is True or QGISRGBFILE.endswith('.JPG') is True:
            entbox.insert(END, QGISRGBFILE, "wrap")
            entbox.config(stat=DISABLED)
            RGBrsc=cv2.imread(QGISRGBFILE,flags=cv2.IMREAD_COLOR)
            RGBrsccopy=cv2.imread(QGISRGBFILE,flags=cv2.IMREAD_COLOR)
            height,width,channel=np.shape(RGBrsc)
            RGBsize=(height,width)
            RGBrsc=cv2.cvtColor(RGBrsc,cv2.COLOR_BGR2RGB)   #white background bright kernels
            RGBrsc=cv2.GaussianBlur(RGBrsc,(5,5),cv2.BORDER_DEFAULT)
            #GRAYrsc=cv2.cvtColor(RGBrsccopy,cv2.COLOR_BGR2Lab)  #black background bright kernels
            GRAYrsc=cv2.cvtColor(RGBrsccopy,cv2.COLOR_BGR2RGB)   #white background black kernels
            GRAYrsc=cv2.cvtColor(GRAYrsc,cv2.COLOR_BGR2GRAY)
            GRAYrsc=cv2.GaussianBlur(GRAYrsc,(5,5),cv2.BORDER_DEFAULT)
            ostu=filters.threshold_otsu(GRAYrsc)
            GRAYrsc=GRAYrsc.astype('float32')
            GRAYrsc=GRAYrsc/ostu

            RGBbands=np.zeros((channel,height,width))
            for j in range(channel):
                band=RGBrsc[:,:,j]
                band=np.where((band==0)|(band<0),1e-6,band)
                ostu=filters.threshold_otsu(band)
                band=band/ostu
                RGBbands[j,:,:]=band
            RGB=img(RGBsize,RGBbands)
            GRAY=img(RGBsize,GRAYrsc)
            #RGBlabel.text = 'Open file: ' + QGISRGBFILE
            print(RGBbands)

        '''

    #Generate_NDVI()

def inserttop(step,usertype):
    global panelWelcome
    #topconteont=['1. Open File','2. Band Calculation', '3. Rotation', '4. Classification','5. Crop Extraction','6. Export']
    if usertype==1:
        topconteont=['1. Open File','2. Choose indices','3. Seed Counting','4. Export']
    if usertype==2:
        topconteont=['1. Open File','2. Choose indices','3. Crop Segmentation','4. Export']
    panelWelcome.config(stat=NORMAL)
    panelWelcome.delete("1.0",END)
    panelWelcome.tag_config("font",font='Helvetia 14')
    panelWelcome.tag_config("bfont",font='Helvetia 14 bold',foreground='red')
    for i in range(len(topconteont)):
        if i==step:
            panelWelcome.insert(END,' '+topconteont[i],"bfont")
        else:
            panelWelcome.insert(END,' '+topconteont[i],"font")
    panelWelcome.config(stat=DISABLED)
    '''
    CreateToolTip(panelWelcome,text="The software is sensitive with noise pixel, please process image that only contains your plant. You can use shapefile to cut the area out via QGIS.\n "
                         "The software processes TIF format image, please convert JPEG formate into TIF. You can use QGIS to do that.\n"
                         "steps:\n"
                         "1. Open RGB and NIR image, if you have HEIGHT image, the software also support that."
                         "Map file is a csv file, you can generate yourself.\n Map file should contain the labels (labels can be number or words) of your plants, and the labels should be arranged in the same way as your plants.\n "
                         "2. You can rotate your image, using 'Rotation' under 'tools'.\n Rotation image usually will lose information, you only need to rotate the image if it tilt over (+/-) 45 degrees.\n"
                         "3. Band calculation let you define your own equation (select 'custom'), the software will present results of your equation. NDVI is default.\n"
                         "4. Classify will do K-Means cluster for bands. At least one band have to be selected to run it. You can select multiple bands as indices to implement K-Means.\n The classes you select within the K-Means results will be the prototype image to process.\n"
                         "5. Extract crop will identify boundaries for each of your crops. Default iteration time is 30.\n"
                         "6. You can export the results using the 'Export' function. The software export three type of files:\n"
                         " A tif image that have the boundaries on your original RGB image.\n"
                         " A tif image have the labeles on each crops. A csv file containing\n"
                         " the NDVI data for each labelled crops.'")
    '''


def on_enter(conenttext,contentwindow):
    contentwindow.config(text=conenttext)

def on_leave(contentwindow):
    contentwindow.config(text="")

def insertbottom(step,usertype):
    global btm_frame
    for widget in btm_frame.winfo_children():
        widget.pack_forget()
    #commands=[partial(QGIS_NDVI,1),partial(bands_calculation,1),partial(rotationpanel,1),partial(KMeansPanel,1),partial(drawboundary,1),partial(export_result,1)]
    commands=[partial(QGIS_NDVI,usertype),partial(KMeansPanel,usertype),partial(drawboundary,usertype),partial(export_result,usertype)]
    tips=['Step 1, open files','Step 2, implement band calculations','Step 3, rotate your image to certain degree to process\n if it is not tilt too much, you can skip to next step,',
          'Step 4, Use K-Means cluster your band calculation results, you can select indecies for K-Means',
          'Step 5, Extract crops from your pre-processed image',
          'Step 6, Export results including a tif image with your original crops with boundaries,\n'
          'a tif that each crop is labeled with a specific number for other analysis,\n'
          'a csv file including #pixel, sum-NDVI, avg-NDVI, std-NDVI'
        ]
    if step!=0:
        lastaction=commands[step-1]
        lastactiontipcontent=tips[step-1]
    else:
        lastaction=None
        lastactiontipcontent='Not valid now'
    #if step!=5:
    if step!=3:
        futureaction=commands[step+1]
        futureactiontipcontent=tips[step+1]
    else:
        futureaction=None
        futureactiontipcontent='Not valid now'
    LastactionButton=Button(btm_frame,text='< Last Step',command=lastaction)
    #CreateToolTip(LastactionButton,text=lastactiontipcontent)
    FutureacitonButton=Button(btm_frame,text='Next Step >',command=futureaction)
    #CreateToolTip(FutureacitonButton,text=futureactiontipcontent)
    RestartButton=Button(btm_frame,text='Restart',command=partial(QGIS_NDVI,usertype))
    #CreateToolTip(RestartButton,text='Go back to Step 1. Cancel all steps you have processed')
    if type(lastaction)==type(None):
        LastactionButton.config(state=DISABLED)
    if type(futureaction)==type(None):
        FutureacitonButton.config(state=DISABLED)
    LastactionButton.pack(side=LEFT,padx=5,pady=5)
    RestartButton.pack(side=LEFT,padx=5,pady=5)
    FutureacitonButton.pack(side=LEFT,padx=5,pady=5)

def QGIS_NDIV_menu():
    global menubar
    menuchildren=menubar.winfo_children()
    usertype=0

    if len(menuchildren)==0:
        filemenu = Menu(menubar)
        toolmenu = Menu(menubar)
        help_ = Menu(menubar)

        filemenu.add_command(label="Open file", command=partial(QGIS_NDVI,usertype))
        #filemenu.add_command(label="Open Map", command=select_map)
        filemenu.add_command(label="Export", command=partial(export_result,usertype))
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=exit)

        toolmenu.add_command(label="1. Band Calculation", command=partial(bands_calculation,usertype))
        #toolmenu.add_command(label="Rotation", command=partial(rotationpanel,usertype))
        toolmenu.add_command(label="2. Choose class", command=partial(KMeansPanel,usertype))
        toolmenu.add_separator()
        toolmenu.add_command(label="3. Item Segmentation", command=partial(drawboundary,usertype))

        help_.add_command(label="Instructions", command=instructions)

        menubar.add_cascade(menu=filemenu, label="File")
        menubar.add_cascade(menu=toolmenu, label="Image processing")
        menubar.add_cascade(menu=help_, label="Help")
    QGIS_NDVI(usertype)


def QGIS_NDVI(usertype):
    global panelA,panelB,RGBlabel,Inflabel,Heightlabel,panelTree,treelist,rotation,root,panelWelcome,root,mapfile,seg_tif,valid_tif,Multilabel
    seg_tif=None
    valid_tif=None
    mapfile=''
    #panelWelcome.configure(text='')
    #root.geometry("")
    if usertype==1:
        emptymenu=Menu(root)
        root.config(menu=emptymenu)
        inserttop(0,usertype)
        insertbottom(0,usertype)
    if usertype==2:
        emptymenu=Menu(root)
        root.config(menu=emptymenu)
        inserttop(0,usertype)
        insertbottom(0,usertype)
    if panelA is not None:
        panelA.destroy()
    if panelB is not None:
        panelB.destroy()
    rotation=None
    panelTree.grid_forget()
    treelist.delete(0,END)
    panelA=Label(ctr_left)
    panelA.grid(row=1,column=0)
    bandarrays.clear()
    workbandarrays.clear()
    modified_tif=None

    RGBlabel=Text(panelA,height=2,width=100)
    if usertype==1:
        #RGBlabel.insert(END,'Single image processing, you can use the same setting for batch image processing (similar environment and same items).')
        RGBlabel.insert(END,'Use sample image -> Click Next Step')
    if usertype==2:
        RGBlabel.insert(END,'Use sample image -> Click Next Step')
    RGBlabel.config(stat=DISABLED)
    #Inflabel=Text(panelA,height=2,width=100)
    #Inflabel.insert(END,'Select images for batch image processing')
    #Inflabel.config(stat=DISABLED)
    Multilabel=Text(panelA,height=20,width=100)
    Multilabel.insert(END,'Select images for batch image processing')
    Multilabel.config(stat=DISABLED)
    #Heightlabel=Text(panelA,height=2,width=40)
    #Heightlabel.config(stat=DISABLED)
    #
    #Notelabel=Label(panelA,text='Note: If you do not have NIR image, open the RGB image as the NIR file.')

    QGISRGBbutton=Button(panelA,text='Open Single image (tif,jpeg,png)',command=partial(Open_RGBfile,RGBlabel,usertype))
    #QGISNIRbutton=Button(panelA,text='Open Multiple images (jpeg,png)',command=partial(Open_NIRfile,Inflabel))
    #QGISHeightbutton=Button(panelA,text='Open HEIGHT file (tif)',command=partial(Open_HEIGHTfile,Heightlabel))
    #
    MULTIbutton=Button(panelA,text='Open Multiple images (tif,jpeg,png)',command=partial(Open_Multifile,Multilabel,usertype))



    QGISRGBbutton.grid(row=1,column=0,sticky=N)

    #QGISNIRbutton.grid(row=2,column=0,sticky=N)
    #QGISHeightbutton.grid(row=3,column=0,sticky=N)
    #
    RGBlabel.grid(row=1,column=1)
    if usertype==1:
        MULTIbutton.grid(row=2,column=0,sticky=N)
        Multilabel.grid(row=2,column=1)
    #Inflabel.grid(row=2,column=1)
    #Heightlabel.grid(row=3,column=1)
    #
    #Notelabel.grid(row=5,column=0,sticky=N)
    if usertype==2:
        Maplabel=Text(panelA,height=2,width=100)
        Maplabel.config(stat=DISABLED)
        QGISMapbutton=Button(panelA,text='Open Map file (*.csv) (OPTIONAL)',command=partial(select_map,Maplabel))
        QGISMapbutton.grid(row=2,column=0,sticky=N)
        Maplabel.grid(row=2,column=1)

    '''
    else:
            image=gdal.Open(path)
            band=image.GetRasterBand(1)
            band=band.ReadAsArray()
            size=band.shape
            rpath=path.replace(" ","\ ")
            commandline="gdalwarp -of GTiff -ts 400 400 "+rpath+'rgboutput.tif'
            os.system('rm rgboutput.tif')
            os.system('gdalwarp -of GTiff -ts 400 400 '+rpath+' rgboutput.tif')
            modified_image=gdal.Open('rgboutput.tif')
            bands=[]
            for j in range(3):
                band=modified_image.GetRasterBand(j+1)
                band=band.ReadAsArray()
                if (j+1)!=4:
                    bands.append(band)
            #modified_image=cv2.resize(bands,(450,450),interpolation=cv2.INTER_AREA)
            bands=np.array(bands)
            RGB=img(size,image)
            #image=Image.fromarray(bands)
            image=rasterio.open('rgboutput.tif')
            show(image)
    '''


def select_image():
    global panelA,RGB,RGBlabel
    if panelA is not None:
        panelA.destroy()

    path=filedialog.askopenfilename()

    if len(path)>0:
        image=cv2.imread(path)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        size=Image.fromarray(image)
        modified_image=cv2.resize(image,(450,450),interpolation=cv2.INTER_LINEAR)
        greenlowerbound=np.array([30,20,20])
        greenupperbound=np.array([90,255,255])
        H=modified_image[:,:,0]
        S=modified_image[:,:,1]
        V=modified_image[:,:,2]
        H=np.where((H<90) & (H>30),H,0)
        S=np.where((S<255) & (S>20),S,0)
        V=np.where((V<255) & (V>20),V,0)
        modified_image[:,:,0]=H
        modified_image[:,:,1]=S
        modified_image[:,:,2]=V
        image=Image.fromarray(modified_image)
        RGB=img(size.size,modified_image)
        image=ImageTk.PhotoImage(image)
        if panelA is None: #or panelB is None:
            panelA=Label(ctr_left,image=image)
            panelA.image=image
            #panelA.pack(side="left",padx=10,pady=10)
            panelA.pack()
            #panelB=Label(image=edged)
            #RGBlabel.grid(row=3,column=0)
        else:
            panelA.destroy()
            panelA=Label(ctr_left,image=image)
            panelA.image=image
            panelA.pack()




def select_inf_image():
    global panelB,Infrared,Inflabel

    path=filedialog.askopenfilename()
    if panelB is not None:
        panelB.destroy()

    if len(path)>0:
        #if path.endswith('.jpg'):
        image=cv2.imread(path)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        size=Image.fromarray(image)
        modified_image=cv2.resize(image,(450,450),interpolation=cv2.INTER_LINEAR)
        greenlowerbound=np.array([30,20,20])
        greenupperbound=np.array([90,255,255])
        H=modified_image[:,:,0]
        S=modified_image[:,:,1]
        V=modified_image[:,:,2]
        H=np.where((H<90) & (H>30),H,0)
        S=np.where((S<255) & (S>20),S,0)
        V=np.where((V<255) & (V>20),V,0)
        modified_image[:,:,0]=H
        modified_image[:,:,1]=S
        modified_image[:,:,2]=V

        #image=cv2.cvtColor(modified_image,cv2.COLOR_BGR2RGB)
        image=Image.fromarray(modified_image)
        Infrared=img(size.size,modified_image)
        image=ImageTk.PhotoImage(image)
        if panelB is None:
            panelB=Label(ctr_right,image=image)
            panelB.image=image
            #panelB.pack(side="right",padx=10,pady=10)
            panelB.pack()
            #Inflabel=Label(text="size of Infrared="+str(size.size[0])+'x'+str(size.size[1]))
            #Inflabel.grid(row=3,column=1)
        else:
            panelB.configure(image=image)
            panelB.image=image
            #Inflabel=Label(text="size of Infrared="+str(size.size[0])+'x'+str(size.size[1]))
            #Inflabel.grid(row=3,column=1)


def select_map(entbox):
    global panelC,mapfile
    path=filedialog.askopenfilename()
    entbox.config(stat=NORMAL)
    entbox.delete("1.0",END)
    entbox.tag_config("wrap",wrap=CHAR)

    if len(path)>0:
        #panelC=Toplevel()
        #panelC.title("about map file...")
        #msg=Message(panelC,text="map file opened: "+path)
        #msg.pack(side="top",padx=5,pady=5)
        #button=Button(panelC,text="Dismiss",command=panelC.destroy)
        #button.pack()
        entbox.insert(END,path,"wrap")
        entbox.config(stat=DISABLED)
        mapfile=path
        #panelC.text="map file: "+path
        #label.pack(side="bottom")
        #label.place(side="bottom",height=10,width=300)
    #else:
        #panelWelcome=Label(text="Welcome to use PlantExtraction!\nPlease use the menubar on top to begin your image processing.")
        #panelWelcome.text="Welcome to use PlantExtraction!\nPlease use the menubar on top to begin your image processing."

def export_resultsel(usertype,select):
    iterkey=select.get()
    files=list(multi_results.keys())
    iterkeys=iterkey.split(',')
    file=iterkeys[0]
    iternum=iterkeys[1]
    path=filedialog.askdirectory()
    labeldict=multi_results[file]
    labels=labeldict[iternum]['labels']
    colortable=labeldict[iternum]['colortable']
    head_tail=os.path.split(file)
    originfile,extension=os.path.splitext(head_tail[1])
    if usertype>0:
        #inserttop(5)
        #insertbottom(5)
        inserttop(3,usertype)
        insertbottom(3,usertype)
    if type(labels)==type(None):
        messagebox.showerror('No processed image',message='Please process image first.')
        return
    if len(path)>0:
        messagebox.showinfo('Save process',message='Program is saving results to'+path)
        #border=border.astype('float32')
        #print(border)
        realborder=cv2.resize(src=border,dsize=(RGB.size[1],RGB.size[0]),interpolation=cv2.INTER_LINEAR)
        floatlabels = labels
        floatlabels = floatlabels.astype('float32')
        floatlabels = cv2.resize(src=floatlabels, dsize=(RGB.size[1], RGB.size[0]), interpolation=cv2.INTER_LINEAR)
        floatvalid=valid_tif
        floatvalid=floatvalid.astype('float32')
        floatvalid=cv2.resize(src=floatvalid,dsize=(RGB.size[1], RGB.size[0]), interpolation=cv2.INTER_LINEAR)
        indicekeys=list(bandarrays.keys())
        indeclist=[ 0 for i in range(len(indicekeys)*3)]
        #originrestoredband=np.multiply(labels,modified_tif)
        #originrestoredband = np.multiply(labels, valid_tif)
        originrestoredband=labels
        for uni in colortable:
            print(uni,colortable[uni])
            uniloc=np.where(labels==(uni))
            print(len(uniloc[0]))
        restoredband=originrestoredband.astype('float32')
        restoredband=cv2.resize(src=restoredband,dsize=(RGB.size[1],RGB.size[0]),interpolation=cv2.INTER_LINEAR)

        datatable={}
        origindata={}
        for key in indicekeys:
            data=bandarrays[key]
            #data=workbandarrays[key]
            data=data.tolist()
            tempdict={key:data}
            origindata.update(tempdict)
        currentsizes=kernelsizes[file][iternum]
        for uni in colortable:
            print(uni,colortable[uni])
            uniloc=np.where(restoredband==float(uni))
            #uniloc=np.where((restoredband<=uni)&(restoredband>(uni-1))&(restoredband!=0))
            if len(uniloc)==0 or len(uniloc[1])==0:
                continue
            #width=max(uniloc[0])-min(uniloc[0])
            #length=max(uniloc[1])-min(uniloc[1])

            #subarea=restoredband[min(uniloc[0]):max(uniloc[0])+1,min(uniloc[1]):max(uniloc[1])+1]
            #findcircle(subarea)
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
            for ki in range(len(indicekeys)):
                #originNDVI=bandarrays[indicekeys[ki]]
                #originNDVI=originNDVI.tolist()
                originNDVI=origindata[indicekeys[ki]]
                pixellist=[]
                for k in range(len(uniloc[0])):
                    #tempdict[colortable[uni]][5]+=databand[uniloc[0][k]][uniloc[1][k]]
                    #tempdict[colortable[uni]][0]+=infrbands[0][uniloc[0][k]][uniloc[1][k]]
                    #tempdict[colortable[uni]][1]+=infrbands[2][uniloc[0][k]][uniloc[1][k]]
                    #tempdict[colortable[uni]][2]+=rgbbands[1][uniloc[0][k]][uniloc[1][k]]
                    #tempdict[colortable[uni]][3]+=rgbbands[0][uniloc[0][k]][uniloc[1][k]]
                    #tempdict[colortable[uni]][4]+=rgbbands[2][uniloc[0][k]][uniloc[1][k]]
                    #tempdict[colortable[uni]][6]+=NDVI[uniloc[0][k]][uniloc[1][k]]
                    tempdict[colortable[uni]][5+ki*3]+=originNDVI[uniloc[0][k]][uniloc[1][k]]
                    tempdict[colortable[uni]][6+ki*3]+=originNDVI[uniloc[0][k]][uniloc[1][k]]
                    pixellist.append(originNDVI[uniloc[0][k]][uniloc[1][k]])
                #for i in range(7):
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

def plotgaussian(frame,iter,filename):
    counts=list(multi_results[filename][iter]['counts'])
    #hist,bin_edges=np.histogram(counts,density=True)
    #counts=[2,3,43,5,35,5,4,6,56]

    fig = Figure(figsize=(5, 4), dpi=100)
    #t = np.arange(minicount, maxcount, .01)
    #fig.add_subplot(111).plot(t, 2 * np.sin(2 * np.pi * t))
    pltfig=fig.add_subplot(111)
    pltfig.hist(counts,bins='auto')
    #pltfig.title('Histogram')

    canvas=FigureCanvasTkAgg(fig,master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=LEFT)

def export_result(usertype):
    #global labels,border,colortable
    if len(multi_results)==0:
        messagebox.showerror('Invalid Option','No Processed Image')
        return
    filenames=(multi_results.keys())
    c=Toplevel()
    resframe=LabelFrame(c)
    resframe.pack()
    for file in filenames:
        currlabeldict=multi_results[file]
        iterkey=list(currlabeldict.keys())
        localframe=LabelFrame(resframe)
        localframe.pack()
        v=StringVar()
        filelabel=Text(localframe,height=2,width=100)
        filelabel.insert(END,file)
        filelabel.pack()
        plotframe=LabelFrame(localframe)
        plotframe.pack()
        for i in range(len(iterkey)):
            subplotframe=LabelFrame(localframe)
            subplotframe.pack(side=LEFT)
            key=iterkey[i]
            plottitle=Text(subplotframe,height=1,width=60)
            plottitle.insert(END,key)
            plottitle.pack()
            counts=list(currlabeldict[key]['counts'])
            fig=Figure(figsize=(5,4),dpi=50)
            a=fig.add_subplot(111)
            a.hist(counts,bins='auto')
            canvas=FigureCanvasTkAgg(fig,master=subplotframe)
            canvas.draw()
            canvas.get_tk_widget().pack()
            b=Radiobutton(localframe,text=key,value=file+','+key,variable=v)
            b.pack(side=LEFT)
        exportbutton=Button(localframe,text='Export',command=partial(export_resultsel,usertype,v))
        exportbutton.pack()




def commentout(usertype):
    files=multi_results.keys()
    path=filedialog.askdirectory()
    for file in files:
        labels=multi_results[file][0]
        border=multi_results[file][1]
        colortable=multi_results[file][2]
        head_tail=os.path.split(file)
        originfile,extension=os.path.splitext(head_tail[1])
        if usertype>0:
            #inserttop(5)
            #insertbottom(5)
            inserttop(3,usertype)
            insertbottom(3,usertype)
        if type(labels)==type(None):
            messagebox.showerror('No processed image',message='Please process image first.')
            return
        if type(rotation)!=type(None):
            if rotation!=0:
                center=(225,225)
                a=rotation*-1.0
                print(a)
                M=cv2.getRotationMatrix2D(center,a,1.0)
                border=cv2.warpAffine(border.astype('float32'),M,dsize=(450,450),flags=cv2.INTER_LINEAR)
                labels=cv2.warpAffine(labels.astype('float32'),M,dsize=(450,450),flags=cv2.INTER_LINEAR)
        if len(path)>0:
            messagebox.showinfo('Save process',message='Program is saving results to'+path)
            border=border.astype('float32')
            print(border)
            realborder=cv2.resize(src=border,dsize=(RGB.size[1],RGB.size[0]),interpolation=cv2.INTER_LINEAR)
            floatlabels = labels
            floatlabels = floatlabels.astype('float32')
            floatlabels = cv2.resize(src=floatlabels, dsize=(RGB.size[1], RGB.size[0]), interpolation=cv2.INTER_LINEAR)
            floatvalid=valid_tif
            floatvalid=floatvalid.astype('float32')
            floatvalid=cv2.resize(src=floatvalid,dsize=(RGB.size[1], RGB.size[0]), interpolation=cv2.INTER_LINEAR)
            '''
            out_img=path+'/OutputRGBwithBorder.tif'
            '''
            outputimg=np.zeros((RGB.size[0],RGB.size[1],3))
            band1=RGB.bands[0]*floatvalid+realborder*255
            band2=RGB.bands[1]*floatvalid
            band3=RGB.bands[2]*floatvalid
            outputimg[:,:,0]=band1
            outputimg[:,:,1]=band2
            outputimg[:,:,2]=band3
            outputimg=outputimg.astype('uint8')
            outputimg=np.where(outputimg==0,255,outputimg)
            pyplt.imsave(path+'/'+originfile+'-OutputRGB-Border.png',outputimg)
            '''
            gtiffdriver=gdal.GetDriverByName('GTiff')
            out_ds=gtiffdriver.Create(out_img,RGB.size[1],RGB.size[0],3,3)
            #out_ds.SetGeoTransform(in_gt)
            #out_ds.SetProjection(dataproj)
            out_band=out_ds.GetRasterBand(1)
            out_band.WriteArray(band1)
            out_band=out_ds.GetRasterBand(2)
            out_band.WriteArray(band2)
            out_band=out_ds.GetRasterBand(3)
            out_band.WriteArray(band3)
            out_ds.FlushCache()
            out_img=path+'/Labeleddata.tif'
            '''

            lastone=np.zeros(floatlabels.shape,dtype='float32')
            unikeys=list(colortable.keys())
            #for uni in colortable:
            for i in range(len(unikeys)):
                lastone=np.where(floatlabels==float(unikeys[i]),i,lastone)
                #lastone = np.where((floatlabels <= unikeys[i]) & (floatlabels>((unikeys[i])-1)) & (floatlabels!=0), i, lastone)

            band1=lastone
            '''
            out_ds=gtiffdriver.Create(out_img,RGB.size[1],RGB.size[0],1,3)
            out_band=out_ds.GetRasterBand(1)
            out_band.WriteArray(band1)
            out_ds.FlushCache()
            '''
            band1=band1.astype('uint8')
            band1=np.where(band1==0,255,band1)
            pyplt.imsave(path+'/'+originfile+'-Labeleddata.png',band1)

            indicekeys=list(bandarrays.keys())
            indeclist=[ 0 for i in range(len(indicekeys)*3)]
            #originrestoredband=np.multiply(labels,modified_tif)
            #originrestoredband = np.multiply(labels, valid_tif)
            originrestoredband=labels
            for uni in colortable:
                print(uni,colortable[uni])
                uniloc=np.where(labels==(uni))
                print(len(uniloc[0]))
            restoredband=originrestoredband.astype('float32')
            restoredband=cv2.resize(src=restoredband,dsize=(RGB.size[1],RGB.size[0]),interpolation=cv2.INTER_LINEAR)

            datatable={}
            origindata={}
            for key in indicekeys:
                data=bandarrays[key]
                #data=workbandarrays[key]
                data=data.tolist()
                tempdict={key:data}
                origindata.update(tempdict)
            currentsizes=kernelsizes[file]
            for uni in colortable:
                print(uni,colortable[uni])
                uniloc=np.where(restoredband==float(uni))
                #uniloc=np.where((restoredband<=uni)&(restoredband>(uni-1))&(restoredband!=0))
                if len(uniloc)==0 or len(uniloc[1])==0:
                    continue
                #width=max(uniloc[0])-min(uniloc[0])
                #length=max(uniloc[1])-min(uniloc[1])

                #subarea=restoredband[min(uniloc[0]):max(uniloc[0])+1,min(uniloc[1]):max(uniloc[1])+1]
                #findcircle(subarea)
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
                for ki in range(len(indicekeys)):
                    #originNDVI=bandarrays[indicekeys[ki]]
                    #originNDVI=originNDVI.tolist()
                    originNDVI=origindata[indicekeys[ki]]
                    pixellist=[]
                    for k in range(len(uniloc[0])):
                        #tempdict[colortable[uni]][5]+=databand[uniloc[0][k]][uniloc[1][k]]
                        #tempdict[colortable[uni]][0]+=infrbands[0][uniloc[0][k]][uniloc[1][k]]
                        #tempdict[colortable[uni]][1]+=infrbands[2][uniloc[0][k]][uniloc[1][k]]
                        #tempdict[colortable[uni]][2]+=rgbbands[1][uniloc[0][k]][uniloc[1][k]]
                        #tempdict[colortable[uni]][3]+=rgbbands[0][uniloc[0][k]][uniloc[1][k]]
                        #tempdict[colortable[uni]][4]+=rgbbands[2][uniloc[0][k]][uniloc[1][k]]
                        #tempdict[colortable[uni]][6]+=NDVI[uniloc[0][k]][uniloc[1][k]]
                        tempdict[colortable[uni]][5+ki*3]+=originNDVI[uniloc[0][k]][uniloc[1][k]]
                        tempdict[colortable[uni]][6+ki*3]+=originNDVI[uniloc[0][k]][uniloc[1][k]]
                        pixellist.append(originNDVI[uniloc[0][k]][uniloc[1][k]])
                    #for i in range(7):
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



def Green_calculation():
    global panelA
    sumbands=np.sum(RGB.bands,axis=2)
    sumbands=np.where(sumbands==0,1,sumbands)
    RGBr=RGB.bands[:,:,0]/sumbands
    RGBg=RGB.bands[:,:,1]/sumbands
    RGBb=RGB.bands[:,:,2]/sumbands
    greenness=2*RGBg+RGBb-2*RGBr
    image=Image.fromarray(greenness)
    image=ImageTk.PhotoImage(image)
    panelA.configure(image=image)
    panelA.image=image

def NDVI_calculation():
    global panelA
    RGBr=RGB.bands[:,:,0]
    RGBg=RGB.bands[:,:,1]
    RGBb=RGB.bands[:,:,2]
    Inf0=Infrared.bands[:,:,0]
    Inf1=Infrared.bands[:,:,1]
    Inf2=Infrared.bands[:,:,2]
    leftband,rightband=None,None
    if comboleft.get()=='RGB band 1':
        leftband=RGBr
    if comboleft.get()=='RGB band 2':
        leftband=RGBg
    if comboleft.get()=='RGB band 3':
        leftband=RGBb
    if comboleft.get()=='Infrared band 1':
        leftband=Inf0
    if comboleft.get()=='Infrared band 2':
        leftband=Inf1
    if comboleft.get()=='Infrared band 3':
        leftband=Inf2
    #messagebox.showinfo('band choice 1',comboleft.get())

    if comboright.get()=='RGB band 1':
        rightband=RGBr
    if comboright.get()=='RGB band 2':
        rightband=RGBg
    if comboright.get()=='RGB band 3':
        rightband=RGBb
    if comboright.get()=='Infrared band 1':
        rightband=Inf0
    if comboright.get()=='Infrared band 2':
        rightband=Inf1
    if comboright.get()=='Infrared band 3':
        rightband=Inf2
    upper=leftband-rightband
    lower=leftband+rightband
    lower=np.where(lower==0,1,lower)
    image=upper/lower
    image=Image.fromarray(image)
    image=ImageTk.PhotoImage(image)
    panelA.configure(image=image)
    panelA.image=image
    if panelA is not None:
            panelA.destroy()
    panelA=Label(ctr_left,image=image)
    panelA.image=image
    panelA.grid(row=1,column=0)

def default_NDVI(widgets):
    '''
    global panelA,modified_tif,bandarrays,workbandarrays
    if RGB is not None and Infrared is not None and Infrared.size==RGB.size:

        #upper=-(Inf0-RGBg)
        #lower=Inf0+RGBg
        #lower=np.where(lower==0,1,lower)

        upper=Infrared.bands[0,:,:]-RGB.bands[0,:,:]
        lower=Infrared.bands[0,:,:]+RGB.bands[0,:,:]
        lower=np.where(lower==0,1,lower)
        NDVI=upper/lower
        tempdict={'LabOstu':NDVI}
        bandarrays.update(tempdict)


        if 'LabOstu' not in workbandarrays:
            worktempdict={'LabOstu':cv2.resize(NDVI,(450,450),interpolation=cv2.INTER_LINEAR)}
            workbandarrays.update(worktempdict)
        if 'Greenness' not in workbandarrays:
            worktempdict={'Greenness':cv2.resize(Greenness,(450,450),interpolation=cv2.INTER_LINEAR)}
            workbandarrays.update(worktempdict)

        out_fn='tempNDVI.tif'
        gtiffdriver=gdal.GetDriverByName('GTiff')
        out_ds=gtiffdriver.Create(out_fn,upper.shape[1],upper.shape[0],1,3)
        out_band=out_ds.GetRasterBand(1)
        out_band.WriteArray(NDVI)
        out_ds.FlushCache()

        modified_tif=cv2.resize(NDVI,(450,450),interpolation=cv2.INTER_NEAREST)
        pyplt.imsave(out_png,modified_tif)
        image=cv2.imread(out_png)
        image=Image.fromarray(image)
        image=ImageTk.PhotoImage(image)
        if panelA is not None:
            panelA.destroy()
        panelA=Label(ctr_left,image=image)
        panelA.image=image
        panelA.grid(row=0,column=0)
    '''
    global panelB
    calframe,RGBframe,Infframe,Heightframe,Opframe,Numframe,Entframe,entbox,button=widgets[:9]
    calframe.grid_forget()
    RGBframe.grid_forget()
    Infframe.grid_forget()
    Heightframe.grid_forget()
    Opframe.grid_forget()
    Numframe.grid_forget()
    Entframe.grid_forget()
    entbox.pack_forget()
    button.pack_forget()
    for widget in widgets[9:]:
        widget.pack_forget()

    pass



def calculatecustom(entbox):
    global bandarrays,panelA,treelist,modified_tif,workbandarrays
    equation=entbox.get("1.0",END)
    if 'Band' not in equation:
        if 'Height' not in equation:
            messagebox.showerror('No band selected',message='Select bands you want to calculate.')
        return
    equation=equation.split()
    checklist=['RGBBand1','RGBBand2','RGBBand3','InfraredBand1','InfraredBand2','InfraredBand3','HeightBand',
               '+','-','*','/','^','(',')']
    for ele in equation:
        if ele not in checklist:
            try:
                float(ele)
            except:
                messagebox.showerror('Invalid input',message='input: '+ele+' is invalid')
                return
    leftpar=equation.count('(')
    rightpar=equation.count(')')
    if leftpar!=rightpar:
        messagebox.showerror('Invalid input',message='input is invalid')
        return
    else:
        if '(' in equation:
            leftpar=equation.index('(')
            rightpar=equation.index(')')
            if rightpar<leftpar:
                messagebox.showerror('Invalid input',message='input is invalid')
                return


    tempband=calculator.init(RGB,Infrared,Height,equation)
    if type(tempband)==type(None):
        return
    bandname="".join(equation)
    if bandname not in bandarrays:
        tempdict={"".join(equation):tempband}
        bandarrays.update(tempdict)
        currentkey="".join(equation)
        if currentkey not in workbandarrays:
            img=cv2.resize(tempband,(450,450),interpolation=cv2.INTER_LINEAR)
            if type(rotation)!=type(None):
                if rotation!=0:
                    center=(225,225)
                    M=cv2.getRotationMatrix2D(center,rotation,1.0)
                    img=cv2.cv2.warpAffine(img,M,dsize=(450,450),flags=cv2.INTER_LINEAR)
            worktempdict={currentkey:img}
            workbandarrays.update(worktempdict)
        treelist.insert(END,"".join(equation))
        modified_tif=cv2.resize(tempband,(450,450),interpolation=cv2.INTER_NEAREST)
        pyplt.imsave(out_png,modified_tif)
        image=cv2.imread(out_png)
        image=Image.fromarray(image)
        image=ImageTk.PhotoImage(image)
        if panelA is not None:
            panelA.destroy()
        panelA=Label(ctr_left,image=image)
        panelA.image=image
        panelA.grid(row=0,column=0)
    entbox.config(stat=NORMAL)
    entbox.delete("1.0",END)
    entbox.config(stat=DISABLED)


def Entrydelete(entbox):
    entbox.config(stat=NORMAL)
    entbox.delete("1.0",END)
    entbox.config(stat=DISABLED)

def Entryinsert(content,entbox):
    if content=='Height' and Height is None:
        messagebox.showerror('No Height',message='No Height is loaded.')
        return
    entbox.config(stat=NORMAL)
    entbox.insert(END,content)
    print(content)
    entbox.config(stat=DISABLED)
    #need to check Height available




def custom_cal(widgets):
    global panelB,root
    #root.geometry("")
    calframe,RGBframe,Infframe,Heightframe,Opframe,Numframe,Entframe,entbox,button=widgets[:9]
    calframe.grid(row=1,column=0,padx=5,pady=5)
    RGBframe.grid(row=0,column=0)
    Infframe.grid(row=0,column=1,padx=5,pady=5)
    Heightframe.grid(row=0,column=2,padx=5,pady=5)
    Opframe.grid(row=1,column=0)
    Numframe.grid(row=2,column=0,padx=5,pady=5)
    Entframe.grid(row=1,column=1,columnspan=2,rowspan=2,padx=5,pady=5)
    entbox.pack(side='left')
    button.pack(side='left')
    for widget in widgets[9:]:
        widget.pack(side=LEFT)

    pass


def treelistop(e):
    global panelA,modified_tif
    if e.widget.get(0)=='':
        messagebox.showerror('No Data',message='2D array list is empty. Need to add or compute 2D array')
        return
    w=e.widget
    print('treelist select: '+w.selection_get())
    #tempband=bandarrays[w.selection_get()]
    tempband=workbandarrays[w.selection_get()]
    #out_fn='temppanelA.tif'
    #gtiffdriver=gdal.GetDriverByName('GTiff')
    #out_ds=gtiffdriver.Create(out_fn,tempband[1],tempband[0],1,3)
    #out_band=out_ds.GetRasterBand(1)
    #out_band.WriteArray(tempband)
    #out_ds.FlushCache()
    #modified_tif=cv2.resize(tempband,(450,450),interpolation=cv2.INTER_NEAREST)
    modified_tif=tempband
    pyplt.imsave(out_png,modified_tif)
    image=cv2.imread(out_png)
    image=Image.fromarray(image)
    image=ImageTk.PhotoImage(image)
    if panelA is not None:
        panelA.destroy()
    panelA=Label(ctr_left,image=image)
    panelA.image=image
    panelA.grid(row=0,column=0)


def singleimg_bandcal(rgbimg,grayimg,grayrgb):
    global currentfilename
    singlebandarrays={}
    singleworkbandarrays={}
    NDVI=grayimg.bands[:,:]
    tempdict={'LabOstu(darkbackground,grains)':NDVI}
    if 'LabOstu(darkbackground,grains)' not in singlebandarrays:
        singlebandarrays.update(tempdict)
        worktempdict={'LabOstu(darkbackground,grains)':cv2.resize(NDVI,(int(rgbimg.size[1]/ratio),int(rgbimg.size[0]/ratio)),interpolation=cv2.INTER_LINEAR)}
        singleworkbandarrays.update(worktempdict)
        #treelist.insert(END,'LabOstu(wheat,grains)')

    NDVI2=grayrgb.bands[:,:]
    tempdict={'LabOstu(whightbackground,black seeds)':NDVI2}
    if 'LabOstu(whightbackground,black seeds)' not in singlebandarrays:
        singlebandarrays.update(tempdict)
        worktempdict={'LabOstu(whightbackground,black seeds)':cv2.resize(NDVI2,(int(rgbimg.size[1]/ratio),int(rgbimg.size[0]/ratio)),interpolation=cv2.INTER_LINEAR)}
        singleworkbandarrays.update(worktempdict)
    NDI=128*((rgbimg.bands[1,:,:]-rgbimg.bands[0,:,:])/(rgbimg.bands[1,:,:]+rgbimg.bands[0,:,:])+1)
    tempdict={'NDI':NDI}
    #Greenness = rgbimg.bands[1, :, :] / (rgbimg.bands[0, :, :] + rgbimg.bands[1, :, :] + rgbimg.bands[2, :, :])
    #Greenness=np.where(CannyEdge!=0,0,Greenness)

    tempdict = {'NDI': NDI}
    if 'NDI' not in singlebandarrays:
        singlebandarrays.update(tempdict)
        worktempdict={'NDI':cv2.resize(NDI,(int(rgbimg.size[1]/ratio),int(rgbimg.size[0]/ratio)),interpolation=cv2.INTER_LINEAR)}
        singleworkbandarrays.update(worktempdict)
        #treelist.insert(END,'Greenness(seeds)')
    return singlebandarrays,singleworkbandarrays

def bands_calculation(usertype):
    global Inflabel,comboleft,comboright,panelB,panelA,treelist,bandarrays,modified_tif,panelTree,workbandarrays,ratio
    global pixelmmratio,RGB,GRAY,RGBGRAY,currentfilename
    #if usertype>0:
    #    inserttop(1)
    #    insertbottom(1)
    if panelB is not None:
        panelB.destroy()
    #panelB.destory()
    if RGB is None:
        if usertype==2:
            Open_defaultRGBfile('dup1OUTPUT.tif')
        if usertype==1:
            if type(Multiimage)==type(None):
                Open_defaultRGBfile('seedsample.JPG')
            else:
                files=list(Multiimage.keys())
                samplefile=files[0]
                currentfilename=samplefile
                RGB=Multiimage[samplefile]
                GRAY=Multigray[samplefile]
                RGBGRAY=Multigrayrgb[samplefile]
        #messagebox.showerror('No RGB',message="No RGB file\n Cannot extract R, G, B bands.")
    #if Infrared is None:
    #    messagebox.showerror('No Infrared',message='No infrared file.\n Cannot extract Infrared bands.')
    #if RGB is not None and Infrared is not None:
    #    if Infrared.size!=RGB.size:
    #        messagebox.showerror('Image size issue',message='Size of RGB and Infrared image is different.')
    #        return
    #    if Height is not None and Height.size!=RGB.size:
    #        messagebox.showerror('Image size issue',message='Size of Height differs from RGB and Infrared image.'
    #                                                        'Height size='+str(Height.size)+', RGB size='+str(RGB.size))
    #        return
    panelTree.grid(row=0,column=0)
    #upper=Infrared.bands[0,:,:]-RGB.bands[0,:,:]
    #lower=Infrared.bands[0,:,:]+RGB.bands[0,:,:]
    #lower=np.where(lower==0,1,lower)
    #NDVI=upper/lower

    #NDVI=np.where(CannyEdge!=0,0,NDVI)
    '''
    out_fn='tempNDVI.tif'
    gtiffdriver=gdal.GetDriverByName('GTiff')
    out_ds=gtiffdriver.Create(out_fn,upper.shape[1],upper.shape[0],1,3)
    out_band=out_ds.GetRasterBand(1)
    out_band.WriteArray(NDVI)
    out_ds.FlushCache()
    '''
    #messagebox.showinfo('Default color indexx',message=' color index were calculated.')
    if usertype==1:
        NDVI=GRAY.bands[:,:]
        tempdict={'LabOstu(darkbackground,grains)':NDVI}
        if RGB.size[0]>1000 or RGB.size[1]>1000:
            ratio=int(max(RGB.size[0]/1000,RGB.size[1]/1000))
        else:
            ratio = 1
        pixelmmratio=(210*297)/int(RGB.size[0]/ratio*RGB.size[1]/ratio)
        if 'LabOstu(darkbackground,grains)' not in bandarrays:
            bandarrays.update(tempdict)
            worktempdict={'LabOstu(darkbackground,grains)':cv2.resize(NDVI,(int(RGB.size[1]/ratio),int(RGB.size[0]/ratio)),interpolation=cv2.INTER_LINEAR)}
            #tempband=cv2.resize(NDVI,(int(RGB.size[1]/ratio),int(RGB.size[0]/ratio)),interpolation=cv2.INTER_LINEAR)
            #robert_edge=filters.roberts(tempband)
            #pyplt.imsave('edge.png',robert_edge)
            #edgeimg=Image.open('edge.png')
            #edgeimg.show()
            workbandarrays.update(worktempdict)
            treelist.insert(END,'LabOstu(darkbackground,grains)')
        NDVI2=RGBGRAY.bands[:,:]
        tempdict={'LabOstu(whightbackground,black seeds)':NDVI2}
        if 'LabOstu(whightbackground,black seeds)' not in bandarrays:
            bandarrays.update(tempdict)
            worktempdict={'LabOstu(whightbackground,black seeds)':cv2.resize(NDVI,(int(RGB.size[1]/ratio),int(RGB.size[0]/ratio)),interpolation=cv2.INTER_LINEAR)}
            #tempband=cv2.resize(NDVI,(int(RGB.size[1]/ratio),int(RGB.size[0]/ratio)),interpolation=cv2.INTER_LINEAR)
            #robert_edge=filters.roberts(tempband)
            #pyplt.imsave('edge.png',robert_edge)
            #edgeimg=Image.open('edge.png')
            #edgeimg.show()
            workbandarrays.update(worktempdict)
            treelist.insert(END,'LabOstu(whightbackground,black seeds)')
        NDI=128*((RGB.bands[1,:,:]-RGB.bands[0,:,:])/(RGB.bands[1,:,:]+RGB.bands[0,:,:])+1)
        tempdict={'NDI':NDI}
        if 'NDI' not in bandarrays:
            bandarrays.update(tempdict)
            worktempdict={'NDI':cv2.resize(NDI,(int(RGB.size[1]/ratio),int(RGB.size[0]/ratio)),interpolation=cv2.INTER_LINEAR)}
            workbandarrays.update(worktempdict)
            treelist.insert(END,'NDI')
        modified_tif=cv2.resize(NDVI,(int(RGB.size[1]/ratio),int(RGB.size[0]/ratio)),interpolation=cv2.INTER_NEAREST)

    if usertype==2:
        channels,height,width=RGB.bands.shape
        if channels<3:
            messagebox.showerror('No Enough Bands',message='Color channels should be at least 3')
        if RGB.size[0]>450 or RGB.size[1]>450:
            ratio=int(max(height/450,width/450))
        else:
            ratio = 1
        if channels>=3:
            Greenness = RGB.bands[1, :, :] / (RGB.bands[0, :, :] + RGB.bands[1, :, :] + RGB.bands[2, :, :])
            #Greenness=np.where(CannyEdge!=0,0,Greenness)
            tempdict = {'Greenness': Greenness}
            if 'Greenness' not in bandarrays:
                bandarrays.update(tempdict)
                image=cv2.resize(Greenness,(int(RGB.size[1]/ratio),int(RGB.size[0]/ratio)),interpolation=cv2.INTER_LINEAR)
                worktempdict={'Greenness':cv2.GaussianBlur(image,(3,3),cv2.BORDER_DEFAULT)}
                #tempband=cv2.resize(Greenness,(int(RGB.size[1]/ratio),int(RGB.size[0]/ratio)),interpolation=cv2.INTER_LINEAR)
                #robert_edge=filters.roberts(tempband)
                #robert_edge=tkintercore.get_boundary(robert_edge)
                #print(maxvalue)
                #robert_edge=np.where(robert_edge<maxvalue,0,1)
                #pyplt.imsave('greenedge.png',robert_edge)
                #edgeimg=Image.open('greenedge.png')
                #edgeimg.show()
                workbandarrays.update(worktempdict)
                treelist.insert(END,'Greenness')
            NDI=128*((RGB.bands[1,:,:]-RGB.bands[0,:,:])/(RGB.bands[1,:,:]+RGB.bands[0,:,:])+1)
            tempdict={'NDI':NDI}
            if 'NDI' not in bandarrays:
                bandarrays.update(tempdict)
                image=cv2.resize(NDI,(int(RGB.size[1]/ratio),int(RGB.size[0]/ratio)),interpolation=cv2.INTER_LINEAR)
                worktempdict={'NDI':cv2.GaussianBlur(image,(3,3),cv2.BORDER_DEFAULT)}
                workbandarrays.update(worktempdict)
                treelist.insert(END,'NDI')
            VEG=RGB.bands[1,:,:]/(np.power(RGB.bands[0,:,:],0.667)*np.power(RGB.bands[2,:,:],(1-0.667)))
            tempdict={'VEG':VEG}
            if 'VEG' not in bandarrays:
                bandarrays.update(tempdict)
                image=cv2.resize(VEG,(int(RGB.size[1]/ratio),int(RGB.size[0]/ratio)),interpolation=cv2.INTER_LINEAR)
                worktempdict={'VEG':cv2.GaussianBlur(image,(3,3),cv2.BORDER_DEFAULT)}
                workbandarrays.update(worktempdict)
                treelist.insert(END,'VEG')
            CIVE=0.441*RGB.bands[0,:,:]-0.811*RGB.bands[1,:,:]+0.385*RGB.bands[2,:,:]+18.78745
            tempdict={'CIVE':CIVE}
            if 'CIVE' not in bandarrays:
                bandarrays.update(tempdict)
                image=cv2.resize(CIVE,(int(RGB.size[1]/ratio),int(RGB.size[0]/ratio)),interpolation=cv2.INTER_LINEAR)
                worktempdict={'CIVE':cv2.GaussianBlur(image,(3,3),cv2.BORDER_DEFAULT)}
                workbandarrays.update(worktempdict)
                treelist.insert(END,'CIVE')
            MExG=1.262*RGB.bands[1,:,:]-0.884*RGB.bands[0,:,:]-0.311*RGB.bands[2,:,:]
            tempdict={'MExG':MExG}
            if 'MExG' not in bandarrays:
                bandarrays.update(tempdict)
                image=cv2.resize(MExG,(int(RGB.size[1]/ratio),int(RGB.size[0]/ratio)),interpolation=cv2.INTER_LINEAR)
                worktempdict={'MExG':cv2.GaussianBlur(image,(3,3),cv2.BORDER_DEFAULT)}
                workbandarrays.update(worktempdict)
                treelist.insert(END,'MExG')
        if channels>=5:
            NDVI=(RGB.bands[3,:,:]-RGB.bands[1,:,:])/(RGB.bands[3,:,:]+RGB.bands[1,:,:])
            tempdict={'NDVI':NDVI}
            if 'NDVI' not in bandarrays:
                bandarrays.update(tempdict)
                image=cv2.resize(NDVI,(int(RGB.size[1]/ratio),int(RGB.size[0]/ratio)),interpolation=cv2.INTER_LINEAR)
                worktempdict={'NDVI':cv2.GaussianBlur(image,(3,3),cv2.BORDER_DEFAULT)}
                workbandarrays.update(worktempdict)
                treelist.insert(END,'NDVI')
        if channels==6:
            Height=RGB.bands[5,:,:]
            tempdict={'HEIGHT':Height}
            if 'HEIGHT' not in bandarrays:
                bandarrays.update(tempdict)
                worktempdict={'HEIGHT':cv2.resize(Height,(int(RGB.size[1]/ratio),int(RGB.size[0]/ratio)),interpolation=cv2.INTER_LINEAR)}
                workbandarrays.update(worktempdict)
                treelist.insert(END,'HEIGHT')
        modified_tif=cv2.resize(Greenness,(int(RGB.size[1]/ratio),int(RGB.size[0]/ratio)),interpolation=cv2.INTER_NEAREST)

    '''




    ExR=1.3*RGB.bands[0,:,:]-RGB.bands[1,:,:]
    tempdict={'ExR':ExR}
    if 'ExR' not in bandarrays:
        bandarrays.update(tempdict)
        worktempdict={'ExR':cv2.resize(ExR,(int(RGB.size[1]/ratio),int(RGB.size[0]/ratio)),interpolation=cv2.INTER_LINEAR)}
        workbandarrays.update(worktempdict)
        treelist.insert(END,'ExR')
    CIVE=0.441*RGB.bands[0,:,:]-0.811*RGB.bands[1,:,:]+0.385*RGB.bands[2,:,:]+18.78745
    tempdict={'CIVE':CIVE}
    if 'CIVE' not in bandarrays:
        bandarrays.update(tempdict)
        worktempdict={'CIVE':cv2.resize(CIVE,(int(RGB.size[1]/ratio),int(RGB.size[0]/ratio)),interpolation=cv2.INTER_LINEAR)}
        workbandarrays.update(worktempdict)
        treelist.insert(END,'CIVE')
    Gstar=RGB.bands[1,:,:]/np.max(RGB.bands[1,:,:])
    Rstar=RGB.bands[0,:,:]/np.max(RGB.bands[0,:,:])
    Bstar=RGB.bands[2,:,:]/np.max(RGB.bands[2,:,:])
    r=Rstar/(Gstar+Rstar+Bstar)
    g=Gstar/(Gstar+Rstar+Bstar)
    b=Bstar/(Gstar+Rstar+Bstar)
    ExG=2*g-r-b
    tempdict={'ExG':ExG}
    if 'ExG' not in bandarrays:
        bandarrays.update(tempdict)
        worktempdict={'ExG':cv2.resize(ExG,(int(RGB.size[1]/ratio),int(RGB.size[0]/ratio)),interpolation=cv2.INTER_LINEAR)}
        workbandarrays.update(worktempdict)
        treelist.insert(END,'ExG')
    ExGR=ExG-ExR
    tempdict={'ExGR':ExGR}
    if 'ExGR' not in bandarrays:
        bandarrays.update(tempdict)
        worktempdict={'ExGR':cv2.resize(ExGR,(int(RGB.size[1]/ratio),int(RGB.size[0]/ratio)),interpolation=cv2.INTER_LINEAR)}
        workbandarrays.update(worktempdict)
        treelist.insert(END,'ExGR')
    NGRDI=(RGB.bands[1,:,:]-RGB.bands[0,:,:])/(RGB.bands[1,:,:]+RGB.bands[0,:,:])
    tempdict={'NGRDI':NGRDI}
    if 'NGRDI' not in bandarrays:
        bandarrays.update(tempdict)
        worktempdict={'NGRDI':cv2.resize(NGRDI,(int(RGB.size[1]/ratio),int(RGB.size[0]/ratio)),interpolation=cv2.INTER_LINEAR)}
        workbandarrays.update(worktempdict)
        treelist.insert(END,'NGRDI')
    VEG=RGB.bands[1,:,:]/(np.power(RGB.bands[0,:,:],0.667)*np.power(RGB.bands[2,:,:],(1-0.667)))
    tempdict={'VEG':VEG}
    if 'VEG' not in bandarrays:
        bandarrays.update(tempdict)
        worktempdict={'VEG':cv2.resize(VEG,(int(RGB.size[1]/ratio),int(RGB.size[0]/ratio)),interpolation=cv2.INTER_LINEAR)}
        workbandarrays.update(worktempdict)
        treelist.insert(END,'VEG')
    COM1=ExG+CIVE+ExGR+VEG
    tempdict={'COM1':COM1}
    if 'COM1' not in bandarrays:
        bandarrays.update(tempdict)
        worktempdict={'COM1':cv2.resize(COM1,(int(RGB.size[1]/ratio),int(RGB.size[0]/ratio)),interpolation=cv2.INTER_LINEAR)}
        workbandarrays.update(worktempdict)
        treelist.insert(END,'COM1')
    MExG=1.262*RGB.bands[1,:,:]-0.884*RGB.bands[0,:,:]-0.311*RGB.bands[2,:,:]
    tempdict={'MExG':MExG}
    if 'MExG' not in bandarrays:
        bandarrays.update(tempdict)
        worktempdict={'MExG':cv2.resize(MExG,(int(RGB.size[1]/ratio),int(RGB.size[0]/ratio)),interpolation=cv2.INTER_LINEAR)}
        workbandarrays.update(worktempdict)
        treelist.insert(END,'MExG')
    COM2=0.36*ExG+0.47*CIVE+0.17*VEG
    tempdict={'COM2':COM2}
    if 'COM2' not in bandarrays:
        bandarrays.update(tempdict)
        worktempdict={'COM2':cv2.resize(COM2,(int(RGB.size[1]/ratio),int(RGB.size[0]/ratio)),interpolation=cv2.INTER_LINEAR)}
        workbandarrays.update(worktempdict)
        treelist.insert(END,'COM2')
    '''



    pyplt.imsave(out_png,modified_tif)
    image=cv2.imread(out_png)
    image=Image.fromarray(image)
    image=ImageTk.PhotoImage(image)
    if panelA is not None:
        panelA.destroy()
    panelA=Label(ctr_left,image=image)
    panelA.image=image
    panelA.grid(row=0,column=0)

    panelB=Label(ctr_right)
    panelB.grid(row=0,column=0)


    calframe=LabelFrame(panelB,text='Band Calculator')
    #calframe.grid(row=1,column=0,padx=5,pady=5)
    RGBframe=LabelFrame(calframe,text='RGB bands')
    #RGBframe.grid(row=0,column=0)
    Infframe=LabelFrame(calframe,text='Infrared band')
    #Infframe.grid(row=0,column=1,padx=5,pady=5)
    Heightframe=LabelFrame(calframe,text='Height')
    #Heightframe.grid(row=0,column=2,padx=5,pady=5)
    Opframe=LabelFrame(calframe,text='operations')
    #Opframe.grid(row=1,column=0)
    Numframe=LabelFrame(calframe,text='Number')
    #Numframe.grid(row=2,column=0,padx=5,pady=5)
    Entframe=LabelFrame(calframe,text='Entry box')
    #Entframe.grid(row=1,column=1,columnspan=2,rowspan=2,padx=5,pady=5)
    entbox=Text(Entframe,height=3,width=30)
    entbox.config(state=DISABLED)
    #entbox.pack(side='left')
    button=Button(Entframe,text='Calculate',command=partial(calculatecustom,entbox))
    #button.pack(side='left')

    widgettuple=(calframe,RGBframe,Infframe,Heightframe,Opframe,Numframe,Entframe,entbox,button,)

    for i in range(3):
        button=Button(RGBframe,text='Band'+str(i+1),command=partial(Entryinsert,'RGBBand'+str(i+1),entbox))
        #button.pack(side='left')
        widgettuple=widgettuple+(button,)
    for i in range(3):
        button=Button(Infframe,text='Band'+str(i+1),command=partial(Entryinsert,'InfraredBand'+str(i+1),entbox))
        #button.pack(side='left')
        widgettuple=widgettuple+(button,)
    button=Button(Heightframe,text='Height',command=partial(Entryinsert,'HeightBand',entbox))
    #button.pack(side='left')
    widgettuple=widgettuple+(button,)
    button=Button(Opframe,text='+',command=partial(Entryinsert,' + ',entbox))
    #button.pack(side='left')
    widgettuple=widgettuple+(button,)
    button=Button(Opframe,text='-',command=partial(Entryinsert,' - ',entbox))
    #button.pack(side='left')
    widgettuple=widgettuple+(button,)
    button=Button(Opframe,text='*',command=partial(Entryinsert,' * ',entbox))
    #button.pack(side='left')
    widgettuple=widgettuple+(button,)
    button=Button(Opframe,text='/',command=partial(Entryinsert,' / ',entbox))
    #button.pack(side='left')
    widgettuple=widgettuple+(button,)
    button=Button(Opframe,text='(',command=partial(Entryinsert,' ( ',entbox))
    #button.pack(side='left')
    widgettuple=widgettuple+(button,)
    button=Button(Opframe,text=')',command=partial(Entryinsert,' ) ',entbox))
    #button.pack(side='left')
    widgettuple=widgettuple+(button,)
    button=Button(Opframe,text='clear',command=partial(Entrydelete,entbox))
    #button.pack(side='left')
    widgettuple=widgettuple+(button,)
    for i in range(10):
        button=Button(Numframe,text=str(i),command=partial(Entryinsert,str(i),entbox))
        #button.pack(side='left')
        widgettuple=widgettuple+(button,)
    button=Button(Numframe,text='.',command=partial(Entryinsert,'.',entbox))
    #button.pack(side='left')
    widgettuple=widgettuple+(button,)

    calframe=LabelFrame(panelB,text='Calculation Type')
    calframe.grid(row=0,column=0)
    choice=None
    calradio0=Radiobutton(calframe,text='Default Color index (Keep in current page)',value=0,command=partial(default_NDVI,widgettuple),variable=choice)
    calradio1=Radiobutton(calframe,text='Customize color index (Color index calculator)',value=1,command=partial(custom_cal,widgettuple),variable=choice)
    calradio0.pack()
    calradio1.pack()

def blur_panel():
    global panelA,panelB
    if panelB is not None:
        panelB.destroy()


def instructions():
    panelC=Toplevel()
    panelC.title("Instructions")
    msg=Message(panelC,text="The software is sensitive with noise pixel, please process image that only contains your plant. You can use shapefile to cut the area out via QGIS.\n "
                         "The software processes TIF format image, please convert JPEG formate into TIF. You can use QGIS to do that.\n"
                         "steps:\n"
                         "1. Open RGB and NIR image, if you have HEIGHT image, the software also support that.\n"
                         "2. Map file is a csv file, you can generate yourself.\n It should contain the labels (labels can be number or words) of your plants, and the labels should be arranged in the same way as your plants.\n "
                         "3. You can rotate your image, using 'Rotation' under 'tools'.\n Rotation image usually will lose information, you only need to rotate the image if it tilt over (+/-) 45 degrees.\n"
                         "4. Band calculation let you define your own equation (select 'custom'), the software will present results of your equation. NDVI is default.\n"
                         "5. Classify will do K-Means cluster for bands. At least one band have to be selected to run it. You can select multiple bands as indices to implement K-Means.\n The classes you select within the K-Means results will be the prototype image to process.\n"
                         "6. Extract crop will identify boundaries for each of your crops. Default iteration time is 30.\n"
                         "7. You can export the results using the 'Export' function. The software export three type of files:\n"
                         " A tif image that have the boundaries on your original RGB image.\n"
                         " A tif image have the labeles on each crops. A csv file containing\n"
                         " the NDVI data for each labelled crops.'")
    msg.pack(side="top",padx=5,pady=5)
    button=Button(panelC,text="Close",command=panelC.destroy)
    button.pack()

#def init():
    #global  root,panelTree,panelA,panelB,ctr_left,ctr_right,center,bindtree,scrollbar,treelist,
root=Tk()
root.title('GridFree')
root.geometry("")

root.option_add('*tearoff',False)

top_frame=Frame(root,width=1000,height=50)
center=Frame(root,width=1000,height=450)
btm_frame=Frame(root,width=1000,height=40)
root.grid_rowconfigure(1,weight=1)

root.grid_rowconfigure(1,weight=1)
root.grid_columnconfigure(0,weight=1)

top_frame.grid(row=0)
center.grid(row=1)
btm_frame.grid(row=2)

ctr_tree=Frame(center,width=250,height=450,padx=5,pady=5)
ctr_left=Frame(center,width=450,height=450)
ctr_right=Frame(center,width=450,height=450)

ctr_tree.pack(side=LEFT)
ctr_left.pack(side=LEFT)
ctr_right.pack(side=LEFT)

panelTree=LabelFrame(ctr_tree,text='Color Index',padx=5,pady=5)
panelTree.grid(row=0,column=0)
treelist=Listbox(panelTree,width=20,height=15)
treelist.pack(side='left')
bindtree=treelist.bind('<<ListboxSelect>>',treelistop)
scrollbar=Scrollbar(panelTree)
scrollbar.pack(side='right',fill=Y)
treelist.config(yscrollcommand=scrollbar.set)
scrollbar.config(command=treelist.yview)
panelTree.grid_forget()
#for i in range(20):
#    treelist.insert(END,i)




panelWelcome=Text(top_frame,height=3,width=100)
panelWelcome.tag_config("just",justify=CENTER)
panelWelcome.insert(END,"Welcome to use GridFree!\nPlease use the menubar on top to begin your image processing.\n"
                                  "GridFree is a pixel-level label plants in drone images\n","just")


#panelWelcome.grid(row=0,columnspan=3)
#panelWelcome.pack(side="top",padx=10,pady=10)
panelWelcome.configure(state=DISABLED)
panelWelcome.grid(row=0,column=0,padx=10,pady=10)

cropimage=cv2.imread('crop.png')
cropimage=cv2.cvtColor(cropimage,cv2.COLOR_BGR2RGB)
cropimage=cv2.resize(cropimage,(150,150))
cropimage=Image.fromarray(cropimage)
cropimage=ImageTk.PhotoImage(file='crop.png')

seedimage=cv2.imread('seed.png')
seedimage=cv2.cvtColor(seedimage,cv2.COLOR_BGR2RGB)
seedimage=cv2.resize(seedimage,(195,195))
seedimage=Image.fromarray(seedimage)
seedimage=ImageTk.PhotoImage(seedimage)

panelA=Label(ctr_left)
#startbutton=Button(panelA,text='Crop images',command=QGIS_NDIV_menu)
startbutton=Button(panelA,text='Crop images',image=cropimage,command=partial(QGIS_NDVI,2),compound=LEFT)
newuserbutton=Button(panelA,text='Kernel images',image=seedimage,command=partial(QGIS_NDVI,1),compound=LEFT)
versiondes=Text(panelA)
versiondes.insert(END,'                          Implerment Python version = 3.6.5, QGIS version = 3.4',"just")
versiondes.config(stat=DISABLED)
startbutton.pack()
newuserbutton.pack()
versiondes.pack(side=RIGHT)
panelA.pack()


menubar=Menu(root)
root.config(menu=menubar)

#CreateToolTip(startbutton,'Operate with menu bar')
#CreateToolTip(newuserbutton,'Operate step by step')

#def testbind(event):
#    print(event.x,event.y)
#startbutton.bind('<Enter>',testbind)
''' MENU BAR SETTING
filemenu=Menu(menubar)
toolmenu=Menu(menubar)
help_=Menu(menubar)
#panelWelcome.configure(text='')
#panelWelcome.grid(row=0,column=0,padx=10,pady=10)

#filemenu.add_command(label="Open RGB Image(mosaic)",command=select_image)
#filemenu.add_command(label="Open Infrared Image(mosaic)",command=select_inf_image)
filemenu.add_command(label="Open reflectance(wavelength) image",command=QGIS_NDVI)
filemenu.add_command(label="Open Map",command=select_map)
filemenu.add_command(label="Export",command=export_result)
filemenu.add_separator()
filemenu.add_command(label="Exit",command=exit)

toolmenu.add_command(label="Band Calculation",command=bands_calculation)
toolmenu.add_command(label="Rotation",command=rotationpanel)
toolmenu.add_command(label="Classify",command=KMeansPanel)
#toolmenu.add_command(label="Blur",command=blur_panel)
toolmenu.add_separator()
toolmenu.add_command(label="Extract Crops",command=drawboundary)

help_.add_command(label="Instructions",command=instructions)

menubar.add_cascade(menu=filemenu,label="File")
menubar.add_cascade(menu=toolmenu,label="Tools")
menubar.add_cascade(menu=help_,label="Help")

#btn1=Button(root,text="Select an image",command=select_image).grid()
#btn2=Button(root,text="Select a map",command=select_map).grid()
#btn1.pack(side="bottom",fill="both",expand="yes",padx="10",pady="10")
#btn2.pack(side="bottom",fill="both",expand="yes",padx="10",pady="10")
'''

root.mainloop()
