from tkinter import *
from tkinter import ttk
from PIL import Image,ImageDraw,ImageFont
from PIL import ImageTk,ImageGrab
import matplotlib.pyplot as pyplt
import cv2
import time
box=None
boxlist=[]

def zoom(event):
    global box
    x=event.x
    y=event.y
    print(x,y)
    if box is not None:
        canvas.delete(box)
        time.sleep(0.1)
        print('delete')
    crop=oriimg.crop((x-10,y-10,x+10,y+10))
    w,h=crop.size
    #print(w,h)
    crop=crop.resize([w*5,h*5],resample=Image.BILINEAR)
    w,h=crop.size
    #crop=PhotoImage(width=20,height=20)
    #crop.blank()
    crop=ImageTk.PhotoImage(crop)
    #boxlist.append(crop)
    #crop.put("{red green} {blue yellow}", (x,y))
    box=canvas.create_image(x+5,y-5,image=crop)
    canvas.update()
    #time.sleep(0.1)



root=Tk()
canvas=Canvas(root,width=800,height=800)
oriimg=Image.open('seedsample.JPG')
img=ImageTk.PhotoImage(oriimg)
canvas.create_image(0,0,image=img,anchor=NW)
canvas.pack()
canvas.bind('<Motion>',zoom)
root.mainloop()
