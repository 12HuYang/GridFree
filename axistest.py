#Based on:
#An Introduction to Tkinter
#Fredrik Lundh
#http://www.pythonware.com/library/tkinter/introduction/

#Copyright 1999 by Fredrik Lundh

# Modifications 2013 by Nina Amenta

# get all of the functions in the tkinter module
from tkinter import *

loccanvas=None
minx=0
maxx=0
totalbins=0
linelocs=[0,0]
bins=None
'''
def cal_xvalue(x):
    print(maxx,minx,max(bins),min(bins))
    binwidth=int(maxx-minx)/(max(bins)-min(bins))
    print(x,minx,binwidth)
    xloc=int((x-minx)/binwidth)
    print(xloc)
    #value=min(bins)+xloc*binwidth
    return xloc



def item_enter(event):
    global loccanvas
    loccanvas.config(cursor='hand2')
    loccanvas._restorItem=None
    loccanvas._restoreOpts=None
    itemType=loccanvas.type(CURRENT)
    #print(itemType)

    pass

def item_leave(event):
    global loccanvas
    pass

def item_start_drag(event):
    global loccanvas,linelocs
    itemType=loccanvas.type(CURRENT)
    print(itemType)
    if itemType=='line':
        fill=loccanvas.itemconfigure(CURRENT,'fill')[4]
        if fill=='red':
            loccanvas._lastX=event.x
            #loccanvas._lastY=event.y
            linelocs[0]=event.x
        else:
            if fill=='orange':
                loccanvas._lastX=event.x
                #loccanvas._lastY=event.y
                linelocs[1]=event.x
            else:
                loccanvas._lastX=None
    else:
        loccanvas._lastX=None
    pass

def item_drag(event):
    global loccanvas,linelocs
    x=event.x
    y=event.y
    if x<minx:
        x=minx
    if x>maxx:
        x=maxx
    try:
        fill=loccanvas.itemconfigure(CURRENT,'fill')[4]
    except:
        return
    #itemType=loccanvas.type(CURRENT)
    try:
        test=0-loccanvas._lastX
    except:
        return
    loccanvas.move(CURRENT,x-loccanvas._lastX,0)
    loccanvas._lastX=x
    if fill=='red':
        linelocs[0]=x
    if fill=='orange':
        linelocs[1]=x
            #print(line_a)
    #print(minline)
    #print(maxline)
    print(cal_xvalue(linelocs[0]),cal_xvalue(linelocs[1]))

    pass
'''

def drawdots(ulx,uly,rlx,rly,x_bins,y_bins,datalist,canvas):
    global loccanvas,minx,maxx,totalbins,bins,linelocs
    loccanvas=canvas

    minx=ulx
    maxx=rlx

    canvas.create_line(ulx,uly,rlx,uly,width=2)
    canvas.create_line(ulx,uly,ulx,rly,width=2)
    canvas.create_line(ulx,rly,rlx,rly,width=2)
    canvas.create_line(rlx,uly,rlx,rly,width=2)
    vlinelocs=[ulx,rlx]
    hlinelocs=[rly,uly]

    canvas.create_text(ulx-25-10,int(uly/2)+25,text='\n'.join('L+W'),font=('Times',12),anchor=E)
    canvas.create_text(int(rlx/2)+50,uly+30,text='Area',font=('Times',12),anchor=N)

    xbinwidth=(rlx-ulx-50)/(len(x_bins)-1)
    for i in range(len(x_bins)):
        x=ulx+(i*xbinwidth)
        canvas.create_line(x+25,uly+5,x+25,uly,width=2)
        canvas.create_text(x+25,uly+6,text='%d'%(x_bins[i]),font=('Times',12),anchor=N)

    ybinwidth=(uly-rly-50)/(len(y_bins)-1)
    for i in range(len(y_bins)):
        y=uly-(i*ybinwidth)
        canvas.create_line(ulx-5,y-25,ulx,y-25,width=2)
        canvas.create_text(ulx-6,y-25,text='%d'%(y_bins[i]),font=('Times',12),anchor=E)

    for (xs,ys) in datalist:
        canvas.create_oval(xs-1,ys-1,xs+1,ys+1,width=1,outline='black',fill='SkyBlue')

    canvas.create_line(ulx+12,rly,ulx+12,uly,arrow=LAST,fill='red',width=2,dash=(5,1))
    canvas.create_line(rlx-12,rly,rlx-12,uly,arrow=LAST,fill='red',width=2)
    canvas.create_line(ulx,rly+12,rlx,rly+12,arrow=FIRST,fill='blue',width=2)
    canvas.create_line(ulx,uly-12,rlx,uly-12,arrow=FIRST,fill='blue',width=2,dash=(5,1))

def drawPlot(ulx,uly,rlx,rly,hist,bin_edges,canvas):
    global loccanvas,minx,maxx,totalbins,bins,linelocs
    loccanvas=canvas

    # The window is an object of type tk
    #root = Tk()
    #root.title('Simple Plot')

    # A canvas object is something you can draw on
    # we put it into the root window
    #canvas = Canvas(root, width=400, height=300, bg = 'white')
    # figures out how the canvas sits in the window
    #canvas.pack()

    # draw x and y axes
    minx=ulx
    maxx=rlx
    linelocs=[minx,maxx]
    totalbins=len(bin_edges)
    bins=bin_edges
    canvas.create_line(ulx,uly,rlx,uly, width=2)
    canvas.create_line(ulx,uly,ulx,rly,  width=2)

    # markings on x axis
    binwidth=(rlx-ulx)/(len(hist))
    for i in range(len(bin_edges)):
        x = ulx + (i * binwidth)
        canvas.create_line(x,uly+5,x,uly, width=2)
        canvas.create_text(x,uly+5, text='%d'% (bin_edges[i]), font=('Times',12),anchor=N)

    # markings on y axis
    maxhist=max(hist)
    histwidth=(uly-rly)/maxhist
    histbreak=int(maxhist/10)
    for i in range(maxhist):
        y = uly - (i * histwidth)
        if i%histbreak==0:
            canvas.create_line(ulx-5,y,ulx,y, width=2)
            canvas.create_text(ulx-6,y, text='%d'% (i), font=('Times',12),anchor=E)
        if i==maxhist-1 and i%histbreak!=0:
            canvas.create_line(ulx-5,y,ulx,y, width=2)
            canvas.create_text(ulx-6,y, text='%d'% (i), font=('Times',12),anchor=E)

    canvas.create_line(ulx,rly,ulx,uly,arrow=LAST,fill='red',width=2)
    #minline=canvas.create_text(ulx,rly-5,text='%d'% 0,fill='red',font=('Times',12))
    canvas.create_line(rlx,rly,rlx,uly,arrow=LAST,fill='orange',width=2)
    #maxline=canvas.create_text(rlx,rly-5,text='%d'% max(bin_edges),fill='red',font=('Times',12))

    #canvas.bind('<Any-Enter>',item_enter)
    #canvas.bind('<Any-Leave>',item_leave)
    #canvas.bind('<Button-1>',item_start_drag)
    #canvas.bind('<B1-Motion>',item_drag)

    # rescale the input data so it matches the axes
##    scaled = []
##    for (x,y) in dataList:
##        scaled.append((100 + 3*x, 250 - (4*y)/5))

    # draw the wiggly line
    #canvas.create_line(dataList, fill='black')

    # and some dots at the corner points
    #for (xs,ys) in dataList:
    #    canvas.create_oval(xs-6,ys-6,xs+6,ys+6, width=1,
    #                       outline='black', fill='SkyBlue2')

    # display window and wait for it to close
    #root.mainloop()


def main():

    # detect if this is being run by itself or because it was imported
    if __name__ != "__main__":
        return
    # some meaningless x-y data points to plot
    # the input data is in the range x = 0 to 100, and y = 0 to 250.
    originalData = [(12, 56), (20, 94), (33, 98), (45, 120), (61, 180),
                (75, 160), (98, 223)]

    # rescale the data to lie in the graph range x = 100 to 400, y = 250 to 50
    # remember y is zero at the top of the window.
    scaledDataList = []
    for (x,y) in originalData:
        scaledDataList.append((100 + 3*x, 250 - (4*y)/5))

    drawPlot(scaledDataList)

if __name__ == '__main__' :
    main()