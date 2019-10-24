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