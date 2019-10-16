import numpy, os
import csv
import time
from skimage.feature import corner_fast,corner_peaks,corner_harris,corner_shi_tomasi
global lastlinecount,misslabel
from scipy.stats import shapiro
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
import tkintercore

colortable={}
colormatch={}
caliavgarea=0
calimax=0
calimin=0
calisigma=0
greatareas=[]

class node:
    def __init__(self,i,j):
        self.i=i
        self.j=j
        self.label=0
        self.check=False

def boundarywatershed(area,segbondtimes,boundarytype):   #area = 1's
    if caliavgarea is not None and numpy.count_nonzero(area)<caliavgarea/2:
        return area
    x=[0,-1,-1,-1,0,1,1,1]
    y=[1,1,0,-1,-1,-1,0,1]

    areaboundary=tkintercore.get_boundary(area)

    temparea=area-areaboundary
    arealabels=tkintercore.labelgapnp(temparea)
    unique, counts = numpy.unique(arealabels, return_counts=True)
    if segbondtimes>=20:
        return area
    if(len(unique)>2):
        res=arealabels+areaboundary
        leftboundaryspots=numpy.where(areaboundary==1)

        leftboundary_y=leftboundaryspots[0].tolist()
        leftboundary_x=leftboundaryspots[1].tolist()
        for uni in unique[1:]:
            labelboundaryloc=tkintercore.get_boundaryloc(arealabels,uni)
            for m in range(len(labelboundaryloc[0])):
                for k in range(len(y)):
                    i = labelboundaryloc[0][m] + y[k]
                    j = labelboundaryloc[1][m] + x[k]
                    if i >= 0 and i < res.shape[0] and j >= 0 and j < res.shape[1]:
                        if res[i, j] == 1:
                            res[i,j]=uni
                            for n in range(len(leftboundary_y)):
                                if leftboundary_y[n]==i and leftboundary_x[n]==j:
                                    leftboundary_y.pop(n)
                                    leftboundary_x.pop(n)
                                    break

        res=numpy.asarray(res)-1
        res=numpy.where(res<0,0,res)
        return res
    else:
        newarea=boundarywatershed(temparea,segbondtimes+1,boundarytype)*2
        res=newarea+areaboundary
        leftboundaryspots=numpy.where(res==1)

        leftboundary_y = leftboundaryspots[0].tolist()
        leftboundary_x = leftboundaryspots[1].tolist()
        unique=numpy.unique(newarea)
        for uni in unique[1:]:
            labelboundaryloc = tkintercore.get_boundaryloc(newarea, uni)
            for m in range(len(labelboundaryloc[0])):
                for k in range(len(y)):
                    i = labelboundaryloc[0][m] + y[k]
                    j = labelboundaryloc[1][m] + x[k]
                    if i >= 0 and i < res.shape[0] and j >= 0 and j < res.shape[1]:
                        if res[i, j] == 1:
                            res[i, j] = uni
                            for n in range(len(leftboundary_y)):
                                if leftboundary_y[n] == i and leftboundary_x[n] == j:
                                    leftboundary_y.pop(n)
                                    leftboundary_x.pop(n)
                                    break

        res=numpy.asarray(res)/2
        res=numpy.where(res<1,0,res)
        return res

def manualboundarywatershed(area):

    '''
    if numpy.count_nonzero(area)<avgarea/2:
        return area
    x=[0,-1,-1,-1,0,1,1,1]
    y=[1,1,0,-1,-1,-1,0,1]
    leftboundaryspots=numpy.where(area==1)
    pixelcount=1
    label=1

    for k in range(len(leftboundaryspots[0])):
        i=leftboundaryspots[0][k]
        j=leftboundaryspots[1][k]
        area[i][j]=label
        pixelcount+=1
        if pixelcount==int(avgarea):
            pixelcount=1
            label+=1
    unique,count=numpy.unique(area,return_counts=True)
    for i in range(1,len(count)):
        if count[i]<avgarea/2:
            area=numpy.where(area==unique[i],unique[i-1],area)
    '''
    maskpara=0.5
    possiblecount=int(numpy.count_nonzero(area)/caliavgarea)
    distance=ndi.distance_transform_edt(area)
    masklength=int((caliavgarea*maskpara)**0.5)-1
    local_maxi=peak_local_max(distance,indices=False,footprint=numpy.ones((masklength,masklength)),labels=area)
    markers=ndi.label(local_maxi)[0]
    unique=numpy.unique(markers)
    while(len(unique)-1>possiblecount):
        maskpara+=0.1
        masklength=int((caliavgarea*maskpara)**0.5)-1
        local_maxi=peak_local_max(distance,indices=False,footprint=numpy.ones((masklength,masklength)),labels=area)
        markers=ndi.label(local_maxi)[0]
        unique=numpy.unique(markers)
    while(len(unique)-1<possiblecount):
        maskpara-=0.1
        masklength=int((caliavgarea*maskpara)**0.5)-1
        try:
            local_maxi=peak_local_max(distance,indices=False,footprint=numpy.ones((masklength,masklength)),labels=area)
        except:
            maskpara+=0.1
            masklength=int((caliavgarea*maskpara)**0.5)-1
            local_maxi=peak_local_max(distance,indices=False,footprint=numpy.ones((masklength,masklength)),labels=area)
            markers=ndi.label(local_maxi)[0]
            break
        markers=ndi.label(local_maxi)[0]
        unique=numpy.unique(markers)
    localarea=watershed(-distance,markers,mask=area)

    return localarea

def manualdivide(area,greatareas):
    global exceptions
    unique, counts = numpy.unique(area, return_counts=True)
    hist=dict(zip(unique,counts))
    del hist[0]
    meanpixel=sum(counts[1:])/len(counts[1:])
    countseed=numpy.asarray(counts[1:])
    stdpixel=numpy.std(countseed)
    sortedkeys=list(sorted(hist,key=hist.get,reverse=True))
    while len(greatareas)>0:
        topkey=greatareas.pop(0)
        locs=numpy.where(area==topkey)
        ulx,uly=min(locs[1]),min(locs[0])
        rlx,rly=max(locs[1]),max(locs[0])
        subarea=area[uly:rly+1,ulx:rlx+1]
        subarea=subarea.astype(float)
        tempsubarea=subarea/topkey
        newtempsubarea=numpy.where(tempsubarea!=1.,0,1).astype(int)
        antitempsubarea=numpy.where((tempsubarea!=1.) & (tempsubarea!=0),subarea,0)
        times=len(locs[0])/meanpixel
        averagearea=len(locs[0])/times
        newsubarea=manualboundarywatershed(newtempsubarea)
        labelunique,labcounts=numpy.unique(newsubarea,return_counts=True)
        labelunique=labelunique.tolist()
        labcounts=labcounts.tolist()
        if len(labelunique)>2:
            newsubarea=newsubarea*topkey
            newlabel=labelunique.pop(-1)
            maxlabel=area.max()
            add=1
            while newlabel>1:
                newsubarea=numpy.where(newsubarea==topkey*newlabel,maxlabel+add,newsubarea)
                print('new label: '+str(maxlabel+add))
                newlabelcount=len(numpy.where(newsubarea==maxlabel+add)[0].tolist())
                print('add '+'label: '+str(maxlabel+add)+' count='+str(newlabelcount))
                newlabel=labelunique.pop(-1)
                add+=1

            newsubarea=newsubarea+antitempsubarea.astype(int)

            area[uly:rly+1,ulx:rlx+1]=newsubarea
            #labels=relabel(labels)
            unique, counts = numpy.unique(area, return_counts=True)
            hist=dict(zip(unique,counts))
            del hist[0]
            print('hist length='+str(len(counts)-1))
            print('max label='+str(area.max()))
            sortedkeys=list(sorted(hist,key=hist.get,reverse=True))
            meanpixel=sum(counts[1:])/len(counts[1:])
            countseed=numpy.asarray(counts[1:])
            stdpixel=numpy.std(countseed)

def combineloop(area,misslabel):
    global tinyareas
    localarea=numpy.asarray(area)
    unique, counts = numpy.unique(localarea, return_counts=True)
    hist=dict(zip(unique,counts))
    del hist[0]
    #print('hist length='+str(len(counts)-1))
    #print('max label='+str(labels.max()))
    meanpixel=sum(counts[1:])/len(counts[1:])
    countseed=numpy.asarray(counts[1:])
    stdpixel=numpy.std(countseed)
    leftsigma=(meanpixel-min(countseed))/stdpixel
    rightsigma=(max(countseed)-meanpixel)/stdpixel
    minisigma=min(leftsigma,rightsigma)
    #uprange=meanpixel+minisigma*stdpixel
    #lowrange=meanpixel-minisigma*stdpixel
    uprange=calimax
    lowrange=calimin
    sortedkeys=list(sorted(hist,key=hist.get))
    topkey=sortedkeys.pop(0)
    tinyareas=[]
    while misslabel<=0:# or gocombine==True:
    #while hist[topkey]<max(avgarea*0.75,lowrange):
        #topkey=sortedkeys.pop(0)
        print('uprange='+str(uprange))
        print('lowrange='+str(lowrange))
        print('combine part')
        i=topkey
        print(i,hist[i])
        if hist[i]<lowrange and i not in tinyareas:
        #if hist[i]<meanpixel:
            locs=numpy.where(localarea==i)
            ulx,uly=min(locs[1]),min(locs[0])
            rlx,rly=max(locs[1]),max(locs[0])
            width=rlx-ulx
            height=rly-uly
            #windowsize=min(width,height)
            #dividen=2
            subarea=localarea[uly:rly+1,ulx:rlx+1]
            tempsubarea=subarea/i
            #four direction searches
            stop=False
            poscombines=[]
            for j in range(1,11):
                up_unique=[]
                down_unique=[]
                left_unique=[]
                right_unique=[]
                maxlabel={}
                tempcombines=[]
                if uly-j>=0 and stop==False and len(up_unique)<2:
                    uparray=localarea[uly-j:uly,ulx:rlx+1]
                    up_unique=numpy.unique(uparray)
                    for x in range(len(up_unique)):
                        if up_unique[x]>0:
                            tempdict={up_unique[x]:hist[up_unique[x]]}
                            maxlabel.update(tempdict)
                if rly+j<localarea.shape[0] and stop==False and len(down_unique)<2:
                    downarray=localarea[rly+1:rly+j+1,ulx:rlx+1]
                    down_unique=numpy.unique(downarray)
                    for x in range(len(down_unique)):
                        if down_unique[x]>0:
                            tempdict={down_unique[x]:hist[down_unique[x]]}
                            maxlabel.update(tempdict)
                if ulx-j>=0 and stop==False and len(left_unique)<2:
                    leftarray=localarea[uly:rly+1,ulx-j:ulx]
                    left_unique=numpy.unique(leftarray)
                    for x in range(len(left_unique)):
                        if left_unique[x]>0:
                            tempdict={left_unique[x]:hist[left_unique[x]]}
                            maxlabel.update(tempdict)
                if ulx+j<localarea.shape[1] and stop==False and len(right_unique)<2:
                    rightarray=localarea[uly:rly+1,rlx+1:rlx+j+1]
                    right_unique=numpy.unique(rightarray)
                    for x in range(len(right_unique)):
                        if right_unique[x]>0:
                            tempdict={right_unique[x]:hist[right_unique[x]]}
                            maxlabel.update(tempdict)
                print(up_unique,down_unique,left_unique,right_unique)
                tempcombines.append(up_unique)
                tempcombines.append(down_unique)
                tempcombines.append(left_unique)
                tempcombines.append(right_unique)
                poscombines.append(tempcombines)
            tinylist=[]
            while(len(poscombines)>0 and stop==False):
                top=poscombines.pop(0)
                tinylist.append(top)
                toplist=[]
                for j in range(4):
                    toparray=top[j]
                    topunique=numpy.unique(toparray)
                    for ele in topunique:
                        toplist.append(ele)
                toplist=numpy.array(toplist)
                combunique,combcount=numpy.unique(toplist,return_counts=True)
                toplist=dict(zip(combunique,combcount))
                toplist=list(sorted(toplist,key=toplist.get,reverse=True))
                while(len(toplist)>0):
                    top=toplist.pop(0)
                    if top!=0:
                        topcount=hist[top]
                        if hist[i]+topcount>lowrange and hist[i]+topcount<uprange:
                            localarea=tkintercore.combinecrops(localarea,subarea,i,top,ulx,uly,rlx,rly)
                            stop=True
            if len(poscombines)==0 and stop==False:  #combine to the closest one
                tinyareas.append(topkey)
                #misslabel+=1

            unique, counts = numpy.unique(localarea, return_counts=True)
            hist=dict(zip(unique,counts))
            sortedkeys=list(sorted(hist,key=hist.get))
            meanpixel=sum(counts[1:])/len(counts[1:])
            countseed=numpy.asarray(counts[1:])
            stdpixel=numpy.std(countseed)
            leftsigma=(meanpixel-min(countseed))/stdpixel
            rightsigma=(max(countseed)-meanpixel)/stdpixel
            minisigma=min(leftsigma,rightsigma)
            #uprange=meanpixel+minisigma*stdpixel
            #lowrange=meanpixel-minisigma*stdpixel
            uprange=calimax
            lowrange=calimin
            #if stop==False and leftsigma>rightsigma:
            #    localarea=numpy.where(localarea==topkey,0,localarea)
            topkey=sortedkeys.pop(0)
            print('hist leng='+str(len(unique[1:])))
        else:
            if len(sortedkeys)>0:
                topkey=sortedkeys.pop(0)
            else:
                misslabel+=1


    return localarea



def divideloop(area):
    global greatareas
    unique, counts = numpy.unique(area, return_counts=True)
    hist=dict(zip(unique,counts))
    del hist[0]
    #print('hist length='+str(len(counts)-1))
    #print('max label='+str(labels.max()))
    meanpixel=sum(counts[1:])/len(counts[1:])
    countseed=numpy.asarray(counts[1:])
    stdpixel=numpy.std(countseed)
    leftsigma=(meanpixel-min(countseed))/stdpixel
    rightsigma=(max(countseed)-meanpixel)/stdpixel
    if leftsigma>rightsigma:
        minisigma=min(leftsigma,rightsigma)-0.5
    else:
        minisigma=min(leftsigma,rightsigma)
    #uprange=meanpixel+minisigma*stdpixel
    #lowrange=meanpixel-minisigma*stdpixel
    uprange=calimax
    lowrange=calimin
    sortedkeys=list(sorted(hist,key=hist.get,reverse=True))
    topkey=sortedkeys.pop(0)
    greatareas=[]
    while len(sortedkeys)>0:
        print('divide loop topkey='+str(topkey),hist[topkey])
        if topkey!=0 and hist[topkey]>uprange:
            locs=numpy.where(area==topkey)
            ulx,uly=min(locs[1]),min(locs[0])
            rlx,rly=max(locs[1]),max(locs[0])

            subarea=area[uly:rly+1,ulx:rlx+1]
            tempsubarea=subarea/topkey
            newtempsubarea=numpy.where(tempsubarea!=1.,0,1)
            antitempsubarea=numpy.where((tempsubarea!=1.) & (tempsubarea!=0),subarea,0)


            newsubarea=boundarywatershed(newtempsubarea,1,'inner')#,windowsize)

            labelunique,labcounts=numpy.unique(newsubarea,return_counts=True)
            labelunique=labelunique.tolist()
            if len(labelunique)>2:
                newsubarea=newsubarea*topkey
                newlabel=labelunique.pop(-1)
                maxlabel=area.max()
                add=1
                while newlabel>1:
                    newsubarea=numpy.where(newsubarea==topkey*newlabel,maxlabel+add,newsubarea)
                    print('new label: '+str(maxlabel+add))
                    newlabelcount=len(numpy.where(newsubarea==maxlabel+add)[0].tolist())
                    print('add '+'label: '+str(maxlabel+add)+' count='+str(newlabelcount))
                    newlabel=labelunique.pop(-1)
                    add+=1

                newsubarea=newsubarea+antitempsubarea.astype(int)
                area[uly:rly+1,ulx:rlx+1]=newsubarea
                unique, counts = numpy.unique(area, return_counts=True)
                hist=dict(zip(unique,counts))
                del hist[0]
                print('hist length='+str(len(counts)-1))
                print('max label='+str(area.max()))
                sortedkeys=list(sorted(hist,key=hist.get,reverse=True))
                meanpixel=sum(counts[1:])/len(counts[1:])
                countseed=numpy.asarray(counts[1:])
                stdpixel=numpy.std(countseed)
                leftsigma=(meanpixel-min(countseed))/stdpixel
                rightsigma=(max(countseed)-meanpixel)/stdpixel
                minisigma=min(leftsigma,rightsigma)
                #uprange=meanpixel+minisigma*stdpixel
                #lowrange=meanpixel-minisigma*stdpixel
                topkey=sortedkeys.pop(0)
            else:
                if hist[topkey]>uprange:
                    if topkey not in greatareas:
                        greatareas.append(topkey)
                    topkey=sortedkeys.pop(0)
                else:
                    break

        else:
            topkey=sortedkeys.pop(0)
    return area

def findcoin(area):
    unique, counts = numpy.unique(area, return_counts=True)
    maxpixel=max(counts[1:])
    maxpixelind=list(counts[1:]).index(maxpixel)
    maxpixellabel=unique[1:][maxpixelind]
    coinlocs=numpy.where(area==maxpixellabel)
    coinulx=min(coinlocs[1])
    coinuly=min(coinlocs[0])
    coinrlx=max(coinlocs[1])
    coinrly=max(coinlocs[0])
    coinparts={}
    coinparts.update({maxpixellabel:coinlocs})
    for uni in unique:
        if uni!=maxpixellabel:
            templocs=numpy.where(area==uni)
            tempulx=min(templocs[1])
            tempuly=min(templocs[0])
            temprlx=max(templocs[1])
            temprly=max(templocs[0])
            #inside coin boundingbox
            if tempulx>=coinulx and tempulx<=coinrlx and temprlx>=coinulx and temprlx<=coinrlx:
                if tempuly>=coinuly and tempuly<=coinrly and temprly>=coinuly and temprly<=coinrly:
                    if uni not in coinparts:
                        coinparts.update({uni:templocs})
                        continue
            if (tempulx>coinulx and tempulx<coinrlx) or (temprlx>coinulx and temprlx<coinrlx):
                if (tempuly>coinuly and tempuly<coinrly) or (temprly>coinuly and temprly<coinrly):
                    if uni not in coinparts:
                        coinparts.update({uni:templocs})
                        continue
    return coinparts


def processinput(input,ittimes=30,coin=True):
    band=input
    boundaryarea=boundarywatershed(band,1,'inner')
    boundaryarea=boundaryarea.astype(int)
    originmethod,misslabel,colortable=tkintercore.relabel(boundaryarea)

    labels=numpy.where(boundaryarea<1,0,boundaryarea)
    if coin:
        coinparts=findcoin(labels)
        coinkeys=coinparts.keys()
        for part in coinkeys:
            labels=numpy.where(labels==part,0,labels)
    else:
        coinparts={}

    #labels=boundaryarea
    unique, counts = numpy.unique(labels, return_counts=True)
    hist=dict(zip(unique,counts))
    divide=0
    docombine=0

    with open('countlist.csv','w') as f:
        writer=csv.writer(f)
        templist=counts[1:].tolist()
        for item in templist:
            tempitem=str(item)
            writer.writerow([tempitem])
    f.close()
    #print(numpy.column_stack(counts[1:]))
    meanpixel=sum(counts[1:])/len(counts[1:])
    countseed=numpy.asarray(counts[1:])
    stat,p=shapiro(countseed)
    alpha=0.05
    if p>alpha:
        print('like gaussian')
    else:
        print('does not like gaussian')

    stdpixel=numpy.std(countseed)
    leftsigma=(meanpixel-min(countseed))/stdpixel
    rightsigma=(max(countseed)-meanpixel)/stdpixel
    sortedkeys=list(sorted(hist,key=hist.get,reverse=True))

    allinexceptsions,godivide,gocombine=tkintercore.checkvalid(p,leftsigma,rightsigma)

    #while allinexceptsions is False:
    lastgreatarea=[]
    lasttinyarea=[]
    for it in range(ittimes):
        if godivide==False and gocombine==False:
            break
    #while godivide==True or gocombine==True:
        try:
            del hist[0]
        except KeyError:
            #continue
            pass
        print('hist length='+str(len(counts)-1))
        print('max label='+str(labels.max()))
        meanpixel=sum(counts[1:])/len(counts[1:])
        countseed=numpy.asarray(counts[1:])
        with open('countseed'+str(it)+'.csv','w') as f:
            csvwriter=csv.writer(f)
            content=['index','pixels']
            csvwriter.writerow(content)
            for i in range(len(counts[1:])):
                content=[str(i+1),str(counts[1:][i])]
                csvwriter.writerow(content)
            f.close()


        stdpixel=numpy.std(countseed)
        leftsigma=(meanpixel-min(countseed))/stdpixel
        rightsigma=(max(countseed)-meanpixel)/stdpixel
        minisigma=min(leftsigma,rightsigma)
        #uprange=meanpixel+minisigma*stdpixel
        #lowrange=meanpixel-minisigma*stdpixel
        uprange=calimax
        lowrange=calimin
        sortedkeys=list(sorted(hist,key=hist.get,reverse=True))
        #j=0

        if godivide is True:
            labels=divideloop(labels)
            #unique=numpy.unique(labels).tolist()
            #for i in range(len(unique)):
            #    labels=numpy.where(labels==unique[i],i,labels)
            unique, counts = numpy.unique(labels, return_counts=True)
            meanpixel=sum(counts[1:])/len(counts[1:])
            countseed=numpy.asarray(counts[1:])
            stdpixel=numpy.std(countseed)
            leftsigma=(meanpixel-min(countseed))/stdpixel
            rightsigma=(max(countseed)-meanpixel)/stdpixel
            minisigma=min(leftsigma,rightsigma)
            #uprange=meanpixel+minisigma*stdpixel
            #lowrange=meanpixel-minisigma*stdpixel
            uprange=calimax
            lowrange=calimin
            divide+=1
            outputlabel,misslabel,colortable=tkintercore.relabel(labels)

            if lastgreatarea==greatareas and len(lastgreatarea)!=0:
                manualdivide(labels,greatareas)
                #cornerdivide(labels,greatareas)
            lastgreatarea[:]=greatareas[:]
        stat,p=shapiro(countseed)
        #allinexceptsions,godivide,gocombine=checkvalid(misslabel,hist,sortedkeys,uprange,lowrange,avgarea)
        allinexceptsions,godivide,gocombine=tkintercore.checkvalid(p,leftsigma,rightsigma)
        if gocombine is True:
            labels=combineloop(labels,0)
            #unique=numpy.unique(labels).tolist()
            #for i in range(len(unique)):
            #    labels=numpy.where(labels==unique[i],i,labels)
            unique, counts = numpy.unique(labels, return_counts=True)
            meanpixel=sum(counts[1:])/len(counts[1:])
            countseed=numpy.asarray(counts[1:])
            stdpixel=numpy.std(countseed)
            leftsigma=(meanpixel-min(countseed))/stdpixel
            rightsigma=(max(countseed)-meanpixel)/stdpixel
            minisigma=min(leftsigma,rightsigma)
            #uprange=meanpixel+minisigma*stdpixel
            #lowrange=meanpixel-minisigma*stdpixel
            uprange=calimax
            lowrange=calimin
            docombine+=1
            outputlabel,misslabel,colortable=tkintercore.relabel(labels)

        unique, counts = numpy.unique(labels, return_counts=True)
        meanpixel=sum(counts[1:])/len(counts[1:])
        countseed=numpy.asarray(counts[1:])
        stdpixel=numpy.std(countseed)
        hist=dict(zip(unique,counts))
        for ele in sorted(hist,key=hist.get):
            if hist[ele]<lowrange:
                print('tinyarea:',ele,hist[ele])
            if hist[ele]>uprange:
                print('greatarea:',ele,hist[ele])


        leftsigma=(meanpixel-min(countseed))/stdpixel
        rightsigma=(max(countseed)-meanpixel)/stdpixel
        stat,p=shapiro(countseed)
        #allinexceptsions,godivide,gocombine=checkvalid(misslabel,hist,sortedkeys,uprange,lowrange,avgarea)
        allinexceptsions,godivide,gocombine=tkintercore.checkvalid(p,leftsigma,rightsigma)


    print('DONE!!!  counts='+str(len(counts)))

    labels=tkintercore.renamelabels(labels)

    colorlabels,misslabel,colortable=tkintercore.relabel(labels)

    NDVIbounary=tkintercore.get_boundary(labels)
    NDVIbounary=NDVIbounary*255
    res=NDVIbounary
    return labels,res,colortable,coinparts


def init(input,caliberation,ittimes,coin):
    global caliavgarea,calimax,calimin,calisigma
    caliavgarea=caliberation['mean']
    calimax=caliberation['max']
    calimin=caliberation['min']
    calisigma=caliberation['sigma']
    input=input.astype(int)

    pixellocs=numpy.where(input!=0)
    ulx,uly=min(pixellocs[1]),min(pixellocs[0])
    rlx,rly=max(pixellocs[1]),max(pixellocs[0])
    squarearea=(rlx-ulx)*(rly-uly)
    occupiedratio=len(pixellocs[0])/squarearea
    print(caliavgarea,occupiedratio)
    if occupiedratio>0.1:
        while(occupiedratio>0.1):
            distance=ndi.distance_transform_edt(input)
            input=numpy.where(distance==1.0,0,input)
            pixellocs=numpy.where(input!=0)
            ulx,uly=min(pixellocs[1]),min(pixellocs[0])
            rlx,rly=max(pixellocs[1]),max(pixellocs[0])
            squarearea=(rlx-ulx)*(rly-uly)
            occupiedratio=len(pixellocs[0])/squarearea
            print(caliavgarea,occupiedratio)


    #lastlinecount=lastline

    #if occupiedratio>=0.5:
    labels,res,colortable,coinparts=processinput(input,ittimes,coin)
    #else:
    #    labels,res,colortable,greatareas,tinyareas=kmeansprocess(pixellocs,input,counts)

    return labels,res,colortable,coinparts