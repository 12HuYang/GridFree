## -*- mode:python encoding:UTF-8 -*-
##
##  Copyright © 2008 Sarah Mount, James Shuttleworth, and Russel Winder
##
##  This file is part of the source code from the book "Python for Rookies" by Sarah Mount, James
##  Shuttleworth, and Russel Winder, published by Thomson Learning.
##
##  This software is free software: you can redistribute it and/or modify it under the terms of the
##  GNU General Public License as published by the Free Software Foundation, either version 3 of the
##  License, or (at your option) any later version.
##
##  This software is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
##  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See
##  the GNU General Public License for more details.
##
##  You should have received a copy of the GNU General Public License along with this software.  If
##  not, see <http://www.gnu.org/licenses/>.


#def getData ( fileName ) :
#    return [ float ( i ) for i in file ( fileName ).read ( ).split ( ) ]
import numpy as np
import histograms

def createBins ( hist ,bin_edges, numberOfBins) :
    #minimum = int ( min ( data ) - 1 )
    #maximum = int ( max ( data ) + 1 )
    #binWidth = ( maximum - minimum ) / numberOfBins
    #bins = [ [] for i in range ( numberOfBins ) ]
    #low = minimum
    #high = minimum + binWidth
    #for i in range ( numberOfBins ) :
    #    bins[i] = [ str ( low ) + '--' + str ( high ) , 0 ]
    #    for d in data :
    #        if low < d < high : bins[i][1] += 1
    #    low , high = high , int ( high + binWidth )
    #return bins
    bins = [ [] for i in range ( numberOfBins-1 ) ]
    for i in range(numberOfBins-1):
        bins[i]=[str(int(bin_edges[i]))+'--'+str(int(bin_edges[i+1])),hist[i]]
    return bins

if __name__ == '__main__' :
    #for bar in createBins ( getData ( 'dataForPlotting.txt' ) , 10 ) :
     #   print(bar[0] , bar[1])

    test=[3.4,5.6,98.3,1.01,4.56,3.48,3.5,5.6,99.3,1.03,4.52,3.42,3.45,5.7,98.8]
    print(len(test))
    hist,bin_edges=np.histogram(test,density=False)
    print(hist,bin_edges,len(bin_edges))
    res=createBins(hist.tolist(),bin_edges.tolist(),len(bin_edges))
    print(res)
    histograms.plot(res,hist.tolist(),bin_edges.tolist())