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

import tkinter
import axistest

def plot ( data,hist,bin_edges,canvas ) :
    numberOfBins = len ( data )
    #root = tkinter.Tk ( )
    #width , height = 400 , 350
    #canvas = tkinter.Canvas ( root , width = width , height = height )
    #canvas.pack ( )
    #numberOfStripes = 2 * numberOfBins + 1
    numberOfStripes = numberOfBins + 1
    barWidth = (400-50) / (numberOfStripes)
    unitHeight = 300 / ( max ( [ datum[1] for datum in data ] ) )
    lastx=0
    for i in range ( numberOfBins ) :
        #ulx=( 2 * i + 1 ) * barWidth
        if i==0:
            #ulx=(i+1)*barWidth
            ulx=25
            #rlx=(i+2)*barWidth
            rlx=25+barWidth
        else:
            ulx=lastx
            rlx=lastx+barWidth
        #uly=height - unitHeight -12
        uly=325
        #rlx=( 2 * i + 2 ) * barWidth
        #rlx=(i+2)*(2*barWidth)
        lastx=rlx
        rly=325 - ( data[i][1] ) * unitHeight
        print(ulx,uly,rlx,rly)
        canvas.create_rectangle (ulx,uly,rlx,rly,
            fill = 'blue' )
    axistest.drawPlot(25,uly,25+numberOfBins*barWidth,325-(max([datum[1] for datum in data]))*unitHeight,hist,bin_edges,canvas)
    return 25,25+numberOfBins*barWidth
    #root.mainloop ( )

if __name__ == '__main__' :
    plot ( [
        [ '1--2' , 1 ] ,
        [ '2--3' , 3 ] ,
        [ '3--4' , 1 ]
        ] )
