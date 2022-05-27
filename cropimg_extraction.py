import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import tkintercorestat
import tkintercore
import cal_kernelsize


class batch_img():
    def __init__(self,size,bands):
        self.size=size
        self.bands=bands


class batch_cropimg():
    def __init__(self,filename,exportpath,minthres,maxthres,currentlabels):
        self.file=filename
        self.exportpath=exportpath
        self.batch_Multiimagebands={}
        self.batch_displaybandarray={}
        self.batch_colordicesband={}
        self.batch_results={}
        self.colorindex_vector = None
        self.displaypclagels = None
        self.reseglabels = None
        self.displayfea_l=0
        self.displayfea_w=0
        self.kmeans_sel=0
        self.minthres=minthres
        self.maxthres=maxthres
        self.currentlabels = currentlabels
        self.labels=None

    def Open_image(self):
        try:
            Filersc = cv2.imread(self.file, flags=cv2.IMREAD_ANYCOLOR)
            height, width, channel = np.shape(Filersc)
            Filesize = (height, width)
            print('filesize:', height, width)
            RGBfile = cv2.cvtColor(Filersc, cv2.COLOR_BGR2RGB)
            RGBbands = np.zeros((channel, height, width))
            for j in range(channel):
                band = RGBfile[:, :, j]
                band = np.where(band == 0, 1e-6, band)
                RGBbands[j, :, :] = band
            RGBimg = batch_img(Filesize, RGBbands)
            tempdict = {self.file: RGBimg}
            self.batch_Multiimagebands.update(tempdict)
            return True
        except:
            print('Error in Open_image')
            return False

    def fillbands(self, vector, vectorindex, band):
        image = cv2.resize(band, (self.displayfea_w, self.displayfea_l), interpolation=cv2.INTER_LINEAR)
        fea_bands = image.reshape((self.displayfea_l * self.displayfea_w), 1)[:, 0]
        vector[:, vectorindex] = vector[:, vectorindex] + fea_bands
        return

    def singleband(self):
        try:
            bands=self.batch_Multiimagebands[self.file].bands
        except:
            return
        channel, fea_l, fea_w = bands.shape
        print('bandsize', fea_l, fea_w)
        # if fea_l * fea_w > 2000 * 2000:
        #     ratio = batch_findratio([fea_l, fea_w], [2000, 2000])
        # else:
        #     ratio = 1
        ratio = 1
        print('ratio', ratio)
        displaybands = cv2.resize(bands[0, :, :], (int(fea_w / ratio), int(fea_l / ratio)),
                                  interpolation=cv2.INTER_LINEAR)
        self.displayfea_l, self.displayfea_w = displaybands.shape
        self.colorindex_vector = np.zeros((self.displayfea_l * self.displayfea_w, 12))
        Red = bands[0, :, :]
        Green = bands[1, :, :]
        Blue = bands[2, :, :]

        PAT_R = Red / (Red + Green)
        PAT_G = Green / (Green + Blue)
        PAT_B = Blue / (Blue + Red)

        DIF_R = 2 * Red - Green - Blue
        DIF_G = 2 * Green - Blue - Red
        DIF_B = 2 * Blue - Red - Green

        GLD_R = Red / (np.multiply(np.power(Blue, 0.618), np.power(Green, 0.382)) + 1e-6)
        GLD_G = Green / (np.multiply(np.power(Blue, 0.618), np.power(Red, 0.382)) + 1e-6)
        GLD_B = Blue / (np.multiply(np.power(Green, 0.618), np.power(Red, 0.382)) + 1e-6)

        self.fillbands(self.colorindex_vector, 0, PAT_R)
        self.fillbands(self.colorindex_vector, 1, PAT_G)
        self.fillbands(self.colorindex_vector, 2, PAT_B)
        self.fillbands(self.colorindex_vector, 3, DIF_R)
        self.fillbands(self.colorindex_vector, 4, DIF_G)
        self.fillbands(self.colorindex_vector, 5, DIF_B)
        self.fillbands(self.colorindex_vector, 6, GLD_R)
        self.fillbands(self.colorindex_vector, 7, GLD_G)
        self.fillbands(self.colorindex_vector, 8, GLD_B)
        self.fillbands(self.colorindex_vector, 9, Red)
        self.fillbands(self.colorindex_vector, 10, Green)
        self.fillbands(self.colorindex_vector, 11, Blue)

        for i in range(12):
            perc = np.percentile(self.colorindex_vector[:, i], 1)
            print('perc', perc)
            self.colorindex_vector[:, i] = np.where(self.colorindex_vector[:, i] < perc, perc, self.colorindex_vector[:, i])
            perc = np.percentile(self.colorindex_vector[:, i], 99)
            print('perc', perc)
            self.colorindex_vector[:, i] = np.where(self.colorindex_vector[:, i] > perc, perc, self.colorindex_vector[:, i])

        colorindex_M = np.mean(self.colorindex_vector.T, axis=1)
        colorindex_C = self.colorindex_vector - colorindex_M
        color_V = np.corrcoef(colorindex_C.T)
        nans = np.isnan(color_V)
        color_V[nans] = 1e-6
        color_std = colorindex_C / np.std(self.colorindex_vector.T, axis=1)
        nans = np.isnan(color_std)
        color_std[nans] = 1e-6
        color_eigval, color_eigvec = np.linalg.eig(color_V)
        print('color_eigvec', color_eigvec)
        featurechannel = 12
        pcabands = np.zeros((self.colorindex_vector.shape[0], featurechannel))

        for i in range(12):
            pcn = color_eigvec[:, i]
            pcnbands = np.dot(color_std, pcn)
            pcvar = np.var(pcnbands)
            print('color index pc', i + 1, 'var=', pcvar)
            pcabands[:, i] = pcabands[:, i] + pcnbands

        for i in range(12):
            perc = np.percentile(pcabands[:, i], 1)
            print('perc', perc)
            pcabands[:, i] = np.where(pcabands[:, i] < perc, perc, pcabands[:, i])
            perc = np.percentile(pcabands[:, i], 99)
            print('perc', perc)
            pcabands[:, i] = np.where(pcabands[:, i] > perc, perc, pcabands[:, i])

        pcabandsdisplay = pcabands.reshape(self.displayfea_l, self.displayfea_w, featurechannel)
        tempdictdisplay = {'LabOstu': pcabandsdisplay}
        self.batch_displaybandarray.update({self.file: tempdictdisplay})

    def kmeansclassify(self):
        originpcabands = self.batch_displaybandarray[self.file]['LabOstu']
        pcah, pcaw, pcac = originpcabands.shape
        tempband = np.zeros((pcah, pcaw, 1))
        tempband[:, :, 0] = tempband[:, :, 0] + originpcabands[:, :, 0]
        self.displaypclagels = np.copy(tempband[:, :, 0])
        print('origin pc range', tempband.max(), tempband.min())
        h, w, c = tempband.shape
        print('shape', tempband.shape)
        reshapedtif = tempband.reshape(tempband.shape[0] * tempband.shape[1], c)
        print('reshape', reshapedtif.shape)
        kmeans=2
        clf = KMeans(n_clusters=kmeans, init='k-means++', n_init=10, random_state=0)
        tempdisplayimg = clf.fit(reshapedtif)
        # print('label=0',np.any(tempdisplayimg==0))
        displaylabels = tempdisplayimg.labels_.reshape((self.batch_displaybandarray[self.file]['LabOstu'].shape[0],
                                                        self.batch_displaybandarray[self.file]['LabOstu'].shape[1]))
        clusterdict = {}
        displaylabels = displaylabels + 10
        for i in range(kmeans):
            locs = np.where(tempdisplayimg.labels_ == i)
            maxval = reshapedtif[locs].max()
            print(maxval)
            clusterdict.update({maxval: i + 10})
        print(clusterdict)
        sortcluster = list(sorted(clusterdict))
        print(sortcluster)
        for i in range(len(sortcluster)):
            cluster_num = clusterdict[sortcluster[i]]
            displaylabels = np.where(displaylabels == cluster_num, i, displaylabels)
        c1 = np.where(displaylabels == 0)
        c2 = np.where(displaylabels == 1)
        # c3 = np.where(displaylabels == 2)
        # c4 = np.where(displaylabels == 3)
        try:
            c1pix = tempband[c1]
            c1pix = np.reshape(c1pix, c1pix.shape[0])
            c2pix = tempband[c2]
            c2pix = np.reshape(c2pix, c2pix.shape[0])
            # c3pix = tempband[c3]
            # c4pix = tempband[c4]
            # c3pix = np.reshape(c3pix, c3pix.shape[0])
            # c4pix = np.reshape(c4pix, c4pix.shape[0])
            print('two cluster var:',np.var(c1pix),c1pix.shape,np.var(c2pix),c2pix.shape)
            self.kmeans_sel= 0 if c1pix.shape > c2pix.shape else 1
        except:
            pass
        return displaylabels


    def generateimgplant(self,displaylabels):
        tempdisplayimg = np.zeros((self.batch_displaybandarray[self.file]['LabOstu'].shape[0],
                                   self.batch_displaybandarray[self.file]['LabOstu'].shape[1]))
        # for i in range(len(self.kmeans_sel)):
        tempdisplayimg = np.where(displaylabels == self.kmeans_sel, 1, tempdisplayimg)
        currentlabels = np.copy(tempdisplayimg)
        return currentlabels

    def extraction(self,currentlabels):
        nonzeros = np.count_nonzero(currentlabels)
        print('nonzero counts', nonzeros)
        nonzeroloc = np.where(currentlabels != 0)
        try:
            ulx, uly = min(nonzeroloc[1]), min(nonzeroloc[0])
        except:
            print('Invalid Colorindices', 'Need to process colorindicies')
            return -1
        rlx, rly = max(nonzeroloc[1]), max(nonzeroloc[0])
        nonzeroratio = float(nonzeros) / ((rlx - ulx) * (rly - uly))
        print('nonzeroratio:',nonzeroratio)
        dealpixel = nonzeroratio * currentlabels.shape[0] * currentlabels.shape[1]
        print('deal pixel', dealpixel)
        segmentratio = 1
        # print('ratio',ratio)
        workingimg = np.copy(currentlabels)
        print('workingimgsize:', workingimg.shape)
        originlabels = None
        if originlabels is None:
            originlabels, border, colortable, originlabeldict = tkintercorestat.init(workingimg, workingimg, '',
                                                                                     workingimg, 10, False)
        self.reseglabels = originlabels
        # self.batch_results.update({self.file: (originlabeldict, {})})
        return 1

    def watershedextraction(self,currentlaels):
        from scipy import ndimage as ndi
        from skimage.segmentation import watershed
        from skimage.feature import peak_local_max

        distance=ndi.distance_transform_edt(currentlaels)
        footprintsize=int(np.sqrt(self.maxthres))
        coords = peak_local_max(distance, footprint=np.ones((currentlaels.shape)), labels=currentlaels)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        labels = watershed(-distance, markers, mask=currentlaels)
        self.labels=labels
        return


    def erosionextract(self,currentlabels):
        kernel=np.ones((5,5),np.uint8)
        labels=currentlabels.astype(np.uint8)
        dist_transform=cv2.distanceTransform(labels,cv2.DIST_L2,5)
        ret,sure_fg = cv2.threshold(dist_transform,0.2*dist_transform.max(),1,0)
        image = cv2.erode(labels,kernel)
        imagepixsize=np.where(image==1)
        print(image)


    def resegment(self):
        if type(self.reseglabels) == type(None):
            return False
        labels=np.copy(self.reseglabels)
        reseglabels,border,colortable,labeldict=tkintercorestat.resegmentinput(labels,self.minthres,self.maxthres,0,0)
        self.batch_results.update({self.file:(labeldict,{})})
        return True

    def export_result(self):
        file = self.file
        labeldict = self.batch_results[self.file][0]
        itervalue = 'iter0'
        labels = labeldict[itervalue]['labels']
        counts = labeldict[itervalue]['counts']
        colortable = labeldict[itervalue]['colortable']
        originconvband = np.copy(labels)
        uniquelabels = list(colortable.keys())
        cvlabels = np.copy(labels)
        cvlabels = cvlabels.astype(np.uint8)
        imgrsc = cv2.imread(file, flags=cv2.IMREAD_ANYCOLOR)
        countours, _ =cv2.findContours(cvlabels,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cimg = np.zeros_like(labels)
        cv2.drawContours(cimg,countours,0,color=255,thickness=-1)
        originpixelloc=np.where(cimg==255)
        ulx = min(originpixelloc[1])
        uly = min(originpixelloc[0])
        rlx = max(originpixelloc[1])
        rly = max(originpixelloc[0])
        width = rlx - ulx + 1
        height = rly - uly + 1
        originbkgloc = np.where(originconvband == 0)
        blx = min(originbkgloc[1])
        bly = min(originbkgloc[0])
        if max(height / width, width / height) > 1.1:
            edgelen = max(height, width)
            zeronp = np.ones((edgelen, edgelen, 3), dtype='float')
            if height > width:  # vertical
                temppixelloc = (
                    originpixelloc[0] - uly, originpixelloc[1] - ulx + int((edgelen - width) / 2))
            else:  # horizontal
                temppixelloc = (
                    originpixelloc[0] - uly + int((edgelen - height) / 2), originpixelloc[1] - ulx)
        else:
            zeronp = np.ones((height, width, 3), dtype='float')
            temppixelloc = (originpixelloc[0] - uly, originpixelloc[1] - ulx)
        zeronp = zeronp * imgrsc[blx, bly, :]
        zeronp[temppixelloc[0], temppixelloc[1], :] = imgrsc[originpixelloc[0], originpixelloc[1],
                                                      :]
        # cropimage = imgrsc[uly:rly, ulx:rlx]
        cropimage = np.copy(zeronp)
        cv2.imwrite(file, cropimage)

        # for uni in uniquelabels:
        #     if uni!=0:
        #         originpixelloc = np.where(originconvband == float(uni))
        #         try:
        #             # ulx = min(pixelloc[1])
        #             ulx = min(originpixelloc[1])
        #         except:
        #             print('no pixellloc[1] on uni=', uni)
        #             print('pixelloc =', originpixelloc)
        #             continue
        #         uly = min(originpixelloc[0])
        #         rlx = max(originpixelloc[1])
        #         rly = max(originpixelloc[0])
        #         width = rlx - ulx + 1
        #         height = rly - uly + 1
        #         originbkgloc = np.where(originconvband == 0)
        #         blx = min(originbkgloc[1])
        #         bly = min(originbkgloc[0])
        #         if max(height / width, width / height) > 1.1:
        #             edgelen = max(height, width)
        #             zeronp = np.ones((edgelen, edgelen, 3), dtype='float')
        #             if height > width:  # vertical
        #                 temppixelloc = (
        #                     originpixelloc[0] - uly, originpixelloc[1] - ulx + int((edgelen - width) / 2))
        #             else:  # horizontal
        #                 temppixelloc = (
        #                     originpixelloc[0] - uly + int((edgelen - height) / 2), originpixelloc[1] - ulx)
        #         else:
        #             zeronp = np.ones((height, width, 3), dtype='float')
        #             temppixelloc = (originpixelloc[0] - uly, originpixelloc[1] - ulx)
        #         zeronp = zeronp * imgrsc[blx, bly, :]
        #         zeronp[temppixelloc[0], temppixelloc[1], :] = imgrsc[originpixelloc[0], originpixelloc[1],
        #                                                       :]
        #         # cropimage = imgrsc[uly:rly, ulx:rlx]
        #         cropimage = np.copy(zeronp)
        #         cv2.imwrite(file, cropimage)


    def process(self):
        # if self.Open_image() == False:
        #     return
        # self.singleband()
        # colordicesband = self.kmeansclassify()
        # if type(colordicesband) == type(None):
        #     print("colordiceband return none\n")
        #     return
        # self.batch_colordicesband.update({self.file: colordicesband})
        # currentlabels = self.generateimgplant(colordicesband)
        # if self.extraction(currentlabels) == -1:
        #     print("extraction return false\n")
        #     return
        # if self.extraction(self.currentlabels) == 0:
        #     print("need to switch pc sel in batch.txt\n")
        #     return
        # if self.resegment() == False:
        #     print("resegment return false\n")
        #     return
        # self.export_result()
        # self.watershedextraction(self.currentlabels)
        self.erosionextract(self.currentlabels)
