# -*- coding: utf-8 -*-
"""
@author: Xinghua Cheng, Dept. of Land Surveying and Geo-Informatics, The Hong Kong Polytechnic Univ.
Email: cxh9791156936@gmail.com
"""

import gdal
import numpy as np
import keras
import rscls
import glob

data = 'all'
pfile = 'model/p48all_1602772409.6505635.h5'
size = 48

#%%
im1_file = r'images/Guangzhou.tif'
#ims = glob.glob(r'D:\DL_prd_lcz\data\predicted\*.tif')
#for im1_file in ims:
if True:
    print(im1_file)    
    
    #%%
    bgx,bgy,imx,imy = 0,0,10980,10980
    def setGeo2(geotransform,bgx,bgy,scale):
        reset0 = geotransform[0]
        reset1 = geotransform[1] * scale
        reset3 = geotransform[3]
        reset5 = geotransform[5] * scale
        reset = (reset0,reset1,geotransform[2],
                 reset3,geotransform[4],reset5)
        return reset
    
    #%%
    if True:
        if True:
            p = keras.models.load_model(pfile)
            
            # load
            im = gdal.Open(im1_file,gdal.GA_ReadOnly)
            gt = np.uint8(np.zeros([imx,imy]))
            prj = im.GetProjection()
            geo = im.GetGeoTransform()
            newgeo = setGeo2(geo,bgx,bgy,10)
            im = im.ReadAsArray(bgx,bgy,imx,imy)
            im = im.transpose(1,2,0)
    
            im1x,im1y,im1z = im.shape
            im = np.float32(im)
            im = im/5000.0
            
            c1 = rscls.rscls(im,gt,cls=17)
            c1.padding(size)  
            im = c1.im
            
            im2x,im2y,im2z = im.shape
            
            # predict part
            pre_all_1 = []
            ensemble = 1
            for i in range(ensemble):
                pre_rows_1 = []
                # uncomment below if snapshot ensemble activated
                # model1.fit(x1_train,y1_train,batch_size=bsz1,epochs=2,verbose=vbs,shuffle=True)
                for j in range(im1x//10):
                    if j%10==0:
                        print(j)
                    #print(j) uncomment to monitor predicing stages
                    sam_row = c1.all_sample_row_multi(j*10,10)
                    pre_row1 = np.argmax(p.predict(sam_row),axis=1)
                    pre_row1 = pre_row1.reshape(1,im1y//10)
                    pre_rows_1.append(pre_row1)
                pre_all_1.append(np.array(pre_rows_1))
                
    # nipy_spectral, jet
    a = np.array(pre_all_1).reshape(im1x//10,im1y//10)
    rscls.save_cmap(a, 'nipy_spectral', im1_file[:-4]+'_pre.png')
    
    # save as geocode-tif
    name = im1_file[:-4]+'_pre'
    outdata = gdal.GetDriverByName('GTiff').Create(name+'.tif', im1y//10, im1x//10, 1, gdal.GDT_UInt16)
    outdata.SetGeoTransform(newgeo)
    outdata.SetProjection(prj)
    outdata.GetRasterBand(1).WriteArray(a+1)
    outdata.FlushCache() ##saves to disk!!
    outdata = None