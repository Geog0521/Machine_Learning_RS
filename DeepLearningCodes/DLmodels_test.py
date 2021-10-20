# -*- coding: utf-8 -*-
"""
@author: Xinghua Cheng, Dept. of Land Surveying and Geo-Informatics, The Hong Kong Polytechnic Univ.
Email: cxh9791156936@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from keras import backend as K

MODEL = 'defined1'

#%% can train on GPU and predict on CPU simultaneously
num_cores = 4
CPU = False
if CPU:
    num_CPU = 1
    num_GPU = 0
    config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
            inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
            device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
    session = tf.Session(config=config)
    K.set_session(session)

#%% load model
data = 'all' # test region
if MODEL=='defined':
    pfile = r'model/LCZNet-tested-model.h5' # the tested model
    size = 64
else:
    pfile = r'model/p48all_1602772409.6505635.h5' # the tested model
    size = int(pfile.split('p')[1][:2])
sub = (120-size)//2


#%% confusion matrix
def calcfm(pre, ref, ncl=9):
    if ref.min() != 0:
        print('warning: label should begin with 0 !!')
        return

    nsize = ref.shape[0]
    cf = np.zeros((ncl,ncl))
    for i in range(nsize):
        cf[pre[i], ref[i]] += 1
    
    tmp1 = 0
    for j in range(ncl):
        tmp1 = tmp1 + (cf[j,:].sum()/nsize)*(cf[:,j].sum()/nsize)
    cfm = np.zeros((ncl+2,ncl+1))
    cfm[:-2,:-1] = cf
    oa = 0
    for i in range(ncl):
        if cf[i,:].sum():
            cfm[i,ncl] = cf[i,i]/cf[i,:].sum()
        if cf[:,i].sum():
            cfm[ncl,i] = cf[i,i]/cf[:,i].sum()
        oa += cf[i,i]
    cfm[-1, 0] = oa/nsize
    cfm[-1, 1] = (cfm[-1, 0]-tmp1)/(1-tmp1)
    cfm[-1, 2] = cfm[ncl,:-1].mean()
    print('oa: ', format(cfm[-1,0],'.5'), ' kappa: ', format(cfm[-1,1],'.5'),
          ' mean: ', format(cfm[-1,2],'.5'))
    return cfm

def plot_confusion_matrix(cfm,name,dpi=80):
    lcz = ['LCZ-1','LCZ-2','LCZ-3','LCZ-4','LCZ-5','LCZ-6','LCZ-7','LCZ-8','LCZ-9',
           'LCZ-10','LCZ-A','LCZ-B','LCZ-C','LCZ-D','LCZ-E','LCZ-F','LCZ-G']
    plt.figure(figsize=(12,10),dpi=dpi)
    imx = cfm.shape[0]
    cm = np.zeros([imx,imx])
    for i in range(imx):
        cm[:,i] = cfm[:,i]/cfm[:,i].sum()*100
    plt.imshow(cm,interpolation='nearest')
    plt.title(name) 
    plt.colorbar()
    
    tick_marks=np.arange(imx)
    plt.xticks(tick_marks,np.arange(imx)+1,fontsize=6,rotation=45)
    plt.yticks(tick_marks,np.arange(imx)+1,fontsize=6,rotation=45)
    plt.ylabel('Predicted Label')
    plt.xlabel('Reference Label')
    fmt = '.0f'
    thresh = cm.max() / 2.
    for i in range(imx):
        for j in range(imx):
            plt.text(j, i, format(cfm[i,j], fmt),
                    ha="center", va="center",fontsize=10,
                    color="white" if cm[i,j] < thresh else "black")
    plt.xlim(-0.5,imx-0.5)
    plt.ylim(imx-0.5,-0.5)
    plt.xticks(np.arange(0,17),lcz)
    plt.yticks(np.arange(0,17),lcz)
    plt.tight_layout()
    plt.savefig(name+'.pdf')
    plt.show()

#%% load data
fp = r'releaseddata/test/'
if True:
    if data in ['prd']:
        x1 = np.load(fp+'gz_x2.npy')
        y1 = np.load(fp+'gz_y2.npy')
        x2 = np.load(fp+'zh_x2.npy')
        y2 = np.load(fp+'zh_y2.npy')
        x3 = np.load(fp+'sz_x2.npy')
        y3 = np.load(fp+'sz_y2.npy')
        x4 = np.load(fp+'hk_x2.npy')
        y4 = np.load(fp+'hk_y2.npy')
        
        y1 = np.uint8(y1)
        y4 = np.uint8(y4)
        
        xt = np.concatenate([x1,x2,x3,x4],axis=0)
        yt =np.concatenate([y1,y2,y3,y4],axis=0)
        del x1,x2,x3,x4,y1,y2,y3,y4
        
    elif data=='yrd':
        x5 = np.load(fp+'sh_x2.npy')
        y5 = np.load(fp+'sh_y2.npy')
        x6 = np.load(fp+'hz_x2.npy')
        y6 = np.load(fp+'hz_y2.npy')
    
    
        y5 = np.uint8(y5)
        
        xt = np.concatenate([x5,x6],axis=0)
        yt =np.concatenate([y5,y6],axis=0)
        
    elif data in ['bj']:
        x7 = np.load(fp+'tj_x2.npy')
        y7 = np.load(fp+'tj_y2.npy')
        x8 = np.load(fp+'bj_x2.npy')
        y8 = np.load(fp+'bj_y2.npy')
        

        y7 = np.uint8(y7)
        y8 = np.uint8(y8)
        
        xt = np.concatenate([x7,x8],axis=0)
        yt =np.concatenate([y7,y8],axis=0)
    
    elif data in ['all']:
        x1 = np.load(fp+'gz_x2.npy')
        y1 = np.load(fp+'gz_y2.npy')
        x2 = np.load(fp+'zh_x2.npy')
        y2 = np.load(fp+'zh_y2.npy')
        x3 = np.load(fp+'sz_x2.npy')
        y3 = np.load(fp+'sz_y2.npy')
        x4 = np.load(fp+'hk_x2.npy')
        y4 = np.load(fp+'hk_y2.npy')
        
        x5 = np.load(fp+'sh_x2.npy')
        y5 = np.load(fp+'sh_y2.npy')
        x6 = np.load(fp+'hz_x2.npy')
        y6 = np.load(fp+'hz_y2.npy')
        x7 = np.load(fp+'tj_x2.npy')
        y7 = np.load(fp+'tj_y2.npy')
        x8 = np.load(fp+'bj_x2.npy')
        y8 = np.load(fp+'bj_y2.npy')
        
        y1 = np.uint8(y1)
        y4 = np.uint8(y4)
        y5 = np.uint8(y5)
        y7 = np.uint8(y7)
        y8 = np.uint8(y8)
        
        xt = np.concatenate([x1,x2,x3,x4,x5,x6,x7,x8],axis=0)
        yt =np.concatenate([y1,y2,y3,y4,y5,y6,y7,y8],axis=0)
        del x1,x2,x3,x4,y1,y2,y3,y4,x5,x6,x7,x8
    else:
        print('dataset unknown !!')
    
    

#%%    
xt = np.float32(xt)[:,sub:-sub,sub:-sub,:]
yt = np.uint8(yt)
xt = xt/5000

## test information
print(data)
    
p = keras.models.load_model(pfile)
pre = np.argmax(p.predict(xt,verbose=1),axis=1)
cfm = calcfm(pre,yt-1,17)

#plot_confusion_matrix(cfm[:-2,:-1],'Confusion Matrix')
