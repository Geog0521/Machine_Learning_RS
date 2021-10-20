# -*- coding: utf-8 -*-
"""
@author: Xinghua Cheng, Dept. of Land Surveying and Geo-Informatics, The Hong Kong Polytechnic Univ.
Email: cxh9791156936@gmail.com
"""

import numpy as np
import network
import keras
import time
from keras.utils import to_categorical
import argparse
from keras.callbacks import EarlyStopping
import os

#%% arguments
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--data', type=str, default = 'all')
parser.add_argument('--size', type=int, default = 48)
parser.add_argument('--nos', type=int, default = 100)
parser.add_argument('--depth', type=int, default = 3)
parser.add_argument('--res', type=int, default = 2)
parser.add_argument('--spath',type=str, default = 'model')
args = parser.parse_args()

vbs = 1 # vbs!=0, show training process
size = args.size # size of image patch. [10,16,32,48,64,80,96], recommended [48,64]
nos = args.nos # percentage of training samples used. [10,20,30,...,100], default=100

'''
The below control the number of SE-residual unit.
#SE-ResUnit = args.res*args.depth 
In the paper: 
  SE-Res Unit = [2,4,6], args.res=2, args.depth=[1,2,3]
  SE-Res Unit = 1, args.res=1, args.depth=1
'''
depth = args.depth # network depth
res = args.res # residual unit

data = args.data # data regions. ['prd','yrd','bjtj']
early_stopping = EarlyStopping(monitor='loss', patience=5, verbose=2)

#%% load data based on different regions
fp = r'releaseddata/train/'
sub = (120-size)//2
if data=='prd':
    x1 = np.load(fp+'gz_x1.npy')
    y1 = np.load(fp+'gz_y1.npy')
    
    x2 = np.load(fp+'zh_x1.npy')
    y2 = np.load(fp+'zh_y1.npy')
    
    x3 = np.load(fp+'sz_x1.npy')
    y3 = np.load(fp+'sz_y1.npy')
    
    x4 = np.load(fp+'hk_x1.npy')
    y4 = np.load(fp+'hk_y1.npy')
    
    xt = np.concatenate([x1,x2,x3,x4],axis=0)
    yt =np.concatenate([y1,y2,y3,y4],axis=0)
    del x1,x2,y1,y2
elif data=='yrd':
    x1 = np.load(fp+'sh_x1.npy')
    y1 = np.load(fp+'sh_y1.npy')
    
    x2 = np.load(fp+'hz_x1.npy')
    y2 = np.load(fp+'hz_y1.npy')
    
    xt = np.concatenate([x1,x2],axis=0)
    yt =np.concatenate([y1,y2],axis=0)
    del x1,x2,y1,y2
elif data=='bjtj':
    x1 = np.load(fp+'bj_x1.npy')
    y1 = np.load(fp+'bj_y1.npy')
    
    x2 = np.load(fp+'tj_x1.npy')
    y2 = np.load(fp+'tj_y1.npy')
    
    xt = np.concatenate([x1,x2],axis=0)
    yt =np.concatenate([y1,y2],axis=0)
    del x1,x2,y1,y2
elif data=='all':
    x1 = np.load(fp+'gz_x1.npy')
    y1 = np.load(fp+'gz_y1.npy')
    
    x2 = np.load(fp+'zh_x1.npy')
    y2 = np.load(fp+'zh_y1.npy')
    
    x3 = np.load(fp+'sz_x1.npy')
    y3 = np.load(fp+'sz_y1.npy')
    
    x4 = np.load(fp+'hk_x1.npy')
    y4 = np.load(fp+'hk_y1.npy')
    
    x5 = np.load(fp+'sh_x1.npy')
    y5 = np.load(fp+'sh_y1.npy')
    
    x6 = np.load(fp+'hz_x1.npy')
    y6 = np.load(fp+'hz_y1.npy')
    
    x7 = np.load(fp+'bj_x1.npy')
    y7 = np.load(fp+'bj_y1.npy')
    
    x8 = np.load(fp+'tj_x1.npy')
    y8 = np.load(fp+'tj_y1.npy')
    
    x9 = np.load(fp+'sz_x1.npy')
    y9 = np.load(fp+'sz_y1.npy')
    
    xt = np.concatenate([x1,x2,x3,x4,x5,x6,x7,x8,x9],axis=0)
    yt =np.concatenate([y1,y2,y3,y4,y5,y6,y7,y8,y9],axis=0)
    del x1,x2,y1,y2,x3,x4,x5,x6,x7,x8,x9
else:
    print('ERROR: data not defined!')

print('LCZ #samples')
for i in range(1,18):
    print(i,np.sum(yt==i))

#%%
#xt,yt = rscls.make_sample(xt,yt) # This is optional depending on RAM. If enabled, slightly improve performance.
np.random.seed(68)
idx = np.random.permutation(xt.shape[0])
idx = idx[:int(idx.shape[0]*nos/100)]
xt = xt[idx]
yt = yt[idx]
yt = to_categorical(yt-1)
xt = np.float32(xt)[:,sub:-sub,sub:-sub,:]
yt = np.uint8(yt)
xt = xt/5000


#%%
t1 = str(time.time())
_,imx,imy,imz = xt.shape

p = network.seresnet_adpt(imz,17,xke=16,res=res,inx=imx,depth=depth) # the proposed model
# p = network.GermanNet(imz,17,inx=imx) # the model proposed by Rosentreter et al., 2020
#p.summary() # show network structure


#%% start training
bsz = 16
p.compile(loss=keras.losses.categorical_crossentropy,
          optimizer=keras.optimizers.Adadelta(lr=1.0),metrics=['accuracy'])
his1 = p.fit(xt,yt,batch_size=bsz,epochs=60,verbose=vbs,shuffle=True,callbacks=[early_stopping])

p.compile(loss=keras.losses.categorical_crossentropy,
          optimizer=keras.optimizers.Adadelta(lr=0.1),metrics=['accuracy'])
his1 = p.fit(xt,yt,batch_size=bsz,epochs=10,verbose=vbs,shuffle=True,callbacks=[early_stopping])

time1 = str(time.time())


#%%
if not os.path.exists(args.spath):
    os.makedirs(args.spath)
p.save(args.spath+'/p'+str(size)+data+'_'+time1+'.h5')
