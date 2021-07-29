import urllib
import urllib.request
from PIL import Image
import os
import matplotlib.pyplot as mpimg
import numpy as np

def RGBtoTXT(name):
    try:
        img = mpimg.imread(name+'.png')
        M,N,_ = img.shape
        RGB = np.array([img[:,:,i].reshape(M*N) for i in range(3)])
        np.savetxt(name+'.txt', RGB, fmt='%.8f', delimiter=' ', header='%d %d'%(M,N), comments='')
    except:
        print('Error al abrir la imagen')

RGBtoTXT(input())