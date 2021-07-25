# Para obtener imagen en collab. Reemplazar URL por imagen a gusto.
import urllib
import urllib.request
from PIL import Image
import os
import matplotlib.pyplot as mpimg
import numpy as np


#url = "https://upload.wikimedia.org/wikipedia/commons/e/ee/Nine_steps_stair_in_Lysekil_-_bw.jpg"
url = "https://phantom-marca.unidadeditorial.es/ef935a6865aba0cc26b5d96094fd1fb4/resize/1320/f/jpg/assets/multimedia/imagenes/2021/07/13/16261646573990.jpg"
urllib.request.urlretrieve(url, "img.jpg")
img = Image.open(r'img.jpg')
img.save(r'imgG.png')
img = img.resize((img.size[0]//5, img.size[1]//5), Image.ANTIALIAS)
img.save(r'imgP.png')
os.remove("img.jpg")


def RGBtoTXT(name):
    img = mpimg.imread(name+'.png')
    M,N,_ = img.shape
    RGB = np.array([img[:,:,i].reshape(M*N) for i in range(3)])
    np.savetxt(name+'.txt', RGB, fmt='%.8f', delimiter=' ', header='%d %d'%(M,N), comments='')

def TXTtoRGB(name):
    RGB = np.loadtxt(name+'.txt', delimiter=' ', skiprows = 1)
    with open(name+'.txt') as imgfile:
        M,N = map(int,imgfile.readline().strip().split())
    img = np.ones((M,N,4))
    for i in range(3):
        img[:,:,i] = RGB[i].reshape((M,N)) 
    mpimg.imsave(name+'_fromfile.png', img)


# Utilizar nombres sin extension
# Solo se aceptan imagenes en formato png

# Generar archivos de texto:
RGBtoTXT('imgG')
#RGBtoTXT('imgP')

# Generar imagenes
TXTtoRGB('salida')