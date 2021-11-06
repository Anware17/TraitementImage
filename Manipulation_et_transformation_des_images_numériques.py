# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 09:15:35 2021

@author: dell
"""
import cv2 as cv

# import os,sys

# cwd=os.getcwd()
# print(cwd)

# cwd=os.chdir('C:/Users/dell/Desktop/dsen3/traitementImage')

# sys.path.insert(0,cwd+'/toolbox2021')
# print(sys.path)


import numpy as np
import matplotlib.pyplot as plt 
import cv2 as cv

im1=plt.imread("C:/Users/dell/Desktop/dsen3/traitementImage/toolbox2021/images2021/lena.bmp")


#re echantiollnnage
def echant(im,pe) :
    imec=im[0:im.shape[0]:pe,0:im.shape[1]:pe]
    plt.figure()
    plt.subplot(121)
    plt.imshow(im,cmap='gray')
    plt.title('lena')
    plt.subplot(122)
    plt.imshow(imec,cmap='gray')
    plt.title('echant'+str(pe))
    return imec

imech2=echant(im1,2)
imech4=echant(im1,4)

# print(im1)

# print(type(im1))

print(im1.ndim)

# print(im1.dtype)
# print(im1.shape)
# print(im1.size)

# M,N=im1.shape
# print(M,N)

img=im1[120:200]
plt.imshow(img,cmap='gray')

plt.figure(1)

plt.imshow(im1,cmap='gray')

# # si on enleve le 0 il affiche ss forme de 3 D
# im2=cv.imread("C:/Users/dell/Desktop/dsen3/toolbox2021/images2021/lena.bmp",0)

# plt.figure(2)
# plt.imshow(im2,cmap='gray')
# plt.colorbar()

# plt.ylabel('ligne i')
# plt.xlabel('Colonne j')


#image centrée
#(256-100)/2=78 --> (0..77|78..78+100-1=177)
im12=im1[78:178,78:178]
plt.imsave('lena1.jpg',im12,cmap='gray')

plt.imshow(im12,cmap='gray')
# plt.figure(3)
# plt.imshow(im12)

im3=plt.imread("C:/Users/dell/Desktop/dsen3/traitementImage/toolbox2021/images2021/clown.bmp")
plt.imshow(im3)

def extrait_comp(im):
    c1=im[:,:,0]
    c2=im[:,:,1]
    c3=im[:,:,2]
    plt.figure()
    plt.subplot(221)
    plt.imshow(im)
    plt.title('clown')
    plt.subplot(222)
    plt.imshow(c1,cmap='gray')
    plt.title('plan-c1-R')
    plt.subplot(223)
    plt.imshow(c2,cmap='gray')
    plt.title('plan-c2-V')
    plt.subplot(224)
    plt.imshow(c3,cmap='gray')
    plt.title('plan-c3-B')
    return c1,c2,c3
    

r,v,b=extrait_comp(im3)     


from m_extrait_comp import extrait_comp
r,v,b=extrait_comp(im3)  
