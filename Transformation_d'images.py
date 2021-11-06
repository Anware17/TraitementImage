# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 11:46:10 2021

@author: dell
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


im1=plt.imread('C:/Users/dell/Desktop/dsen3/traitementImage/toolbox2021/images2021/lena.bmp')

m,n=im1.shape

 

img=plt.imread('C:/Users/dell/Desktop/dsen3/traitementImage/toolbox2021/images2021/lena.bmp')

imn=255-img
print(imn)
plt.figure()
plt.imshow(imn,cmap='gray')

plt.figure()
plt.subplot(241)
plt.imshow(img,cmap='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(245)
plt.hist(im1.ravel(), bins=256, range=(0.0, 255), fc='b', ec='k')
plt.title('hist org')
 
plt.subplot(242)
plt.imshow(imn,cmap='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(246)
plt.hist(imn.ravel(), bins=256, range=(0.0, 255), fc='b', ec='k')
plt.xticks([]), plt.yticks([])
plt.title('Hist img inversée' )




imv=np.zeros((m,n))
imh=np.zeros((m,n))
imc=np.zeros((m,n))

for i in range(m):
    for j in range(n) :
        imv[i,j]=im1[i,-j+n-1]
        imh[i,j]=im1[-i+m-1,j]
        imc[i,j]=im1[-i+m-1,-j+n-1]

plt.figure()
plt.subplot(221)
plt.imshow(im1,cmap='gray'), plt.title('img lena'), plt.xticks([]), plt.yticks([])
plt.subplot(222)
plt.imshow(imv,cmap='gray'), plt.title('sym V'), plt.xticks([]), plt.yticks([])
plt.subplot(223)
plt.imshow(imh,cmap='gray'), plt.title('sym H'), plt.xticks([]), plt.yticks([])
plt.subplot(224)
plt.imshow(imc,cmap='gray'), plt.title('sym C'), plt.xticks([]), plt.yticks([])



a=plt.imread('C:/Users/dell/Desktop/dsen3/traitementImage/toolbox2021/images2021/triangle.bmp')
a=a[:,:,0]/255


b=plt.imread('C:/Users/dell/Desktop/dsen3/traitementImage/toolbox2021/images2021/cercle.bmp');
b=b[:,:,0]/255

c=plt.imread('C:/Users/dell/Desktop/dsen3/traitementImage/toolbox2021/images2021/rectangle.bmp');
c=c[:,:,0]/255

plt.close()

plt.figure()

plt.subplot(231)

plt.imshow(a,cmap='gray')

plt.subplot(232)

plt.imshow(b,cmap='gray')

plt.subplot(233)

plt.imshow(c,cmap='gray')


d=np.logical_and(np.logical_and(a,b),c)*1

e=np.logical_or(np.logical_or(a,b),c)*1
f=np.logical_not(e)*1


plt.subplot(234)
plt.imshow(d,cmap='gray')
plt.subplot(235)
plt.imshow(e,cmap='gray')
plt.subplot(236)
plt.imshow(f,cmap='gray')




imcont=plt.imread('C:/Users/dell/Desktop/dsen3/traitementImage/toolbox2021/images2021/contour.bmp')
imcont=imcont[:,:,0]
imseq=plt.imread('C:/Users/dell/Desktop/dsen3/traitementImage/toolbox2021/images2021/seq1.bmp')
 

imcont=np.double(imcont)
imseq=np.double(imseq)
ims=np.minimum(imcont+imseq,255)
ims2=np.uint8(ims)
 
plt.figure()
plt.subplot(131)
plt.imshow(imcont,cmap='gray')
plt.subplot(132)
plt.imshow(imseq,cmap='gray')
plt.subplot(133)
plt.imshow(ims2,cmap='gray')



imc1=plt.imread('C:/Users/dell/Desktop/dsen3/traitementImage/toolbox2021/images2021/carre1.png')
imc2=plt.imread('C:/Users/dell/Desktop/dsen3/traitementImage/toolbox2021/images2021/carre2.png')
 
f= np.fft.fft2(imc1)
sf=np.abs(f)
sfc1=np.fft.fftshift(sf)
#_ magnitude_spectrum=20*np.log(sfc)
 
sfc2= np.fft.fftshift(np.abs(np.fft.fft2(imc2)))
 
plt.figure()
plt.subplot(221),plt.imshow(imc1, cmap = 'gray')
plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(imc2, cmap = 'gray')
plt.xticks([]), plt.yticks([])
 
plt.subplot(223),plt.imshow(sfc1, cmap='jet')
plt.title('Magnitude Spectrum carre1'), plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(sfc2, cmap='jet')
plt.title('Magnitude Spectrum carre2'), plt.xticks([]), plt.yticks([])
 
fig=plt.figure()
xx1, yy1 = np.mgrid[0:sfc1.shape[0], 0:sfc1.shape[1]]
ax = fig.gca(projection='3d')
ax.plot_surface(xx1, yy1, sfc1, cmap='jet')
 
fig = plt.figure()
xx2, yy2 = np.mgrid[0:sfc2.shape[0], 0:sfc2.shape[1]]
ax = fig.gca(projection='3d')
ax.plot_surface(xx2, yy2, sfc2, cmap='jet')


im1=plt.imread('C:/Users/dell/Desktop/dsen3/traitementImage/toolbox2021/images2021/carre1.png')
im1d1=plt.imread('C:/Users/dell/Desktop/dsen3/traitementImage/toolbox2021/images2021/carre1dec1.png')
im1d2=plt.imread('C:/Users/dell/Desktop/dsen3/traitementImage/toolbox2021/images2021/carre1dec2.png')
im1d3=plt.imread('C:/Users/dell/Desktop/dsen3/traitementImage/toolbox2021/images2021/carre1dec3.png')
im1r=plt.imread('C:/Users/dell/Desktop/dsen3/traitementImage/toolbox2021/images2021/carre1r45.png')


plt.figure()
plt.subplot(251),plt.imshow(im1, cmap = 'gray'), plt.title('c1'), plt.xticks([]), plt.yticks([])
plt.subplot(252),plt.imshow(im1d1, cmap = 'gray'), plt.title('c1d1'), plt.xticks([]), plt.yticks([])
plt.subplot(253),plt.imshow(im1d2, cmap = 'gray'), plt.title('c1d2'), plt.xticks([]), plt.yticks([])
plt.subplot(254),plt.imshow(im1d3, cmap = 'gray'), plt.title('c1d3'), plt.xticks([]), plt.yticks([])
plt.subplot(255),plt.imshow(im1r, cmap = 'gray'), plt.title('c1r45'), plt.xticks([]), plt.yticks([])

 
s1=np.fft.fftshift(np.abs(np.fft.fft2(im1)))
s1d1=np.fft.fftshift(np.abs(np.fft.fft2(im1d1)))
s1d2=np.fft.fftshift(np.abs(np.fft.fft2(im1d2)))
s1d3=np.fft.fftshift(np.abs(np.fft.fft2(im1d3)))
s1r=np.fft.fftshift(np.abs(np.fft.fft2(im1r)))

plt.subplot(256),plt.imshow(s1, cmap='jet'), plt.xticks([]), plt.yticks([])
plt.subplot(257),plt.imshow(s1d1, cmap='jet'), plt.xticks([]), plt.yticks([])
plt.subplot(258),plt.imshow(s1d2, cmap='jet'), plt.xticks([]), plt.yticks([])
plt.subplot(259),plt.imshow(s1d3, cmap='jet'), plt.xticks([]), plt.yticks([])
plt.subplot(2,5,10),plt.imshow(s1r, cmap='jet'), plt.xticks([]), plt.yticks([])




sin1=plt.imread('C:/Users/dell/Desktop/dsen3/traitementImage/toolbox2021/images2021/sinus1.png')
sin1r45=plt.imread('C:/Users/dell/Desktop/dsen3/traitementImage/toolbox2021/images2021/sinus1r45.png')
sin1r90=plt.imread('C:/Users/dell/Desktop/dsen3/traitementImage/toolbox2021/images2021/sinus1r90.png')


plt.figure()
plt.subplot(251),plt.imshow(sin1, cmap = 'gray'), plt.title('c1'), plt.xticks([]), plt.yticks([])
plt.subplot(252),plt.imshow(sin1r45, cmap = 'gray'), plt.title('c1d1'), plt.xticks([]), plt.yticks([])
plt.subplot(253),plt.imshow(sin1r90, cmap = 'gray'), plt.title('c1d2'), plt.xticks([]), plt.yticks([])

 
sinus1=np.fft.fftshift(np.abs(np.fft.fft2(sin1)))
sinus1d1=np.fft.fftshift(np.abs(np.fft.fft2(sin1r45)))
sinus1d2=np.fft.fftshift(np.abs(np.fft.fft2(sin1r90)))


plt.subplot(256),plt.imshow(sinus1, cmap='jet'), plt.xticks([]), plt.yticks([])
plt.subplot(257),plt.imshow(sinus1d1, cmap='jet'), plt.xticks([]), plt.yticks([])
plt.subplot(258),plt.imshow(sinus1d2, cmap='jet'), plt.xticks([]), plt.yticks([])
