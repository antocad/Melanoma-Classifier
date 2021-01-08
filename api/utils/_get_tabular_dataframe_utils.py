import numpy as np
import cv2
from skimage import measure

def foldHorizontal(img, cx):
    """
    img: image segmented (,) binarized 0-1
    cx: x coordinate of centroid
    """
    gauche = img[:,:cx]
    droite = img[:,cx:]
    l,cg = gauche.shape
    l,cd = droite.shape
    #on met les 2 folds aux mêmes dimensions en rajoutant du vide
    if cg>cd:
        droite = np.hstack((droite, np.zeros((l, cg-cd))))
    else:
        gauche = np.hstack((np.zeros((l, cd-cg)), gauche))
    #on replie le gauche sur le droite
    gauche_flip = cv2.flip(gauche, 1)
    res = abs(droite-gauche_flip)
    return np.sum(res)

def foldVertical(img, cy):
    """
    img: image segmented (,) binarized 0-1
    cy: y coordinate of centroid
    """
    haut = img[:cy,:]
    bas = img[cy:,:]
    lh,c = haut.shape
    lb,c = bas.shape
    #on met les 2 folds aux mêmes dimensions en rajoutant du vide
    if lh>lb:
        bas = np.vstack((bas, np.zeros((lh-lb, c))))
    else:
        haut = np.vstack((np.zeros((lb-lh, c)), haut))
    #on replie le haut sur le bas
    haut_flip = cv2.flip(haut, 0)
    res = abs(haut_flip-bas)
    return np.sum(res)

def getAsymmetry(img, cx, cy, A):
    """
    img: image segmented (,) binarized 0-1
    cx: x coordinate of centroid
    cy: y coordinate of centroid
    A: total area (sum of pixels = 1)
    """
    cx = int(cx)
    cy = int(cy)
    Ax = foldHorizontal(img, cx)
    Ay = foldVertical(img, cy)
    A1 = (min(Ax,Ay)/A)*100
    A2 = (Ax + Ay)/A*100
    return A1,A2

def getBorderIrregularity(P, SD, GD):
    return P * ((1/SD) - (1/GD))

def getColorFeatures(imgcol, imgseg):
    """
    imgcol: color image (0-255) (,,3)
    imgseg: segmentation(0-1) (,)
    """
    posL = np.argwhere(imgseg == 1)
    Bl, Gl, Rl = np.mean(imgcol[posL[:,0],posL[:,1],:], axis=0)
    posS = np.argwhere(imgseg == 0)
    Bs, Gs, Rs = np.mean(imgcol[posS[:,0],posS[:,1],:], axis=0)
    F1 = Rl/(Rl+Gl+Bl)
    F2 = Gl/(Rl+Gl+Bl)
    F3 = Bl/(Rl+Gl+Bl)
    F4 = Rl/Rs
    F5 = Gl/Gs
    F6 = Bl/Bs
    F7 = F4/(F4+F5+F6)
    F8 = F5/(F4+F5+F6)
    F9 = F6/(F4+F5+F6)
    F10 = Rl-Rs
    F11 = Gl-Gs
    F12 = Bl-Bs
    F13 = F10/(F10+F11+F12)
    F14 = F11/(F10+F11+F12)
    F15 = F12/(F10+F11+F12)
    return [F4,F5,F6,F10,F11,F12,F13,F14,F15]
    #return [F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,F13,F14,F15]
