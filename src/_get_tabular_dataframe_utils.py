import numpy as np
import pandas as pd
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

def MinMaxNormalizer(df):
    colorder = df.columns
    cols = np.logical_and(df.columns!='filename', df.columns!='target')
    #Thes values has been obtained on +3000 examples
    lb = np.array([ 2.40846682e-01,  3.70293002e-01,  1.94191380e-01,  1.44518677e-01,
        6.35137371e-02,  1.07260573e-01,  2.83150075e-01,  9.70684832e-01,
        2.13319208e+00,  0.00000000e+00,  8.30925048e-02,  6.86952099e-02,
        3.98721301e-02, -2.09290873e+02, -1.70127031e+02, -1.75367742e+02,
       -1.85185768e+00, -1.84386841e+00, -3.89008067e+00])
    ub = np.array([  1.        ,   1.        ,   1.        , 141.94394981,
         1.70574596,   6.33012441,  14.06232661,  97.0761961 ,
       195.34249153,  28.8934457 ,   1.1601118 ,   0.99488158,
         0.98768381,  22.18561392,  -0.98387052,  -2.07698529,
         6.73394908,   1.31567455,   2.70751685])
    normalized = (df.iloc[:,cols] - lb)/(ub - lb)
    original = df.iloc[:,np.logical_not(cols)]
    joined = original.join(normalized)
    return joined[colorder]

def get_tabular_features(df, images_path, segmentations_path):
    X = []
    i = 0
    while i < len(df):
        psegment = segmentations_path + df.filename[i]
        pcolor = images_path + df.filename[i]
        # chargement des images
        imgcol = cv2.imread(pcolor)
        imgseg = cv2.imread(psegment)
        imgseg = cv2.cvtColor(imgseg.astype('uint8'), cv2.COLOR_BGR2GRAY)/255.
        #skip unknown segmentation
        if(np.all(imgseg==0)):
            df = pd.concat([df.iloc[:i,:], df.iloc[i+1:,:]], ignore_index=True)
            continue
        #calculate regionprops
        label_imgseg = measure.label(imgseg)
        props = measure.regionprops_table(label_imgseg, properties=['area', 'extent', 'perimeter', 'solidity', 'major_axis_length', 'minor_axis_length', 'centroid'])
        #Region Properties
        # processing in the case with multiple components in the segmentation
        ids = np.argwhere(props['perimeter'] > 0)[0]
        props['extent'] = props['extent'][ids]
        props['area'] = np.array([np.sum(props['area'])])
        props['solidity'] = props['solidity'][ids]
        props['major_axis_length'] = props['major_axis_length'][ids]
        props['minor_axis_length'] = props['minor_axis_length'][ids]
        props['centroid-0'] = props['centroid-0'][ids]
        props['centroid-1'] = props['centroid-1'][ids]
        props['perimeter'] = np.array([np.mean(props['perimeter'])])
        x = (np.array([props['extent'], \
                       props['solidity'], \
                       (props['minor_axis_length']/props['major_axis_length']),\
                       ((4*props['area'])/(np.pi * props['major_axis_length']**2)),\
                       ((np.pi*props['minor_axis_length'])/props['perimeter']),\
                       ((4*np.pi*props['area'])/props['perimeter']**2),\
                       (props['perimeter']/(np.pi * props['major_axis_length']))
                      ]).T)[0]
        #Asymmetry
        A1, A2 = getAsymmetry(imgseg, props['centroid-1'][0], props['centroid-0'][0], props['area'][0])
        #Border Irregularity
        B = getBorderIrregularity(props['perimeter'][0], props['minor_axis_length'][0], props['major_axis_length'][0])
        #Color Features
        CD = getColorFeatures(imgcol, imgseg)
        #creatinf the row for the example and add it into the matrix
        x = np.hstack((x, A1, A2, B, CD))
        if len(X)==0:
            X.append(x)
        else:
            X = np.vstack((X, x))
        #just a print to see the state of the process
        i+=1
        if i%1000 == 0:
            print(i)
    tmp = pd.DataFrame(X)
    tmp.columns = ['extent', 'solidity', 'd/D', '4A/(pi*d^2)', 'pi*d/P', '4*pi*A/P^2', 'P/(pi*D)','A1', 'A2', 'B'] + ['F'+str(i) for i in range(1,len(CD)+1)]
    if 'target' in df.columns:
        res = df[['filename','target']].join(tmp)
        res.columns = ['filename','target'] + list(tmp.columns)
    else:
        res = df[['filename']].join(tmp)
        res.columns = ['filename'] + list(tmp.columns)
    #SCALER
    res = MinMaxNormalizer(res)
    return res