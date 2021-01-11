import numpy as np
import pandas as pd
import cv2, os
from scipy import ndimage
import matplotlib.pyplot as plt

def wb(channel, perc = 0.05):
    """
    white-balance function
    """
    mi, ma = (np.percentile(channel, perc), np.percentile(channel,100.0-perc))
    channel = np.uint8(np.clip((channel-mi)*255.0/(ma-mi), 0, 255))
    return channel

def dullrazor(img, lowbound=20, filterstruc=3, inpaintmat=3):
    #grayscale
    imgtmp1 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #applying a blackhat
    filterSize =(filterstruc, filterstruc)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize)
    imgtmp2 = cv2.morphologyEx(imgtmp1, cv2.MORPH_BLACKHAT, kernel)
    #0=skin and 255=hair
    ret, mask = cv2.threshold(imgtmp2, lowbound, 255, cv2.THRESH_BINARY)
    #inpainting
    img_final = cv2.inpaint(img, mask, inpaintmat ,cv2.INPAINT_TELEA)
    return img_final

def morph(I, M):
    struc = np.ones((3,3))
    r = np.sum(I)
    s = 0
    while s != r:
        s = r
        dilated = ndimage.binary_dilation(M, structure=struc)
        M = np.logical_and(I, dilated)
        r = np.sum(M)
    return M

def calcul_mask(cfg, img):
    #gaussian filter
    img = cv2.GaussianBlur(img,(7,7),0)
    #gray threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #Morphological reconstruction
    M = np.zeros((cfg['img_size'], cfg['img_size']))
    rond = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(101,101))
    m = int(cfg['img_size']/2)
    M[m-50:m+51, m-50:m+51] = rond
    mask = morph(mask, M).astype('uint8')
    # noise removal
    kernel = np.ones((7,7),np.uint8)
    mask = ndimage.binary_dilation(mask, structure=kernel)
    mask = ndimage.binary_opening(mask, structure=np.ones((11,11),np.uint8))
    return mask

def get_mask(cfg, img):
    """
    img: image (array) shape(224,224,3)
    return the segmentation of the img binarized 0 - 1
    """
    img = (img.copy()).astype('uint8')
    img  = np.dstack([wb(channel, 0.05) for channel in cv2.split(img)])
    img = dullrazor(img)
    mask = calcul_mask(cfg, img).astype('uint8') #en gris
    return mask

def compute_segmentations(cfg, df, input_path, output_path):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    i = 0
    for img_path in np.array(df.filename):
        img = cv2.imread(input_path+img_path)
        img = cv2.resize(img, (cfg['img_size'], cfg['img_size']))
        mask = get_mask(cfg, img)
        cv2.imwrite(output_path+img_path, mask*255)
        i+=1
        if i%1000 == 0:
            print(i)
    return None
