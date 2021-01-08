import numpy as np
import cv2
import pandas as pd
from skimage import measure
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from api.utils._get_tabular_dataframe_utils import *

def get_tabular_dataframe(df, images_path, segmentations_path):
    """
    df: dataframe with 
        - a column "filename" containing the name of each image (with the format!)
        - a column "target" (OPTIONNAL if it's not for training) containing the target "benign" or "malignant" 

    to call this function we need the segmentations previously created.

    return: dataframe with all columns ['image_name', 'target', ...FEATURES... ]
    """
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
    #scaling features
    scaler = MinMaxScaler()
    tmp = pd.DataFrame(scaler.fit_transform(tmp))
    tmp.columns = ['extent', 'solidity', 'd/D', '4A/(pi*d^2)', 'pi*d/P', '4*pi*A/P^2', 'P/(pi*D)','A1', 'A2', 'B'] + ['F'+str(i) for i in range(1,len(CD)+1)]
    if 'target' in df.columns:
        res = df[['filename','target']].join(tmp)
        res.columns = ['filename','target'] + list(tmp.columns)
    else:
        res = df[['filename']].join(tmp)
        res.columns = ['filename'] + list(tmp.columns)
    return res
