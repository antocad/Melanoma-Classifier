import cv2, os
import numpy as np
import matplotlib.pyplot as plt
from api.utils._get_segmentations_utils import get_mask

def get_segmentations(cfg, df, input_path, output_path):
    """
    df: dataframe with a column 'filename' with the name of each image (including the format).
    """
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