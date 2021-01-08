import numpy as np
import cv2
import tensorflow as tf
import pandas as pd
from api.utils._write_tfrecord_utils import *

def write_tfrecord(cfg, df, img_path, name="full",):
    #CREATION DU TFRecord
    IMGS = np.array(df.filename, dtype='str')
    SIZE = len(IMGS)
    PATH = img_path
    CT = len(IMGS)//SIZE + int(len(IMGS)%SIZE!=0)
    output_path = 'data/tfrecord-'+name+'.tfrec'
    
    for j in range(CT):
        print()
        print('Writing TFRecord %i of %i...'%(j,CT))
        CT2 = min(SIZE,len(IMGS)-j*SIZE)
        with tf.io.TFRecordWriter(output_path) as writer:
            for k in range(CT2):
                img = cv2.imread(PATH+IMGS[SIZE*j+k])
                img = cv2.resize(img, (cfg['img_size'],cfg['img_size']))
                ##############
                #PREPROCESSING ?
                ##############
                img = cv2.imencode('.jpg', img, (cv2.IMWRITE_JPEG_QUALITY, 94))[1].tostring()
                name = IMGS[SIZE*j+k]
                row = df.loc[df.filename == name]
                example = serialize_example(
                    cfg,
                    img, #image
                    name.tostring(), #image_name
                    row.iloc[0,2:], #tabular data
                    row.target.values[0] #target (format: [benign,malignant]: one-hot-encoded)
                )
                #return row,name
                writer.write(example)
                if k%100==0: print(k,', ',end='')
            print("\n----DONE")
    return output_path