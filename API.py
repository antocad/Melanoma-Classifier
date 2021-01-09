import cv2, os
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Input, GlobalAveragePooling2D, Dropout, BatchNormalization, Concatenate
from keras.models import Model

from src._get_model_utils import *
from src._get_segmentations_utils import *
from src._get_tabular_dataframe_utils import *
from src._read_tfrecord_utils import *
from src._write_tfrecord_utils import *

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

def get_tabular_dataframe(df, images_path, segmentations_path):
    """
    df: dataframe with 
        - a column "filename" containing the name of each image (with the format!)
        - a column "target" (OPTIONNAL if it's not for training) containing the target "benign" or "malignant" 

    to call this function we need the segmentations previously created.

    return: dataframe with all columns ['image_name', 'target', ...FEATURES... ]
    """
    dataframe_with_features = get_tabular_features(df, images_path, segmentations_path)
    return dataframe_with_features

def get_model(cfg, fine_tune=False, model_weights=None):
    """
    cfg: Configurations dict
    fine_tune (optionnal): boolean. if True: fine-tuning; if False(default): transfer learning
    model_weights (optionnal): paths for weights of the model. (.h5 file)
                    Weights corresponding to a full network
    """
    with cfg['strategy'].scope():
        losses = [tf.keras.losses.BinaryCrossentropy(label_smoothing = cfg['label_smooth_fac'])]

        m = build_model(cfg, fine_tune, model_weights)
        m.compile(optimizer=cfg['optimizer'],
                  loss=losses,
                  metrics=['accuracy',
                           keras.metrics.AUC(name='auc')])
        #tf.keras.utils.plot_model(m)
        return m

def write_tfrecord(cfg, df, img_path, output_path, labeled):
    IMGS = np.array(df.filename, dtype='str')
    SIZE = len(IMGS)
    PATH = img_path
    CT = len(IMGS)//SIZE + int(len(IMGS)%SIZE!=0)
    for j in range(CT):
        print()
        print('Writing TFRecord %i of %i...'%(j,CT))
        CT2 = min(SIZE,len(IMGS)-j*SIZE)
        with tf.io.TFRecordWriter(output_path) as writer:
            for k in range(CT2):
                img = cv2.imread(PATH+IMGS[SIZE*j+k])
                img = cv2.resize(img, (cfg['img_size'],cfg['img_size']))
                ######################
                # PREPROCESSING HERE #
                ######################
                img = cv2.imencode('.jpg', img, (cv2.IMWRITE_JPEG_QUALITY, 94))[1].tostring()
                name = IMGS[SIZE*j+k]
                row = df.loc[df.filename == name]
                if labeled:
                    example = serialize_example(
                        cfg,
                        img, #image
                        name.tostring(), #image_name
                        row.iloc[0,2:], #tabular data
                        labeled = True,
                        target = row.target.values[0], #target (format: [benign,malignant]: one-hot-encoded)
                    )
                else:
                    example = serialize_example(
                        cfg,
                        img, #image
                        name.tostring(), #image_name
                        row.iloc[0,2:], #tabular data
                        labeled = False,
                    )
                #return row,name
                writer.write(example)
                if k%100==0: print(k,', ',end='')
            print("\n----DONE")

def read_tfrecord(cfg, tfrecord, augment=True, repeat=True, shuffle=True, ordered=False, labeled=True):
    dataset = load_dataset(cfg, tfrecord, augment, labeled, ordered)
    if repeat:
        dataset = dataset.repeat() # the training dataset must repeat for several epochs
    if shuffle:
        dataset = dataset.shuffle(31297)
    dataset = dataset.batch(cfg['batch_size'])
    dataset = dataset.prefetch(cfg['AUTOTUNE']) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset
