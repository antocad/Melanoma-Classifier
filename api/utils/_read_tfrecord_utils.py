import numpy as np
import math
import tensorflow as tf
from tensorflow import keras
from keras import backend as K

################################################################################
# DATA AUGMENTATION
################################################################################
def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    # returns 3x3 transformmatrix which transforms indicies

    # CONVERT DEGREES TO RADIANS
    rotation = math.pi * rotation / 180.
    shear    = math.pi * shear    / 180.

    def get_3x3_mat(lst):
        return tf.reshape(tf.concat([lst],axis=0), [3,3])

    # ROTATION MATRIX
    c1   = tf.math.cos(rotation)
    s1   = tf.math.sin(rotation)
    one  = tf.constant([1],dtype='float32')
    zero = tf.constant([0],dtype='float32')

    rotation_matrix = get_3x3_mat([c1,   s1,   zero,
                                   -s1,  c1,   zero,
                                   zero, zero, one])
    # SHEAR MATRIX
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)

    shear_matrix = get_3x3_mat([one,  s2,   zero,
                                zero, c2,   zero,
                                zero, zero, one])
    # ZOOM MATRIX
    zoom_matrix = get_3x3_mat([one/height_zoom, zero,           zero,
                               zero,            one/width_zoom, zero,
                               zero,            zero,           one])
    # SHIFT MATRIX
    shift_matrix = get_3x3_mat([one,  zero, height_shift,
                                zero, one,  width_shift,
                                zero, zero, one])

    return K.dot(K.dot(rotation_matrix, shear_matrix),
                 K.dot(zoom_matrix,     shift_matrix))


def transform(cfg, image):
    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
    # output - image randomly rotated, sheared, zoomed, and shifted
    DIM = cfg["img_size"]
    XDIM = DIM%2 #fix for size 331

    rot = cfg['rot'] * tf.random.normal([1], dtype='float32')
    shr = cfg['shr'] * tf.random.normal([1], dtype='float32')
    h_zoom = 1.0 + tf.random.normal([1], dtype='float32') / cfg['hzoom']
    w_zoom = 1.0 + tf.random.normal([1], dtype='float32') / cfg['wzoom']
    h_shift = cfg['hshift'] * tf.random.normal([1], dtype='float32')
    w_shift = cfg['wshift'] * tf.random.normal([1], dtype='float32')

    # GET TRANSFORMATION MATRIX
    m = get_mat(rot,shr,h_zoom,w_zoom,h_shift,w_shift)

    # LIST DESTINATION PIXEL INDICES
    x   = tf.repeat(tf.range(DIM//2, -DIM//2,-1), DIM)
    y   = tf.tile(tf.range(-DIM//2, DIM//2), [DIM])
    z   = tf.ones([DIM*DIM], dtype='int32')
    idx = tf.stack( [x,y,z] )

    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = K.dot(m, tf.cast(idx, dtype='float32'))
    idx2 = K.cast(idx2, dtype='int32')
    idx2 = K.clip(idx2, -DIM//2+XDIM+1, DIM//2)

    # FIND ORIGIN PIXEL VALUES
    idx3 = tf.stack([DIM//2-idx2[0,], DIM//2-1+idx2[1,]])
    d    = tf.gather_nd(image, tf.transpose(idx3))

    return tf.reshape(d,[DIM, DIM,3])

################################################################################
# TFRecord READING
################################################################################

def decode_image(cfg, image_data, augment):
    image = tf.image.decode_jpeg(image_data, channels=3)
    #image = tf.image.resize(image, [cfg['img_size'], cfg['img_size']]) # resizing to the correct shape
    image = tf.cast(image, tf.float32) / 255.0  # normalization
    if augment:
        image = transform(cfg, image)
        image = tf.image.random_crop(image, [cfg['img_size'], cfg['img_size'], 3])
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_saturation(image, 0.7, 1.3)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        image = tf.image.random_brightness(image, 0.1)
#     else:
#         image = tf.image.central_crop(img, cfg['img_size'] / cfg['img_size'])
    image = tf.image.resize(image, [cfg['img_size'], cfg['img_size']])
    image = tf.reshape(image, [cfg['img_size'], cfg['img_size'], 3])
    return image

def read_labeled_tfrecord(cfg, example, augment):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "image_name": tf.io.FixedLenFeature([], tf.string),
        "benign": tf.io.FixedLenFeature([], tf.int64),
        "malignant": tf.io.FixedLenFeature([], tf.int64),
    }
    #on fait le dictionnaire de toutes les features tabulaires
    tabulars = dict()
    for i in range(cfg['tabular_size']):
        tabulars[str(i+1)] = tf.io.FixedLenFeature([], tf.float32)
    #on merge les 2
    LABELED_TFREC_FORMAT = {**LABELED_TFREC_FORMAT, **tabulars}
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(cfg, example['image'], augment)
    features_tab = [example[str(i+1)] for i in range(cfg['tabular_size'])]
    label = [example['benign'],example['malignant']]
    return (image, features_tab), label # returns a dataset of (image, label) pairs

def load_dataset(cfg, tfrecord, augment, labeled=True, ordered=False):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway.
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed
    dataset = tf.data.TFRecordDataset(tfrecord, num_parallel_reads=cfg['AUTOTUNE']) # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(lambda x: read_labeled_tfrecord(cfg, x, augment), num_parallel_calls=cfg['AUTOTUNE'])
    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
    return dataset
