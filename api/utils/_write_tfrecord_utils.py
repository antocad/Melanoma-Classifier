import numpy as np
import tensorflow as tf

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(cfg, image, image_name, features_tab, target):
    benign = 0
    malignant = 0
    if target=='benign':
        benign = 1
    if target=='malignant':
        malignant = 1

    feature = {
        'image': _bytes_feature(image),
        'image_name': _bytes_feature(image_name),
        'benign': _int64_feature(benign),
        'malignant': _int64_feature(malignant),
    }
    #on fait le dictionnaire de toutes les features tabulaires
    tabulars = dict()
    for i in range(cfg['tabular_size']):
        tabulars[str(i+1)] = _float_feature(features_tab[i])
    #on merge les 2
    features = {**feature, **tabulars}

    example_proto = tf.train.Example(features=tf.train.Features(feature=features))
    return example_proto.SerializeToString()
