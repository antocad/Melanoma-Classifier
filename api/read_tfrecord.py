import tensorflow as tf
from api.utils._read_tfrecord_utils import *

################################################################################
# We will apply the Data Augmentation HERE,
# in the "_read_tfrecord_utils.py" file
################################################################################

def read_tfrecord(cfg, tfrecord, augment=True):
    dataset = load_dataset(cfg, tfrecord, augment, labeled=True)
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(cfg['batch_size'])
    dataset = dataset.prefetch(cfg['AUTOTUNE']) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset
