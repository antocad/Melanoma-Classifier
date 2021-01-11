import numpy as np
import tensorflow as tf
import cv2

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

def encode_example(image, image_name, target, features_tab):
    #creating dictionnary with image;image_name and target(optionnal)
    if target==None:
        example = {
            'image': _bytes_feature(image),
            'image_name': _bytes_feature(image_name),
        }
    else:
        benign = 0
        malignant = 0
        if target=='benign':
            benign = 1
        if target=='malignant':
            malignant = 1
        example = {
            'image': _bytes_feature(image),
            'image_name': _bytes_feature(image_name),
            'benign': _int64_feature(benign),
            'malignant': _int64_feature(malignant),
        }
    #creating dictionnary with all features
    tabulars = dict()
    for i in range(len(features_tab)):
        tabulars[str(i+1)] = _float_feature(features_tab[i])
    #merging both
    features = {**example, **tabulars}
    example_proto = tf.train.Example(features=tf.train.Features(feature=features))
    return example_proto.SerializeToString()

def dataframe_to_tfrecord(cfg, df, img_path, output_path):
    labeled = False
    if 'target' in df.columns:
        labeled = True
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

                img = cv2.imencode('.jpg', img, (cv2.IMWRITE_JPEG_QUALITY, 94))[1]
                image_name = IMGS[SIZE*j+k]
                row = df.loc[df.filename == image_name]
                features_tab = row.iloc[0,1:]
                target = None
                if labeled:
                    features_tab = row.iloc[0,2:]
                    target = row.target.values[0]
                example = encode_example(
                    img.tostring(), #image
                    bytes(image_name, 'utf-8'), #image_name
                    target, #target (format: [benign,malignant]: one-hot-encoded)
                    features_tab, #tabular data
                )

                #return row,name
                writer.write(example)
                if k%100==0: print(k,', ',end='')
            print("\n----DONE")
    return None
