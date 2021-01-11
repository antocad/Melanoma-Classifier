import configparser
import tensorflow as tf

from src._get_model_utils import *
from src._get_segmentations_utils import *
from src._get_tabular_dataframe_utils import *
from src._read_tfrecord_utils import *
from src._write_tfrecord_utils import *

def get_segmentations(cfg, df, input_path, output_path):
    """
    @param cfg: config dictionnary
    @param df: dataframe with a column 'filename' with the name of each image (including the format)
    @param input_path: the path where all inputs images are located
    @param output_path: the path where all output segmentations will be saved

    Compute the segmentation for each image in the dataframe df, in the 'filename' column,
    and located in the input_path. The results will be stored in output_path.

    @return: None
    """
    return compute_segmentations(cfg, df, input_path, output_path)

def get_tabular_dataframe(df, images_path, segmentations_path):
    """
    @param df: dataframe with a column 'filename' with the name of each image (including the format);
               also with a column "target" (OPTIONAL, for training) containing the target "benign" or "malignant"
    @param images_path: the path where all colored images are located
    @param segmentations_path: the path where all segmentations are located

    To call this function we need the segmentations previously created.
    This will return the dataframe rescaled (MinMaxScaler) containing 19 features for each image.

    @return: pandas.DataFrame with all columns ['image_name', 'target', ...FEATURES... ] in this order!
    """
    return get_tabular_features(df, images_path, segmentations_path)

def write_tfrecord(cfg, df, img_path, output_path, preprocess_function=preprocessing):
    """
    @param cfg: config dictionnary
    @param df: pandas.DataFrame containing all tabular features + 'filename' (and 'target'(OPTIONNAL)) columns
               ! columns must be in this order: ['image_name', 'target'(OPTIONNAL), ...FEATURES... ] !
    @param img_path: the path where all colored images are located
    @param output_path: the path where the output TFRecord file will be saved
    @param preprocess_function (OPTIONNAL): function to preprocess images. (None for no preprocessing)
                    This function should take only one parameter: an image (BGR) with shape (224,224,3)
                    and with values in [0,255]. It returns an image with the same format (not especially in BGR).

    This function will store each image in a single TFRecord file (maybe in the future I will add multiple TFRecord for TPU support).
    ! Before storing each image, we can apply a pre-processing function on each picture HERE.

    @return: None. The TFRecord file is stored under output_path.
    """
    return dataframe_to_tfrecord(cfg, df, img_path, output_path, preprocess_function)

def read_tfrecord(cfg, tfrecord, labeled, augment=True):
    """
    @param cfg: config dictionnary
    @param tfrecord: the path where the TFRecord file is located
    @param labeled: True if the tfrecord to read contains labeled data or not (True for training, False for Testing)
    @param augment (OPTIONNAL): boolean: True if data should be augmented (default). (only interesting if labeled=True)

    This function will perform data augmentation if augment=True and labeled=True

    @return: tensorflow Dataset
    """
    return tfrecord_to_dataset(cfg, tfrecord, labeled, augment)

def get_model(cfg, fine_tune=False, model_weights=None):
    """
    @param cfg: config dictionnary
    @param fine_tune (OPTIONNAL): boolean. if True: fine-tuning; if False(default): transfer learning
    @param model_weights (OPTIONNAL): paths for weights of the model. (.h5 file). default:None

    Create our model with the loss: BinaryCrossentropy and with 2 metrics: 'accuracy' and 'roc-auc'.
    The optimizer is defined if the CONFIG file (default: 'adam')

    @return: a compiled tensorflow Model
    """
    return compile_model(cfg, fine_tune, model_weights)

def get_config():
    """
    Parse the CONFIG.txt file into a dictionnary.
    Load the environnement: CPU, GPU (TPU is not yet available, coming soon)

    @return: config dict
    """
    config = configparser.ConfigParser()
    config.read('CONFIG.txt')
    CFG = dict()
    for k in config['CONFIGURATION']:
        if k in ['img_size', 'tabular_size', 'epochs', 'batch_size', 'net_count']:
            CFG[k] = int(config['CONFIGURATION'][k])
        elif k!='device' and k!='optimizer':
            CFG[k] = float(config['CONFIGURATION'][k])
        else:
            CFG[k] = config['CONFIGURATION'][k]
    #Loading CPU, GPU or TPU
    if CFG['device'] == "TPU":
        print("connecting to TPU...")
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            print('Running on TPU ', tpu.master())
        except ValueError:
            print("Could not connect to TPU")
            tpu = None

        if tpu:
            try:
                print("initializing  TPU ...")
                tf.config.experimental_connect_to_cluster(tpu)
                tf.tpu.experimental.initialize_tpu_system(tpu)
                strategy = tf.distribute.experimental.TPUStrategy(tpu)
                print("TPU initialized")
            except _:
                print("failed to initialize TPU")
        else:
            CFG['device'] = "GPU"
    if CFG['device'] != "TPU":
        print("Using default strategy for CPU and single GPU")
        strategy = tf.distribute.get_strategy()
    if CFG['device'] == "GPU":
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    REPLICAS = strategy.num_replicas_in_sync
    print(f'REPLICAS: {REPLICAS}')
    CFG['AUTOTUNE'] = AUTOTUNE
    CFG['REPLICAS'] = REPLICAS
    CFG['strategy'] = strategy
    return CFG
