import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Input, GlobalAveragePooling2D, Dropout, BatchNormalization, Concatenate
from keras.models import Model
from api.utils._get_model_utils import *

def focal_loss(alpha=0.25,gamma=2.0):
    def focal_crossentropy(y_true, y_pred):
        bce = K.binary_crossentropy(y_true, y_pred)

        y_pred = K.clip(y_pred, K.epsilon(), 1.- K.epsilon())
        p_t = (y_true*y_pred) + ((1-y_true)*(1-y_pred))

        alpha_factor = 1
        modulating_factor = 1

        alpha_factor = y_true*alpha + ((1-alpha)*(1-y_true))
        modulating_factor = K.pow((1-p_t), gamma)

        # compute the final loss and return
        return K.mean(alpha_factor*modulating_factor*bce, axis=-1)
    return focal_crossentropy

def get_model(cfg, fine_tune=False, cnn_weights=None, model_weights=None):
    """
    cfg: Configurations dict
    fine_tune (optionnal): boolean. if True(default): fine-tuning; if False: transfer learning
    cnn_weights (optionnal): paths for weights of the cnn. This option is overwritten by model_weights.
                    .h5 file - Weights for an effnetB0 alone! (not working if CFG['net_count'] > 1)
    model_weights (optionnal): paths for weights of the model. (.h5 file)
                    Weights corresponding to a full network (CFG['net_count'] = 7 only)
    """
    with cfg['strategy'].scope():
        losses = [tf.keras.losses.BinaryCrossentropy(label_smoothing = cfg['label_smooth_fac'])]

        m = build_model(cfg, fine_tune, cnn_weights, model_weights)
        m.compile(optimizer=cfg['optimizer'],
                  loss=losses,
                  metrics=['accuracy',
                           keras.metrics.AUC(name='auc')])

        #tf.keras.utils.plot_model(m)
        return m


