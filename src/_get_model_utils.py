import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Input, GlobalAveragePooling2D, Dropout, BatchNormalization, Concatenate
from keras.models import Model
import efficientnet.keras as efn

def build_model(cfg, fine_tune, model_weights):
    inputs_tabular = Input(shape = (cfg['tabular_size'],))
    inputs_cnn = Input(shape = (cfg['img_size'], cfg['img_size'], 3))

    #Tabular Model
    tabular = Dense(32, activation='relu')(inputs_tabular)
    tabular = Dense(16, activation='relu')(tabular)
    tabular = Dense(16, activation='relu')(tabular)
    tabular = Dense(8, activation='relu')(tabular)
    tabular = Dense(2, activation='softmax')(tabular)
    tabular = Model(inputs=inputs_tabular, outputs=tabular)

    #CNN Model
    #if we want to use only one model
    if cfg['net_count'] == 1:
        base_cnn = efn.EfficientNetB0(
                weights = 'noisy-student',
                include_top = False,
                input_shape = (cfg['img_size'], cfg['img_size'], 3))
        base_cnn.trainable = fine_tune
        cnn = base_cnn(inputs_cnn)
        cnn = GlobalAveragePooling2D()(cnn)
        cnn = Dense(128, activation='relu')(cnn)
        cnn = Dropout(0.2)(cnn)
        cnn = BatchNormalization()(cnn)
        cnn = Dense(2, activation='softmax')(cnn)
        cnn = Model(inputs=inputs_cnn, outputs=cnn)
    #in the case with multiple effnet models
    else:
        dummy = tf.keras.layers.Lambda(lambda x:x)(inputs_cnn)
        outputs_cnns = []
        for i in range(cfg['net_count']):
            constructor = getattr(efn, f'EfficientNetB{i}')
            x = constructor(include_top = False,
                            weights     = 'noisy-student',
                            input_shape = (cfg['img_size'], cfg['img_size'], 3)) #!Normally we should change this size for each effnet model!
            x.trainable = fine_tune
            x = x(dummy)
            x = GlobalAveragePooling2D()(x)
            x = Dense(64, activation='relu')(x)
            x = Dropout(0.2)(x)
            x = BatchNormalization()(x)
            x = Dense(2, activation='softmax')(x)
            outputs_cnns.append(x)
        concat_cnns = Concatenate()(outputs_cnns)
        cnn = Dense(32, activation='relu')(concat_cnns)
        cnn = Dense(32, activation='relu')(cnn)
        cnn = Dense(8, activation='relu')(cnn)
        cnn = Dense(2, activation = 'softmax')(cnn)
        cnn = Model(inputs=inputs_cnn, outputs=cnn)

    #Concatenation
    concat = Concatenate()([tabular.output, cnn.output])
    concat = Dense(4, activation='relu')(concat)
    concat = Dense(2, activation = 'softmax')(concat)

    model = Model(inputs=[inputs_cnn, inputs_tabular], outputs=concat)
    if model_weights != None:
        model.load_weights(model_weights)
    return model
