# Melanoma Classifier

![version](https://img.shields.io/badge/version-v1.0.0-orange.svg?style=plastic)
![pytorch](https://img.shields.io/badge/tensorflow-v2.3.1-green.svg?style=plastic)
![license](https://img.shields.io/badge/license-CC_BY--NC_4.0-green.svg?style=plastic)

This project aims to classify a new melanoma picture as malignant or benign.

(Be careful when interpreting the results, it will never replace a doctor's opinion.)

![presentation](https://nsa40.casimages.com/img/2021/01/13//210113084647642103.png)

## 0) Requirements
Don't forget to modify the CONFIG.txt file (default configuration works)
```python
!pip install -U efficientnet
from API import *

CFG = get_config()
```

## 1) Extracting tabular features
1. First of all, before extracting features, we need to segment all images from the dataset.
Here is the API call:
```python
# CFG : the config dictionnary
# dataframe: a pandas.DataFrame with at least 1 column: "filename" containing the name(including the extension)
#            for each image, and another column "target"(OPTIONNAL) with the labels "benign" or "malignant"
# images_path: the path where all inputs images are located
# segmentations_path: the path where all outputs segmentations will be saved
get_segmentations(CFG, dataframe, images_path, segmentations_path)
```
2. Finally, we can compute tabular features:
```python
# CFG : the config dictionnary
# dataframe: a pandas.DataFrame with at least 1 column: "filename" containing the name(including the extension)
#            for each image, and another column "target"(OPTIONNAL) with the labels "benign" or "malignant"
# images_path: the path where all inputs images are located
# segmentations_path: the path where all outputs segmentations are located
df = get_tabular_dataframe(CFG, dataframe, images_path, segmentations_path)
```
The returned DataFrame is normalized (MinMax scaler)

Features are calculated with region properties.
I also tried to reproduce the famous *ABCD* rule (Assymetry, Border irregularity & Colors Descriptors).
(In the following tab, A = Area, P = Perimeter, d = minor axis length, D = major axis length. Others variables are explained in references)

Morphological formulas | Assymetry [ref](https://www.researchgate.net/publication/319354997_Classification_of_Benign_and_Malignant_Melanocytic_Lesions_A_CAD_Tool) | Border irregularity [ref](https://www.researchgate.net/publication/319354997_Classification_of_Benign_and_Malignant_Melanocytic_Lesions_A_CAD_Tool) | Colors [ref](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3160648/) |
------------ | ------------ | ------------ | ------------ |
extent | ![equation](https://latex.codecogs.com/svg.latex?min(A_x,%20A_y)/A) | ![equation](https://latex.codecogs.com/png.latex?P%20*%20%281/d%20-%201/D%29) | F4, F5, F6
solidity | ![equation](https://latex.codecogs.com/svg.latex?(A_x%20+%20A_y)/A) | | F10, F11, F12
![equation](https://latex.codecogs.com/svg.latex?d/D) | | | F13, F14, F15
![equation](https://latex.codecogs.com/svg.latex?4A/(\pi%20d^2)) |
![equation](https://latex.codecogs.com/svg.latex?(\pi%20d)/P) |
![equation](https://latex.codecogs.com/svg.latex?(4\pi%20A)/P^2) |
![equation](https://latex.codecogs.com/svg.latex?P/(\pi%20D)) |


## 2) Writing & reading TFRecords
1. Writing a TFRecord
```python
# CFG : the config dictionnary
# df: a pandas.DataFrame containing columns in this order: "filename" containing the name(including the extension)
#            for each image, "target"(OPTIONNAL) with the labels "benign" or "malignant", and all other columns are the features.
#            In our exemple, we will have 21 columns ("filename","target",+19 features)
# images_path: the path where all inputs images are located
# output_path: path where TFRecord files will be stored)
# nb: (OPTIONNAL, default=1) the number of TFRecord files to create
# preprocess_function: (OPTIONNAL) a function to preprocess images. None = no preprocessing. Default: whit-balancing + hair removal.
write_tfrecord(CFG, df, images_path, output_path, preprocess_function)
```
2. Reading a TFRecord
```python
# CFG : the config dictionnary
# tfrecord_train(test): the path containing tfrecord files built with training(test) data
# labeled: if images are labeled (True if training, False if testing)
# augment (OPTIONNAL): if Images should be augmented (only if labeled=True for training)
dataset_train = read_tfrecord(CFG, tfrecord_train, labeled=True)
dataset_test  = read_tfrecord(CFG, tfrecord_test, labeled=False)
# dataset_train(test) (output): a dataset to give to our model.
```
## 3) Create Model
Here we have pre-trained weights for different configurations:
* B0 (net_count=1, fine_tune=False, preprocess_function=None): [Download](https://drive.google.com/file/d/19QyTLsYVqXR54EoKhFpNZ0Jcyzr1IrMa/view?usp=sharing)
* **B0 (net_count=1, fine_tune=True, preprocess_function=None):** [Download](https://drive.google.com/file/d/1HLpf0pGm36w8fGWyTlFUudpq8gw2Hzd7/view?usp=sharing)
* B0-2 (net_count=3, fine_tune=True, preprocess_function=None): [Download](https://drive.google.com/file/d/18u2BoJ2O1UX9QDTuLeHyJIhwwDtJWuLI/view?usp=sharing)
* B0-4 (net_count=5, fine_tune=False, preprocess_function=None): [Download](https://drive.google.com/file/d/19TjZ3EJUzmzClT1KuYUHINM5qsP8Q2PI/view?usp=sharing)

```python
# CFG : the config dictionnary
# fine_tune(OPTIONNAL, default:False): True:Fine-tuning, False:Transfer-learning
# model_weights(OPTIONNAL): the path containing valid weights for the model
model = get_model(CFG, fine_tune=False, model_weights="???")
# model(output): a tensorflow model
```

## 4) Fit & Predict
1. Fit
```python
callbacks = [tf.keras.callbacks.ReduceLROnPlateau(),]
             #tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
             #tf.keras.callbacks.ModelCheckpoint("models/best.h5", save_best_only=True, monitor='val_auc', mode='max', save_weights_only=True)]

history = model.fit(
    dataset_train,
    steps_per_epoch = len(df)/(CFG['batch_size'] * CFG['REPLICAS']),
    epochs = CFG['epochs'],
    callbacks = callbacks,
)
```

2. Predict
```python
preds = model.predict(
        dataset_test,
        steps = len(df)/(CFG['batch_size'] * CFG['REPLICAS']),
)
```


## To-Do List

- [x] Easier way to implement his own preprocessing function (when writing a TFRecord file)
- [x] TPU support (It needs to be tested, not sure at all)
- [ ] Improve the segmentation part to improve features' quality

## License

CC 4.0 Attribution-NonCommercial International

The software is for educational and academic research purposes only.

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Licence Creative Commons" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />Ce(tte) œuvre est mise à disposition selon les termes de la <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Licence Creative Commons Attribution - Pas d’Utilisation Commerciale 4.0 International</a>.
