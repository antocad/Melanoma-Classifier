# Melanoma Classifier

This project aims to classify a new melanoma picture as malignant or benign. 

To do it, I have created a Deep Learning model (with Tensorflow) taking 2 inputs: 
* Inputs 1: tabular features, previously extracted from each images
* Inputs 2: color pictures resized to (224,224,3)

## 1) Extracting tabular features
1. First of all, before extracting features, we need to segment all images from the dataset.
Here is the API call:
```python
# CFG : the config dictionnary
# dataframe: a pandas.DataFrame whith at least 1 columns: "filename" containing the name(including the extension)
#            for each image, and another column "target" with the labels "benign" or "malignant"
# images_path: the path where all inputs images are located
# segmentations_path: the path where all outputs segmentations will be saved
get_segmentations(CFG, dataframe, images_path, segmentations_path)
```
2. Finally, we can compute tabular features:
```python
# dataframe: a pandas.DataFrame whith at least 1 columns: "filename" containing the name(including the extension)
#            for each image, and another column "target" with the labels "benign" or "malignant"
# images_path: the path where all inputs images are located
# segmentations_path: the path where all outputs segmentations are located
df = get_tabular_dataframe(dataframe, images_path, segmentations_path)
```

Features are calculated with region properties.
I also tried to reproduce the famous *ABCD* rule (Assymetry, Border irregularity & Colors Descriptors)

Morphological formulas | Assymetry [ref](https://www.researchgate.net/publication/319354997_Classification_of_Benign_and_Malignant_Melanocytic_Lesions_A_CAD_Tool) | Border irregularity [ref](https://www.researchgate.net/publication/319354997_Classification_of_Benign_and_Malignant_Melanocytic_Lesions_A_CAD_Tool) | Colors [ref](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3160648/) |
------------ | ------------ | ------------ | ------------ | 
extent | ![equation](https://latex.codecogs.com/svg.latex?min(A_x,%20A_y)/A) | ![equation](https://latex.codecogs.com/svg.latex?P%20*%20(1/d%20-%201/D)) | F4, F5, F6
solidity | ![equation](https://latex.codecogs.com/svg.latex?(A_x%20+%20A_y)/A) | | F10, F11, F12
![equation](https://latex.codecogs.com/svg.latex?d/D) | | | F13, F14, F15
![equation](https://latex.codecogs.com/svg.latex?4A/(\pi%20d^2)) |
![equation](https://latex.codecogs.com/svg.latex?(\pi%20d)/P) |
![equation](https://latex.codecogs.com/svg.latex?(4\pi%20A)/P^2) |
![equation](https://latex.codecogs.com/svg.latex?P/(\pi%20D)) |
