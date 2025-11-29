# SSCLNet-Implementation Report

Access the paper from [here](https://doi.org/10.1109/ACCESS.2023.3237542).

## Steps involved in implementation:
Following are the steps involved in the implementation of this paper:
1.  Dataset Preparation
2.  Dataset Preprocessing
3.  Training the model (available in the `implementation.ipynb` file)
4.  Results we got
5.  Comparison and conclusion

## About the Datasets
* MRI 4 Class Dataset: Available [here](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)
* MRI 2 Class Dataset: Available [here](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
* MRI 4 Class Dataset with bounding boxes: Available [here](https://www.kaggle.com/datasets/ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes/data)
* Glioblastoma Multiforme (GBM) 1 Class Dataset: Available [here](https://data.mendeley.com/datasets/mxsb7snyvx/1)
</br>
* <a href="https://drive.google.com/drive/folders/17iNx6mt5FTt3cxwrsUvVoEhyODNX0gyi" target="_blank">Dataset folder</a>
 which has all the data from raw to its preprocessed zip file.

### 1. Data Preparation
1. Deleted the `labels` folder from the `MRI 4 Class Dataset with bounding boxes` as it was not needed for our paper implementation.
2. Merged the two `4 Class Datasets` and moved the `Glioblastoma Multiforme` tumor images to the `glioma` tumor folder, since glioblastoma is a sub-class of glioma.
3. Deduplicated the complete dataset (including 2-class data and 4-class data) using MD5 Hashing. Work is shown in the `dataset-preparation.ipynb` file.

The following two steps were performed using the `dataset.py` module.

### Data Splitting Strategy
* **Pretrain**: 70% of the complete dataset was randomly sampled from both 2-class and 4-class folders, then placed in a flat directory structure without labels for self-supervised contrastive learning.
* **Train**: 20% of the complete dataset was randomly selected from the remaining 30% unseen data, maintaining the original hierarchical folder structure with proper class labels for supervised fine-tuning.
* **Test**: 10% of the complete dataset was taken from the final portion of unseen data, preserving the folder hierarchy with accurate tumor type classifications for final model evaluation.

### 2. Data Preprocessing
The complete dataset underwent the following preprocessing pipeline (as specified in the base paper):

1. **Grayscale Conversion**: All MRI images were converted to single-channel grayscale to standardize color space and reduce computational complexity.
2. **Background Removal**: Image backgrounds were eliminated using Otsu's thresholding combined with contour detection, isolating the brain region for focused analysis.
3. **Contrast Enhancement**: Histogram equalization (CLAHE) was applied to improve image contrast, followed by median filtering to reduce noise while preserving edges.
4. **Standardized Resizing**: All images were resized to 224×224 pixels to ensure consistent input dimensions for deep learning models and optimize GPU memory usage.

### 3. Implementation of the paper
Model architecture for `contrastive learning`
```mermaid
flowchart TD
classDef wraptext text-wrap:wrap,white-space:pre-line;
    A[Image i] --LFG Block--> B(augmented image i_a)
    A[Image i] --LFG BLock--> C(augmented image i_b)
    B --> D(σ - ResNet50)
    C --> E(σ - ResNet50)
    D --> F["label feature Li(a)"]
    E --> G["label feature Li(b)"]
    F --ILCL Block--> H(α - 4 layer nonlinear MLP)
    G --ILCL Block--> I(α - 4 layer nonlinear MLP)
    H --> J["z_i(a)"]
    I --> K["z_i(b)"]
    J --> L["maximize similarity between z_i(a) and z_i(b)<br>minimize similarity between \{z_i(a), z_i(b)\} against all other images"]
    K --> L["maximize similarity between z_i(a) and z_i(b)<br>minimize similarity between \{z_i(a), z_i(b)\} against all other images"]
    class L wraptext;
```
Model architecture for `supervised learning`
* We first get label-features against all images whose actual label we know:
```mermaid
flowchart TD
A[Image i] --"trained-σ(.)-SSCLNet"--> B("Label-feature-Li")
```
* Then we use the images’ label-features (Li’s) and their labels to train a seven-layer dense neural network that uses categorical cross-entropy loss to learn accurate mappings between features and their respective classes.

### 4. Results we got
* **Contrastive Pre-training Loss** <br>
<img width="500" height="500" alt="contrastive_loss (1)" src="https://github.com/user-attachments/assets/1497ec8d-58b5-4d5e-8924-e56d84dc65f0" /><br>
* **Class Distribution For Supervised Learning**<br>
<img width="500" height="500" alt="class_distribution" src="https://github.com/user-attachments/assets/e8b8585f-44e8-4d92-8b1c-6161dde56d4d" /><br>
* **Supervised Training Loss and Accuracy**<br>
<img width="4470" height="1523" alt="supervised_training_history_cv" src="https://github.com/user-attachments/assets/826bc122-b97a-453b-a14c-a8c0240f0994" /><br>
* **Confusion Matrix**<br>
<img width="500" height="500" alt="confusion_matrix" src="https://github.com/user-attachments/assets/c06988b7-fa29-4b9e-9d6a-ef87375f71b7" /><br>
* **Multi-Class ROC Curve**<br>
<img width="500" height="500" alt="roc_curve" src="https://github.com/user-attachments/assets/90d761d9-248f-4a97-b420-ad7c54666218" /><br>





