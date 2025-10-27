# SSCLNet-Implementation

Access the paper from [here](https://doi.org/10.1109/ACCESS.2023.3237542).

## Dataset
* MRI 4 Class Dataset: Available [here](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)
* MRI 2 Class Dataset: Available [here](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
* MRI 4 Class Dataset with bounding boxes: Available [here](https://www.kaggle.com/datasets/ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes/data)
* Glioblastoma Multiforme (GBM) 1 Class Dataset: Available [here](https://data.mendeley.com/datasets/mxsb7snyvx/1)
</br>
* <a href="https://drive.google.com/drive/folders/17iNx6mt5FTt3cxwrsUvVoEhyODNX0gyi" target="_blank">Dataset folder</a>
 which has all the data from raw to its preprocessed zip file.

### Data Preparation
1. Deleted the `labels` folder from the `MRI 4 Class Dataset with bounding boxes` as it was not needed for our paper implementation.
2. Merged the two `4 Class Datasets` and added a fifth class of `Glioblastoma Multiforme` tumor, making the resultant dataset to be a `5-class` dataset.
3. Deduplicated the complete dataset (including 2-class data and 5-class data) using MD5 Hashing. Work is shown in the `dataset-preparation.ipynb` file.

### Data Splitting Strategy
* **Pretrain**: 70% of the complete dataset was randomly sampled from both 2-class and 5-class folders, then placed in a flat directory structure without labels for self-supervised contrastive learning.
* **Train**: 20% of the complete dataset was randomly selected from the remaining 30% unseen data, maintaining the original hierarchical folder structure with proper class labels for supervised fine-tuning.
* **Test**: 10% of the complete dataset was taken from the final portion of unseen data, preserving the folder hierarchy with accurate tumor type classifications for final model evaluation.

### Data Preprocessing
The complete dataset underwent the following preprocessing pipeline (as specified in the base paper):

1. **Grayscale Conversion**: All MRI images were converted to single-channel grayscale to standardize color space and reduce computational complexity.
2. **Background Removal**: Image backgrounds were eliminated using Otsu's thresholding combined with contour detection, isolating the brain region for focused analysis.
3. **Contrast Enhancement**: Histogram equalization (CLAHE) was applied to improve image contrast, followed by median filtering to reduce noise while preserving edges.
4. **Standardized Resizing**: All images were resized to 224×224 pixels to ensure consistent input dimensions for deep learning models and optimize GPU memory usage.
