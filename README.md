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

### Data Preprocessing:
1. Deleted the `labels` folder from the `MRI 4 Class Dataset with bounding boxes` as it was not needed for our paper implementation.
2. Merged the two `4 Class Datasets` and added a fifth class of `Glioblastoma Multiforme` tumor, making the resultant dataset to be a `5-class` dataset.
3. Deduplicated the complete dataset (including 2-class data and 5-class data) using MD5 Hashing. Work is shown in the `dataset-preparation.ipynb` file.

### Data Splitting Strategy

* **Pretrain**: 70% of the complete dataset was randomly sampled from both 2-class and 5-class folders, then placed in a flat directory structure without labels for self-supervised contrastive learning.
* **Train**: 20% of the complete dataset was randomly selected from the remaining 30% unseen data, maintaining the original hierarchical folder structure with proper class labels for supervised fine-tuning.
* **Test**: 10% of the complete dataset was taken from the final portion of unseen data, preserving the folder hierarchy with accurate tumor type classifications for final model evaluation.
   
