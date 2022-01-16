# ImageNet-S Dataset for Large-scale Unsupervised Semantic Segmentation

The ImageNet-S dataset and toolbox.

[Project page](https://unsupervisedsemanticsegmentation.github.io/) [Paper link](https://arxiv.org/abs/2106.03149)

## Introduction

Powered by the ImageNet dataset, unsupervised learning on large-scale data has made significant advances for classification tasks. There are two major challenges to allowing such an attractive learning modality for segmentation tasks: i) a large-scale benchmark for assessing algorithms is missing; ii) unsupervised shape representation learning is difficult. We propose a new problem of large-scale unsupervised semantic segmentation (LUSS) with a newly created benchmark dataset to track the research progress. Based on the ImageNet dataset, we propose the ImageNet-S dataset with 1.2 million training images and 50k high-quality semantic segmentation annotations for evaluation. Our benchmark has a high data diversity and a clear task objective. We also present a simple yet effective baseline method that works surprisingly well for LUSS. In addition, we benchmark related un/weakly/fully supervised methods accordingly, identifying the challenges and possible directions of LUSS.


## ImageNet-S Dataset Preparation

#### Prepare the ImageNet-S dataset with one command:
The ImageNet-S dataset is based on the ImageNet-1k dataset.
**You need to have a copy of ImageNet-1k dataset**, 
and you can also get the rest of the ImageNet-S dataset (split/annotations) with the following command:
```shell
cd datapreparation
bash data_preparation.sh [your imagenet path] [the path to save ImageNet-S datasets] [split: 50 300 919 all] [whether to copy new images: false, true]
```

#### Get part of the ImageNet-S dataset:
The `data_preparation.sh` command is composed of the following steps, and you can run separate scripts if you only need parts of the ImageNet-S dataset:
- **Extract training datasets:**
To extract the training set from the existing ImageNet dataset, run:

```shell
    python datapreparation_train.py \
      --imagenet-dir [your imagenet path] \
      --save-dir [the path to save ImageNet-S datasets] \
      --mode [split: 50 300 919 all]
```
You can set the mode to `50`, `300`, and `919` to extract the ImageNet-S-50, ImageNet-S-300, and ImageNet-S datasets. To extract all datasets one time, set mode to `all`. The script uses soft links to create datasets by default. To copy new images, please add `--copy`.

- **Extract validation and test datasets:**
To extract the validation and test datasets from the existing ImageNet dataset, run:

```shell
    python datapreparation_val.py \
      --imagenet-dir [your imagenet path] \
      --save-dir [the path to save ImageNet-S datasets] \
      --mode [split: 50 300 919 all]
```
You can set mode to `50`, `300`, and `919` to extract the ImageNet-S-50, ImageNet-S-300 and ImageNet-S datasets. To extract all datasets one time, set mode to `all`. The script copy new images of validation and test set simultaneously.

- **Download semantic segmentation annotations:**
```shell
bash datapreparation_anno.sh [the path to save ImageNet-S datasets] [split: 50 300 919 all]
```


### Evaluation
Before evaluation, note that given the I-th category, the value of pixels that belong to the I-th category should be set to (I % 256, I / 256, 0) with the order of RGB.
```shell
    cd evaluation
```

##### Matching
We provide a default matching algorithm, run:
```shell
    python match.py \ 
      --predict-dir [the path to validation ground truth] \
      --gt-dir [the path to validation prediction] \
      --mode [split: 50 300 919] \
      --workers [the number of workers for dataloader] \
      --session-name [the file name of saved matching]
```
The matching will be saved under the directory of results.
##### Evaluation
With the segmentation results of your algorithm, you can evaluate the quality of your results by running:
```shell
    python evaluator.py \ 
      --predict-dir [the path to validation ground truth] \
      --gt-dir [the path to validation prediction] \
      --mode [split: 50 300 919] \
      --workers [the number of workers for dataloader]
```
You can also use a matching between your prediction and ground truth for evaluation by adding --match [the file name of saved matching]. 
However, the script defaults that the prediction has been matched to the ground truth.


### Citation
```
@article{gao2021luss,
  title={Large-scale Unsupervised Semantic Segmentation},
  author={Gao, Shanghua and Li, Zhong-Yu and Yang, Ming-Hsuan and Cheng, Ming-Ming and Han, Junwei and Torr, Philip},
  journal={arXiv preprint arXiv:2106.03149},
  year={2021}
}
```