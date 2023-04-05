# ImageNet-S Dataset for Large-scale Unsupervised/Semi-supervised Semantic Segmentation

The ImageNet-S dataset and toolbox.

[Project page](https://LUSSeg.github.io/) [Paper link](https://arxiv.org/abs/2106.03149) [PaperWithCode Leaderboard](https://paperswithcode.com/dataset/imagenet-s)

![image](https://user-images.githubusercontent.com/20515144/149651945-94501ffc-78c0-41be-a1d9-b3bfb3253370.png)



# Introduction

Powered by the ImageNet dataset, unsupervised learning on large-scale data has made significant advances for classification tasks. There are two major challenges to allowing such an attractive learning modality for segmentation tasks: i) a large-scale benchmark for assessing algorithms is missing; ii) unsupervised shape representation learning is difficult. We propose a new problem of large-scale unsupervised semantic segmentation (LUSS) with a newly created benchmark dataset to track the research progress. Based on the ImageNet dataset, we propose the ImageNet-S dataset with 1.2 million training images and 50k high-quality semantic segmentation annotations for evaluation. Our benchmark has a high data diversity and a clear task objective. We also present a simple yet effective baseline method that works surprisingly well for LUSS. In addition, we benchmark related un/weakly/fully supervised methods accordingly, identifying the challenges and possible directions of LUSS.

# News
- 2022.11.24. The semantic segmentation on the MMSegmentation codebase is released, better performance is observed thanks to the MMSegmentation [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/imagenets).
- 2022.10.18. The code of baseline method (PASS) for unsupervised semantic segmentation on the ImageNet-S dataset is released on [PASS](https://github.com/LUSSeg/PASS).
- 2022.9.21. The code of semi-supervised semantic segmentation on the ImageNet-S dataset is released on [ImageNetSegModel](https://github.com/LUSSeg/ImageNetSegModel).

# Apps and Sourcecode
- Unsupervised semantic segmentation: [PASS](https://github.com/LUSSeg/PASS)
- Semi-supervised semantic segmentation: [ImageNetSegModel](https://github.com/LUSSeg/ImageNetSegModel) [MMSegmentation](https://github.com/LUSSeg/mmsegmentation/tree/imagenets/configs/imagenets)


# ImageNet-S Dataset Preparation

### Prepare the ImageNet-S dataset with one command:
The ImageNet-S dataset is based on the ImageNet-1k dataset.
**You need to have a copy of ImageNet-1k dataset**, 
and you can also get the rest of the ImageNet-S dataset (split/annotations) with the following command:
```shell
cd datapreparation
bash data_preparation.sh [your imagenet path] [the path to save ImageNet-S datasets] [split: 50 300 919 all] [whether to copy new images: false, true]
```

### Get part of the ImageNet-S dataset:
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

# Dataset Information

### Structure
```
├── imagenet-s
    ├── ImageNetS919
        ├── train-full  # full imagenet-1k training set with 1000 classes.
        ├── train       # imagenet-s-919 training set with 919 classes.
        ├── train-semi  # imagenet-s-919 training images with pixel-level annotations (10 images for each class)
        ├── train-semi-segmentation # semantic segmentation mask of the train-semi images.
        ├── validation  # imagenet-s-919 validation set.
        ├── validation-segmentation # semantic segmentation mask of the validation images.
        └── test        # magenet-s-919 test set, the segmentation mask is stored on the online evalution server.
    ├── ImageNetS300        
        ├── train       # imagenet-s-300 training set with 300 classes.
        ├── train-semi  # imagenet-s-300 training images with pixel-level annotations (10 images for each class)
        ├── train-semi-segmentation # semantic segmentation mask of the train-semi images.
        ├── validation  # imagenet-s-300 validation set.
        ├── validation-segmentation # semantic segmentation mask of the validation images.
        └── test        # magenet-s-300 test set, the segmentation mask is stored on the online evalution server.                     
    └── ImageNetS50                            
        ├── train     # imagenet-s-50 training set with 50 classes.
        ├── train-semi  # imagenet-s-50 training images with pixel-level annotations (10 images for each class)
        ├── train-semi-segmentation # semantic segmentation mask of the train-semi images.
        ├── validation  # imagenet-s-50 validation set.
        ├── validation-segmentation # semantic segmentation mask of the validation images.
        └── test        # magenet-s-50 test set, the segmentation mask is stored on the online evalution server.   
```

### Image Numbers
The ImageNet-S dataset contains 1183322 training, 12419 validation, and 27423 testing images from 919 categories. We annotate 39842 val/test images and 9190 training images with precise pixel-level masks.

| Dataset | category | train   | val   | test  |
|------------------|----------|---------|-------|-------|
| ImageNet-S_{50}  | 50       | 64431   | 752   | 1682  |
| ImageNet-S_{300} | 300      | 384862  | 4097  | 9088  |
| ImageNet-S        | 919      | 1183322 | 12419 | 27423 |

### Q&A
**How to get the class id from the segmentation mask images?**

The image annotation (eg. an image in validation-segmentation) is stored in the png form with RGB channels,
you can get the class id by **R+G*256**.  
The `ignored part` is annotated as **1000**, and the `other category` is annotated as **0**.

```
segmentation = Image.open(path) # RGB
segmentation_id = segmentation[:, :, 1] * 256 + segmentation[:, :, 0] # R+G*256
```


**How to match the class id in ImageNet-S with the ImageNet tag id?**

The imagenet-s class id is obtained by sorting the imagenet tag id:
```
with open('ImageNetS_categories_im919.txt') as f:
    msg = f.read().splitlines()

msg = sorted(msg) # sort tags to get the imagenet-s class id.
msg = '\n'.join(msg)

with open('ImageNetS_categories_im919_sort.txt', 'w') as f:
    f.write(msg)
```
We provide a matching table between imagenet-s class id and imagenet tag id as follows (The i-th row is the class id of imagenet-s (start from 1)):
[imagenet-s-919](https://github.com/LUSSeg/ImageNet-S/blob/main/data/categories/ImageNetS_categories_im919.txt)
[imagenet-s-300](https://github.com/LUSSeg/ImageNet-S/blob/main/data/categories/ImageNetS_categories_im300.txt)
[imagenet-s-50](https://github.com/LUSSeg/ImageNet-S/blob/main/data/categories/ImageNetS_categories_im50.txt)

Note: for imagenet-s-919, we merge some categories as follows:
```
# key is merged into the value, eg. merge n04356056 into n04355933.
merge = {'n04356056': 'n04355933',
         'n04493381': 'n02808440',
         'n03642806': 'n03832673',
         'n04008634': 'n03773504',
         'n03887697': 'n15075141'}
```

**If you have any other question, open an issue or email us via shgao@live.com**


# Evaluation
Before evaluation, note that given the I-th category, the value of pixels that belong to the I-th category should be set to (I % 256, I / 256, 0) with the order of RGB.
```shell
    cd evaluation
```

### Matching
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
### Evaluation on val set
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

# Online benchmark
Due to the lack of ground-truth (GT) category labels during training, 
LUSS models cannot be directly evaluated like in the supervised setting.
We present three evaluation protocols for LUSS, including the fully unsupervised evaluation, 
semi-supervised evaluation, and distance matching evaluation.
To explore the upper bound of ImageNet-S semantic segmentation, 
we also present a free evaluation benchmark with no limitations.

* Fully unsupervised protocol [link](https://codalab.lisn.upsaclay.fr/competitions/1317)
* Distance matching protocol [link](https://codalab.lisn.upsaclay.fr/competitions/1315)
* Semi-supervised protocol [link](https://codalab.lisn.upsaclay.fr/competitions/1318)
* Free protocol [link](https://codalab.lisn.upsaclay.fr/competitions/1316)

### Submission rules

The submission file has the following structure:

```
├── submission.zip
    ├── n...                              // prediction
    ├── n...                              // prediction
    ├── n...                              // prediction
    ├── match.json                        // optional
    └── method.txt                        // description of method
```

**[submission example](https://github.com/LUSSeg/ImageNet-S/releases/download/Example/submission-imagenets50-random-example.zip)**

You must submit your results to the corresponding protocols, and **miss-matched submissions will be deleted**.
We summarize a table for different protocols:

|      Actions          | Fully unsupervised | Distance matching | Semi-supervised | Free |
|:---------------------:|--------------------|-------------------|-----------------|------|
| ImageNet-S~{50/300/full} only^{note1}   |     ✓                |     ✓   |     ✓          |      |
| Only unsupervised pre-training   |                    |        ✓            |                 |      |
| Label generation and fine-tuning         |        ✓             |                    |                 |      |
| Fine-tune with 1% training image annotation   |                    |                   |       ✓            |      |
| Supervised pre-trained weights? |                    |                   |                 | ✓    |
| Extra training data?  |                    |                   |                 |    ✓   |
| Supervised edge/saliency?  |                    |                   |                 |    ✓   |

Note1: Pre-training on the ImageNet-S~{full} and fine-tuning on the ImageNet~{300/50} is not allowed.

### Fully unsupervised protocol [link](https://codalab.lisn.upsaclay.fr/competitions/1317)

The fully unsupervised evaluation protocol requires no human-annotated labels during training and only needs the validation/test set for evaluation. Unlike the supervised tasks, categories are generated by the model in the LUSS task, which needs to match with GT categories during evaluation. 
We present the default image-level matching scheme in the [ImageNet-S toolbox](https://github.com/UnsupervisedSemanticSegmentation/ImageNet-S), 
while an effective matching scheme should improve LUSS evaluation performance.
You need to match the generated categories with GT categories, and assign matched categories to the test images.

### Distance matching protocol [link](https://codalab.lisn.upsaclay.fr/competitions/1315)

In distance matching evaluation protocol, 
we directly get the embeddings of GT categories with
the pixel-level labeled training images
and match them with embeddings in the validation/testing set to assign labels.
You don't need to care about the label generation in LUSS and only need to provide an unsupervised pre-trained model.
The inference code for distance matching is in the  [ImageNet-S toolbox](https://github.com/UnsupervisedSemanticSegmentation/ImageNet-S).   

### Semi-supervised protocol [link](https://codalab.lisn.upsaclay.fr/competitions/1318)

We can conduct semi-supervised fine-tuning to evaluate LUSS models
as we annotate about 1% of training images with pixel-level labels.
The semi-supervised evaluation protocol requires fine-tuning the trained LUSS models
with the 1% pixel-level human-labeled training images.
Therefore, this protocol does not need matching generated and GT category.
Also, this protocol is suitable for real-world applications where
a small part of images are human-labeled and many images are unlabeled.

### Free protocol [link](https://codalab.lisn.upsaclay.fr/competitions/1316)

In this protocol, you can do whatever you want to improve the semantic segmentation performance on ImageNet-S, 
e.g. ImageNet-21K supervised pretraining, image-level annotations, and pixel-level annotations.
The only rule is donot use image-level or pixel-level annotations of val/test sets.


# Citation
```
@article{gao2022luss,
  title={Large-scale Unsupervised Semantic Segmentation},
  author={Gao, Shanghua and Li, Zhong-Yu and Yang, Ming-Hsuan and Cheng, Ming-Ming and Han, Junwei and Torr, Philip},
  journal=TPAMI,
  year={2022}
}
```
