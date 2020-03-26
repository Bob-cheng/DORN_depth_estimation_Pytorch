# DORN_depth_estimation_Pytorch

This is an unoficial Pytorch implementation of [Deep Ordinal Regression Network for Monocular Depth Estimation](http://arxiv.org/abs/1806.02446) paper by Fu et. al.

The implementation closely follows the paper and the [official repo](https://github.com/hufu6371/DORN). The list of known differences:
 - Only training on the labeled part of NYU V2 is currently implemented (not the whole raw data and Eigen's split as in the paper).
 - ColorJitter is used instead of the color transformation from the Eigen paper.
 - Feature extractor is pretrained on a different dataset.

## Requirements

 - Python 3
 - Pytorch (version 1.3 tested)
 - Torchvision
 - Tensorboard

## How to use

To prepare data:
 - Download data and [nyu_depth_v2_labeled.mat](http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat) and [splits.mat](http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat).
 - Edit create_nyu_h5.py to add data_path and output_path.
 - Run:
  ```bash
  python create_nyu_h5.py
  ```

For start training on NYU V2 run:
  ```bash
  train.py [-h] [--dataset DATASET] [--data-path DATA_PATH]
                [--pretrained] [--epochs EPOCHS] [--bs BS] [--bs-test BS_TEST]
                [--lr LR] [--gpu GPU]
  ```
Or simply:
  ```bash
  python train.py --data-path DATA_PATH --pretrained
  ```
(where DATA_PATH is same as output_path used during preparing data).

For more info on arguments run:
  ```bash
  python train.py --help
  ```

To train on a different dataset, implementation of the DataLoader is required.

To monitor training, use Tensorboard:
  ```bash
  tensorboard --logdir ./logs/
  ```


## Pretrained feature extractor

DORN uses a modified version of ResNet-101 as a feature extractor (with dilations and three 3x3 convolutional layers in the begining instead of one 7x7 layer). If you select pretrained=True, weights pretrained on MIT ADE20K dataset will be loaded from [this project](https://github.com/CSAILVision/semantic-segmentation-pytorch). This is different from the paper (the authors suggest pretraining on ImageNet). That is the only suitable pretrained model on the Web that I am aware of.

## Acknowledgements

The code is based on [this implementation](https://github.com/dontLoveBugs/DORN_pytorch).