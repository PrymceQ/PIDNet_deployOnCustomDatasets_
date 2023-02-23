# PIDNet
Thanks to the PIDNet 
Paper: [PIDNet](https://arxiv.org/pdf/2206.02066v2.pdf)
Code: [PIDNet](https://github.com/XuJiacong/PIDNet)

  <img align="center" src="figs/tqdm.png" alt="overview-of-our-method" width="100%"/>

How to use PIDNet for customed datasets.
# What changed?
在官方的版本上，我们做了如下改变
1. 将训练的GPU数量调整成一个，并且处理了一些Bug
2. FLOPS_demo.py
3. 做了一个攻略，方便快速上手


# How to train PIDNet on custom dataset?
> Here, we use pidnet-small to make an example. You need to change these settings one by one:
1. You need to prepare cityscapes-style image data in [data] folder as following:
    - data/cityscapes/custom_datasets/
        - gtFine
        - images
2. Make a [.lst] file for the trainloader to read images as following:
    - data/cityscapes/list/cityscapes/
        - train.lst
        - val.lst  (if you have val data, or you can use a copy of train.lst and rename it to val.lst)
        > custom_datasets/images/1.jpg custom_datasets/gtFine/1.png

3. Datasets
    - datasets/cityscapes.py
        - self.label_mapping
        - self.class_weights # len(self.class_weights) = len(self.label_mapping)

4. Set necessary configs in [.yaml] file, which is in configs/cityscapes/pidnet_small_cityscapes.yaml
    - set DATASET.TEST_SET, DATASET.TRAIN_SET and DATASET.NUM_CLASSES
    - set TRAIN.IMAGE_SIZE, TRAIN.BASE_SIZE, TEST.IMAGE_SIZE, TEST.BASE_SIZE
    - set TRAIN.BATCH_SIZE_PER_GPU, TRAIN.END_EPOCH if you like

5. Train
    - train.py
        - cfg
6. Your output will be saved in
    - output/cityscapes/pidnet_small_cityscapes/

# How to inference?
By the default setting
you need to put your images in [samples] for inference

use inference.py
- c : class_num  = your model setting

color_map>=the class number of your dataset is okay

the output will be saved to [samples/output/]