# PIDNet
Thanks to the PIDNet !:kissing_heart::kissing_heart::kissing_heart:

[PIDNet Paper](https://arxiv.org/pdf/2206.02066v2.pdf) | 
[PIDNet Code](https://github.com/XuJiacong/PIDNet)


# :yellow_heart:What's new?
In the official version, we have made the following changes:
1. Adjusted the number of GPUs for training to one, and handled some Bugs!!
2. Add <tools/FLOPS_demo.py>, which is easy for getting Flops and params!!
3. Add some useful demos!!!
4. Made a cheat sheet to facilitate a quick start!!!


# :blue_heart:How to train PIDNet for a customed datasets.

> Here, we use pidnet-small to make an example. You need to change these settings one by one:

1. You need to prepare cityscapes-style image data in [data] folder as following:
    - data/cityscapes/custom_datasets/
        - gtFine    # grayscale images, e.g. containing only 0 and 255
        - images    # RBG images or some else
2. Make a [.lst] file for the trainloader to read images as following:
    - data/cityscapes/list/cityscapes/
        - train.lst   # like 'custom_datasets/images/1.jpg custom_datasets/gtFine/1.png', one for images, one for gtFine
        - val.lst     # if you don't have val data, or you can use a copy of train.lst and rename it to val.lst)

3. Change dataset setting as following, and make sure len(self.class_weights) = len(self.label_mapping):
    - datasets/cityscapes.py
        - self.label_mapping
        - self.class_weights

4. Set necessary configs in [.yaml] file, which is in:
    - configs/cityscapes/pidnet_small_cityscapes.yaml
        - set DATASET.TEST_SET, DATASET.TRAIN_SET and DATASET.NUM_CLASSES
        - set TRAIN.IMAGE_SIZE, TRAIN.BASE_SIZE, TEST.IMAGE_SIZE, TEST.BASE_SIZE
        - set TRAIN.BATCH_SIZE_PER_GPU, TRAIN.END_EPOCH if you like

5. Train
  ````bash
    python tools/train.py --cfg configs/cityscapes/pidnet_small_cityscapes.yaml
  ````
  
6. Your output will be saved in:
    - output/cityscapes/pidnet_small_cityscapes/

# :green_heart:How to inference?
1. Inference

By the default setting, you need to put your images in <samples> for inference.

> len(color_map in [inference.py]) >= the class number of your dataset is okay.

  ````bash
    cd tools
    python inference.py --c [your_dataset_class_num] --p ../output/cityscapes/pidnet_small_cityscapes/checkpoint.pth.tar
  ````
  
2. Where's the output?

The output will be saved to <samples/output/>
