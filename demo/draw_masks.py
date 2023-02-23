import cv2
import os
from PIL import Image
import matplotlib.image as mpimg
import numpy as np
from tqdm import tqdm

def draw_mask(image, mask, color, thresh):
    image = np.require(image, dtype='f4', requirements=['O', 'W'])
    image_c = image.copy()
    image[:,:,:][mask[:,:] > thresh] = color
    out_image = 0.5 * image + 0.5 * image_c

    return out_image

if __name__ == '__main__':
    # draw gt and pre masks on images
    main_dir = './'
    gt_src = f'{main_dir}/gtFine/'   # png -> gtFine
    img_src = f'{main_dir}/images/'   # jpg -> RBG images
    pre_src = f'{main_dir}/PID_pre/'  # jpg -> pred by PID
    dst = os.path.join(main_dir, dst)
    if not os.path.exists(dst):
        os.mkdir(dst)

    img_name_l = [i.split('.')[0] for i in os.listdir(img_src)]

    for img in tqdm(img_name_l):
        im = mpimg.imread(os.path.join(img_src, f'{img}.jpg'), cv2.IMREAD_COLOR)
        mask_gt = cv2.imread(os.path.join(gt_src, f'{img}.png'), cv2.IMREAD_GRAYSCALE)
        mask_pre = cv2.imread(os.path.join(pre_src, f'{img}.jpg'), cv2.IMREAD_GRAYSCALE)
        im = im[:,:,::-1]

        col_gt = np.array([0.1, 0.1, 0.8])*225  # red
        col_pre= np.array([0.8, 0.1, 0.1])*225  # blue

        im = draw_mask(im, mask=mask_gt, color=col_gt, thresh=100)
        im = draw_mask(im, mask=mask_pre, color=col_pre, thresh=100)

        cv2.imwrite(os.path.join(dst, f'{img}.jpg'), im)