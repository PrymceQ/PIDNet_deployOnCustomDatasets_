import torch
import argparse
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import _init_paths
import glob
import os
import time
import numpy as np
import models

def parse_args():
    parser = argparse.ArgumentParser(description='Custom Input')

    parser.add_argument('--a', help='pidnet-s, pidnet-m or pidnet-l', default='pidnet-s', type=str)
    parser.add_argument('--c', help='class_num', type=int, default=2)
    parser.add_argument('--p', help='dir for pretrained model', default='../output/cityscapes/pidnet_small_cityscapes/checkpoint.pth.tar')   # ../pretrained_models/cityscapes/PIDNet_S_Cityscapes_val.pt', type=str)
    parser.add_argument('--input-size', help='root or dir for input images', default=(1, 3, 640, 640), type=tuple)

    args = parser.parse_args()
    return args

def load_pretrained(model, pretrained):
    pretrained_dict = torch.load(pretrained, map_location='cpu')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
    msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
    print('Attention!!!')
    print(msg)
    print('Over!!!')
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict = False)

    return model

def print_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    print('Number of params: %.2fM' % (total / 1e6))


if __name__ == '__main__':
    args = parse_args()
    device = torch.device("cpu") # cuda:0

    model = models.pidnet.get_pred_model(args.a, args.c)
    model = load_pretrained(model, args.p).to(device)
    model.eval()

    tensor = torch.rand(args.input_size)
    tensor = tensor.to(device)

    # FLOPs
    flops = FlopCountAnalysis(model, tensor)
    print('FLOPs: %.2fG' % (flops.total() / 1e9))

    # params
    print_model_parm_nums(model)
    
    # time
    # torch.cuda.synchronize()
    start = time.time()
    # add cv2.imread + resize
    result = model(tensor)
    # torch.cuda.synchronize()
    end = time.time()
    print('infer_time: %.2fms' % ((end-start)*1000) )
