from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from detect import detection, determine, getFrame
from OCR.ocr import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, default="data/image/", help='path to image_folder which contains text images')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4) # n_cpu= workers
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument("--img_size", type=int, default=800, help="size of each image dimension")
    parser.add_argument("--video", type=str, help="using video format")
    """YOLO setting"""
    parser.add_argument("--model_def", type=str, default="config/custom.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/plate.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/custom/custom.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=str, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=str, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=str, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation") # n_cpu= workers
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--saved_model', type=str, default="OCR/saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111/best_accuracy.pth",  help="path to saved_model to evaluation")
    parser.add_argument('--Detection', type=str, default="Darknet", help='Detect plate stage. None|Darknet')
    parser.add_argument('--Transformation', type=str, default="TPS",  help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default="ResNet", help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, default="BiLSTM", help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, default="Attn",  help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print('opt\n',opt)

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()


    # run model
    determine(opt) # yolo 모델
    OCR(opt) # ocr 모델
