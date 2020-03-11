from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from OCR.ocr import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def detection(opt):


    os.makedirs("./output/", exist_ok=True)
    os.makedirs("./output/images/", exist_ok=True)

    # Set up model
    YOLO = Darknet(opt.model_def, img_size=opt.img_size).to(device)
    print(opt.model_def)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        YOLO.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        YOLO.load_state_dict(torch.load(opt.weights_path))
    print(opt.weights_path)
    YOLO.eval()  # Set in evaluation mode

    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index
    #print("img_det", img_detections)
    print('-' * 80)
    print("Performing object detection:")
    prev_time = time.time()
    time_stemps = []
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = YOLO(input_imgs)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        time_stemps.append(inference_time)
        prev_time = current_time
        #print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    print('-' * 80)
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
        name_ = Path(path).stem
        msg_ = f'[{str(img_i):>3s}] {name_:^12s}|  {path:^25s}  |'
        print(msg_)


        # Create plot
        img = np.array(Image.open(path))
    

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # print("DETECTIONS",detections)
            global left, top, right, bottom
            global image
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2])

            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)

            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                # detecting vehicle < 4, detecting LP > 4
                left, top, right, bottom = None, None, None, None
                if int(cls_pred) > 4:
                    left = float(x1)
                    top = float(y1)
                    right = float(x2)
                    bottom = float(y2)
                    #print("[ BOUNDING BOX ] : x1: ", round(float(x1), 3), " y1: ", round(float(y1), 3), " x2: ", round(float(x2), 3), " y2: ", round(float(y2), 3))

        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        plt.close()
        filename = path.split("/")[-1].split(".")[0]
        # plt.savefig(f"output/{filename}_bbox.jpg", bbox_inches="tight", pad_inches=0.0)
        im = Image.open(f"data/image/{filename}.jpg")
        if left != None or top != None or right != None or bottom != None:
            im1 = im.crop((left, top, right, bottom))
            im1.save(f"output/images/{filename}.jpg")

        else:
            pass
        im.close()



def getFrame(sec,timeStamp,vidcap):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames, image = vidcap.read()

    if hasFrames:
        cv2.imshow('Input Video', image)
        cv2.imwrite("data/image/"+timeStamp+".jpg", image)
        cv2.waitKey(1)
    return hasFrames


def determine(opt):
    if opt.video is None:
        detection(opt)
    else:
        vidcap = cv2.VideoCapture(opt.video)
        sec = 0
        frameRate = 1  # //it will capture image in each 0.5 second
        count = 1
        timeStamp = str(datetime.timedelta(seconds=sec))
        cv2.namedWindow("Input Video", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Input Video", 650, 500)
        success = getFrame(sec, timeStamp,vidcap)

        while success:
            count = count + 1
            sec = sec + frameRate
            sec = round(sec, 2)
            timeStamp = str(datetime.timedelta(seconds=sec))
            print("Converting Video... ",timeStamp)
            success = getFrame(sec, timeStamp, vidcap)
        vidcap.release()
        cv2.destroyAllWindows()
        detection(opt)
