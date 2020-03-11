# <font color="red"> ALPR(차량번호판인식) </font>
## Introduction
 Automatic License Plate Recognition based on Deep learning. 
 - Work on both image and video format
 - Models were trained by datasets which contains Eng/Number. 
   (Unfortunately, it doesn't recognize the Korean yet)
 - Used YOLOv3 for license plate detection
 - OCR based on CNN,RNN,TPS and Attn|CTC
 
## Demo
[<img src="https://j.gifs.com/wVABrX.gif" width="70%">](https://youtu.be/EIZpI8A1Qe0)



## Preparation

#### Clone and install requirements
```
$ git clone https://github.com/Cha-Euy-Sung/ALPR
$ cd ALPR/
$ sudo pip3 install -r requirements.txt
```
#### env
```
| UBUNTU 16.04 | python 3.6 | pytorch 1.4.0 | Opencv-python 4.2 | CUDA 10.1 |
```
#### download virtual environment(Optional)

- [virt_env](https://drive.google.com/drive/folders/1qiPqo5hqrJK2ls1wVOfOg2Y41MnB3NOC?usp=sharing)

download 'yy.zip' at /home/pirl/yy 
```
$ cd yy
$ cd bin
$ source ./activate 
```
#### download pretrained weights

- [weights.zip](https://drive.google.com/file/d/1TVgXuKUXV57BzKNoc4lnE9514QN6Gh7-/view?usp=sharing)


#### sample data set used for training

- [baza_slika.zip](https://drive.google.com/file/d/1eTEZuuWt6ZiV22eOJ4NJYmcz914BwDpE/view?usp=sharing)



## <font color="red"> Training OCR model </font>
### Train OCR model before put in Plate Recognition model
#### Clone and install requirements
```
$ mkdir OCR
$ cd OCR
$ git clone https://github.com/clovaai/deep-text-recognition-benchmark.git
```
- If you want to know how to train OCR model yourself, please check follow link: [click](https://github.com/clovaai/deep-text-recognition-benchmark)
```diff
-CAUTION: DO NOT OVERWRITE *.py files from above link, it may not work properly on our projects.
+RECOMMENDATION : USE ONLY *.sh which is your custom trained model.
```

#### Make Custom Datasets
1. make gt.txt for create 
```
pip3 install fire
python3 make_gt_txt.py --file_name gt.txt --dir_path data/image/
```
2. Create your own lmdb dataset.
```
python3 create_lmdb_dataset.py --inputPath data/ --gtFile data/gt.txt --outputPath result/
```
At this time, `gt.txt` should be `{imagepath}\t{label}\n` <br>
For example
```
test/word_1.png Tiredness
test/word_2.png kills
test/word_3.png A
...
```
3. Modify `--select_data`, `--batch_ratio`, and `opt.character`, see [this issue](https://github.com/clovaai/deep-text-recognition-benchmark/issues/85).

#### After FT model with custom dataset
put the 'best_model.sh' into './PlateRecognition/OCR/saved_models'


## Test

```
python3 main.py 
```
#### Argument parser

|  -- |  <font color="blue">type | default | help </font>|
|:-----:|:-----:|:------:|:-----:|
|image_folder|str | /data/image/| path to image_folder which contains text images|
|batch_size|int|192|input batch size|
|img_size|int|800|size of each image dimension|
|video|str||only need when using video format|
|model_def|str|/config/custom.cfg|path to model definition file|
|weights_path|str|/weights/plate.weights|path to weights file|
|class_path|str|/data/custom/custom.names|path to class label file|
|conf_thres|str|0.8|object confidence threshold|
|n_cpu|int|8|number of cpu threads to use during batch generation|
|saved_model|str|/OCR/saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111/best_accuracy.pth|path to saved_model to evaluation|
|Detection|str|Darknet|Detect plate stage. 'None' or 'Darknet'|
|Transformation|str|TPS|Transformation stage. 'None' or 'TPS'|
|FeatureExtraction|str|ResNet|FeatureExtraction stage. 'VGG'or'RCNN'or'ResNet'|
|Prediction|str|Attn|Prediction stage. 'CTC'or'Attn'|




## Cooperation

Thanks to Euysung @Brandon for implementing License Plate Detection, and working on this project together. 

## Credit
```
eriklindernoren/PyTorch-YOLOv3
tzutalin/labelImg
clovaai/deep-text-recognition-benchmark
```

