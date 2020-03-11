# ALPR (차량 번호판 인식)
[<img src="https://j.gifs.com/wVABrX.gif" width="70%">](https://youtu.be/EIZpI8A1Qe0)

##### env
```
| UBUNTU 16.04 | python 3.6 | pytorch 1.4.0 | Opencv-python 4.2 |
```
##### Clone and install requirements
```
$ git clone https://github.com/hhaahaha/PlateRecognition.git
$ cd ALPR/
$ sudo pip3 install -r requirements.txt
```
## Preparation
##### download virtual environment(Optional)

[virt_env](https://drive.google.com/drive/folders/1qiPqo5hqrJK2ls1wVOfOg2Y41MnB3NOC?usp=sharing)
```
/home/pirl/yy 에 폴더 다운로드

yy/bin/ 경로에서 source ./activate 
```
##### download pretrained weights

[weights.zip](https://drive.google.com/file/d/1TVgXuKUXV57BzKNoc4lnE9514QN6Gh7-/view?usp=sharing)


##### download sample data

[baza_slika.zip](https://drive.google.com/file/d/1eTEZuuWt6ZiV22eOJ4NJYmcz914BwDpE/view?usp=sharing)


## Test

```
python3 detect.py --image_folder data/samples/ --weights_path weights/plate.weights 
```
##### Argument parser
```
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/custom.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/plate.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/custom/custom.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=500, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
```
# Training OCR model
## Train OCR model before put in Plate Recognition model
##### Clone and install requirements
```
$ mkdir OCR
$ cd OCR
$ git clone https://github.com/clovaai/deep-text-recognition-benchmark.git
```
### Getting Started
##### Dependency
- This work was tested with PyTorch 1.3.1, CUDA 10.1, python 3.6 and Ubuntu 16.04. <br> You may need `pip3 install torch==1.3.1`. <br>
In the paper, expriments were performed with **PyTorch 0.4.1, CUDA 9.0**.
- requirements : lmdb, pillow, torchvision, nltk, natsort
```
pip3 install lmdb pillow torchvision nltk natsort
```

##### Download lmdb dataset for traininig and evaluation from [here](https://drive.google.com/drive/folders/192UfE9agQUMNq6AgU3_E05_FcPZK4hyt)
data_lmdb_release.zip contains below. <br>
training datasets : [MJSynth (MJ)](http://www.robots.ox.ac.uk/~vgg/data/text/)[1] and [SynthText (ST)](http://www.robots.ox.ac.uk/~vgg/data/scenetext/)[2] \
validation datasets : the union of the training sets [IC13](http://rrc.cvc.uab.es/?ch=2)[3], [IC15](http://rrc.cvc.uab.es/?ch=4)[4], [IIIT](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html)[5], and [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset)[6].\
evaluation datasets : benchmark evaluation datasets, consist of [IIIT](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html)[5], [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset)[6], [IC03](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2003_Robust_Reading_Competitions)[7], [IC13](http://rrc.cvc.uab.es/?ch=2)[3], [IC15](http://rrc.cvc.uab.es/?ch=4)[4], [SVTP](http://openaccess.thecvf.com/content_iccv_2013/papers/Phan_Recognizing_Text_with_2013_ICCV_paper.pdf)[8], and [CUTE](http://cs-chan.com/downloads_CUTE80_dataset.html)[9].

##### Run demo with pretrained model
1. Download pretrained model from [here](https://drive.google.com/drive/folders/15WPsuPJDCzhp2SvYZLRj8mAlT3zmoAMW)
2. Add image files to test into `demo_image/`
3. Run demo.py (add `--sensitive` option if you use case-sensitive model)
```
CUDA_VISIBLE_DEVICES=0 python3 demo.py \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
--image_folder demo_image/ \
--saved_model TPS-ResNet-BiLSTM-Attn.pth
```
##### Arguments
* `--train_data`: folder path to training lmdb dataset.
* `--valid_data`: folder path to validation lmdb dataset.
* `--eval_data`: folder path to evaluation (with test.py) lmdb dataset.
* `--select_data`: select training data. default is MJ-ST, which means MJ and ST used as training data.
* `--batch_ratio`: assign ratio for each selected data in the batch. default is 0.5-0.5, which means 50% of the batch is filled with MJ and the other 50% of the batch is filled ST.
* `--data_filtering_off`: skip [data filtering](https://github.com/clovaai/deep-text-recognition-benchmark/blob/f2c54ae2a4cc787a0f5859e9fdd0e399812c76a3/dataset.py#L126-L146) when creating LmdbDataset. 
* `--Transformation`: select Transformation module [None | TPS].
* `--FeatureExtraction`: select FeatureExtraction module [VGG | RCNN | ResNet].
* `--SequenceModeling`: select SequenceModeling module [None | BiLSTM].
* `--Prediction`: select Prediction module [CTC | Attn].
* `--saved_model`: assign saved model to evaluation.
* `--benchmark_all_eval`: evaluate with 10 evaluation dataset versions, same with Table 1 in our paper.
* `--FT`: fine tunning model with custom dataset.

##### Make Custom Datasets
1. make gt.txt for create 
```
pip3 install fire
python3 make_gt_txt.py --file_name gt.txt --dir_path data/image/
```
2. Create your own lmdb dataset.
```
pip3 install fire
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

### After FT model with custom dataset, put the 'best_model.sh' into ./PlateRecognition/OCR/saved_models

## Credit
```
eriklindernoren/PyTorch-YOLOv3

tzutalin/labelImg

clovaai/deep-text-recognition-benchmark
```

sudo apt install libdvdnav4 libdvdread4 gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly libdvd-pkg
sudo apt install ubuntu-restricted-extras
