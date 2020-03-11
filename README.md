# ALPR (차량 번호판 인식)
[<img src="https://j.gifs.com/wVABrX.gif" width="70%">](https://youtu.be/EIZpI8A1Qe0)

##### Clone and install requirements
```
mkdir OCR
git clone https://github.com/clovaai/deep-text-recognition-benchmark.git

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
## Credit
```
eriklindernoren/PyTorch-YOLOv3

tzutalin/labelImg

clovaai/deep-text-recognition-benchmark
```

sudo apt install libdvdnav4 libdvdread4 gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly libdvd-pkg
sudo apt install ubuntu-restricted-extras
