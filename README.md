# Visual_Recognition_HW2

## Introduction
The project is training an accurate object detection neteork using Faster R-CNN for [SVHN](http://ufldl.stanford.edu/housenumbers/).

## Usage
We training and testing with Python 3.6, pytorch 1.4 and **Need** to reference [timm](https://github.com/rwightman/pytorch-image-models), [AutoAugment](https://github.com/DeepVoltaire/AutoAugment) and [DataAugmentationForObjectDetection](https://github.com/Paperspace/DataAugmentationForObjectDetection).

### Traning and Testing model
First, this Faster R-CNN applies Efficient-Net-b4 as a backbone to extract features.\
If you want to use this network for training, you must generate a compliant dataset.

Before generating the data, Upload  the training images to `/data/train` and the test data to `/data/train`.\
Make sure you have the mat file(default name: **digitStruct.mat**) in the `/data/train` folder.

#### Generate Dataset
Example:

```
python generate_dataset.py

```

***important parameters in config.py***

Default:

| Argument    | Default value |
| ------------|:-------------:|
| annotation_file         | model/SVHN_annotation.pkl        |

 

  


When the program was finished, we will get a file `SVHN_annotation.pkl` in `/model/`.
Now we have one pkl file .

```
./models/SVHN_annotation.pkl  
```

### Traning

We supply two ways to use Ensemble Learning as well. Need to put trained models in folder `/models/`.

Example:

```
python ensemble.py -m resnet50_1_model -m resnet50_2_model 


Required arguments:
--model -m 		Select the model you want to use ensemble learning in folder /models/.
```

OR

```
from ensemble import ensemble_learning

models = ["resnet50_1_model","resnet50_2_model"]
ensemble_learning(models)
```

When the program was finished, we will get a csv file `/result/`.

```
./result/voting.csv
```
## Result

| Model Name                    | Testing Performance (on Kaggle) |
| ------------------------------|:-------------------------------:|
| Resnet50                      | 0.90040                         |
| Densen201                     | 0.91320                         |
| Resnext50_32x4d               | 0.92140                         |
| Inceptionresnet_v2            | 0.92460                         |
| Resnext101_32x8d              | 0.92820                         |
| Efficienct_net_b4             | 0.93680                         |
| Top 3 model (ensmble learning)| 0.94240                         |
| Top 5 model (ensmble learning)| 0.94640                         |
| Top 7 model (ensmble learning)| 0.94900                         |



