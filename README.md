# PSPNet-logits and feature-distillation

## Introduction
This repository is based on PSPNet and modified from [Pixelwise_Knowledge_Distillation_PSPNet18](https://github.com/ChexuanQiao/Pixelwise_Knowledge_Distillation_PSPNet18), which uses a logits knowledge distillation method to teach the PSPNet model of ResNet18 backbone with the PSPNet model of ResNet50 backbone. All the models are trained and tested on the PASCAL-VOC2012 dataset(Enhanced Version). 




## Innovation and Limitations
This repo adds a feature distillation in the aux layer of PSPNet without a linear feature mapping since the teacher and student model's output dimension after the aux layer is the same. On the other hand, if you want to adapt this repo to other structures, a mapping should be needed. Also, the output of the aux layer is very close to which of the final layer, so you should pay attention to the overfitting problem. Or you can distillate the features in earlier layers and add a mapping, of course, just like [Fitnet](https://arxiv.org/abs/1412.6550).

## For reimplementation
Please download related datasets and symlink the relevant paths. The temperature parameter(T) and corresponding weights can be changed flexibly. All the numbers showed in the name of python code indicate the number of layers; for instance, train_50_18.py represents the distillation of 50 layers to 18 layers.

Please note that you should train a teacher model( PSPNet model of ResNet50 backbone) at first, and save the checkpoints or just use a well trained PSPNet50 model, which you can refer to the original public code at [semseg](https://github.com/hszhao/semseg), and you should download the initial models and corresponding lists in semseg and put them in right paths, also all the environmental requirements in this repo are the same as semseg.
## Usage
1. Requirement: PyTorch>=1.1.0, Python3, tensorboardX, GPU
2. Clone the repository:
```
git clone https://github.com/asaander719/PSPNet-knowledge-distillation.git
```
3. Download initialization models and lists, also trained models and predictions can be optional, by the link shows in [semseg](https://github.com/hszhao/semseg), and put them in files followed by instructions.
4. Download official dataset PASCAL-VOC2012, please note that it is **Enhanced Version**,and put them in corresponding paths follwed by data lists.
5. Train and test a teacher model: adjust parameters in config(voc2012_pspnet50.yaml), like layers. etc.., and the checkpoints will be saved automaticly, or you can just download a trained model, and put it in a right path.  
```
python train_50.py
```
```
python test_50.py
```
6. Train and test a student model(optional, only for comparison): adjust parameters in config(voc2012_pspnet18.yaml), like layers. etc.., and the checkpoints will be saved automaticly, or you can just download a trained model, and put it in a right path.  
```
python train_18.py
```
```
python test_18.py
```
7. Distillation and Test: the results should between the teacher and the student model.
```
python train_50_18.py
```
```
python test_50_18.py
```
