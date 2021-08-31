# PSPNet-logits and feature-distillation
This repository is based on PSPNet and modified from Pixelwise_Knowledge_Distillation_PSPNet18(https://github.com/ChexuanQiao/Pixelwise_Knowledge_Distillation_PSPNet18), which uses a logits knowledge distillation method to teach the PSPNet model of ResNet18 backbone with the PSPNet model of ResNet50 backbone. All the models are trained and tested on the PASCAL-VOC2012 dataset. 


Please note that you should train a teacher model( PSPNet model of ResNet50 backbone) at first, and save the checkpoints or just use a well trained PSPNet50 model, which you can refer to the oringinal public code at https://github.com/hszhao/semseg, and all the enviromental requirements in this repo are the same as it.



 
