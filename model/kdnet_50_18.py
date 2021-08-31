import torch
import torch.nn as nn
import torch.nn.functional as F
from model.pspnet import PSPNet, teacher_loader
class KDNet(nn.Module):
    def __init__(self, tag='teacher', layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=21, zoom_factor=8, temperature = 1.9, alpha = 0.2, use_ppm=True, criterion=nn.CrossEntropyLoss(ignore_index=255), pretrained=True):
        super(KDNet, self).__init__()
        assert layers in [18, 50]
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.criterion = criterion
        self.temperature = temperature
        self.alpha = alpha
        self.student_net = PSPNet(tag='student', layers=18, bins=(1, 2, 3, 6), dropout=0.1, classes=21, zoom_factor=8, use_ppm=True, criterion=nn.CrossEntropyLoss(ignore_index=255), pretrained=True)
        self.teacher_loader = teacher_loader()

    def distillation(self, student, aux, teacher, truth, temperature=1.9, alpha=0.2):
        criterion = nn.CrossEntropyLoss(ignore_index=255)
        p = F.log_softmax(student/temperature, dim = 1)
        q = F.softmax(teacher/temperature, dim = 1)
        kl_loss = F.kl_div(p, q, reduction = 'mean') * (temperature**2) *19
        ce_loss = F.cross_entropy(student, truth,ignore_index=255)
        aux_loss = self.criterion(aux, truth)
        main_loss = kl_loss * (1 - alpha) + ce_loss * alpha
        return student, main_loss, aux_loss, kl_loss, ce_loss
    
    def set_bn_eval(self, m):
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval() 
    
    def forward(self, img, gTruth=None):
        if self.training:
            student_img, aux_img = self.student_net(img)
            teacher_net = self.teacher_loader
            teacher_net.apply(self.set_bn_eval)
            teacher_img = teacher_net(img)
            assert student_img.shape == teacher_img.shape
            student_output, main_loss, aux_loss, kl_loss, ce_loss = self.distillation(student_img, aux_img, teacher_img, gTruth, self.temperature, self.alpha)
            return student_output.max(1)[1], main_loss, aux_loss, kl_loss, ce_loss
        else:
            student_img = self.student_net(img)
            return student_img
