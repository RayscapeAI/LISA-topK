"""
Official code for "Learning to Generalize towards Unseen Domains via
a Content-Aware Style Invariant Model for Disease Detection from Chest X-rays"

https://github.com/rafizunaed/domain_agnostic_content_aware_style_invariant

Version: 1.0 (26 February, 2023)
Programmed by Mohammad Zunaed
"""

import torch
from torch import nn
import torch.nn.functional as F

from utils.cls_densenet_ibn import densenet121_ibn_a


class DenseNet121_IBN_proposed(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        # get densenet architecture
        model = densenet121_ibn_a(pretrained=True)                   
        modules = list(model.children())[0]

        # stem
        self.stem = nn.Sequential(*modules[:4])

        # dense block 1
        self.db1 = nn.Sequential(*modules[4:5])
        self.trn1 = nn.Sequential(*list(modules[5])[:-1])

        # dense block 2
        self.db2 = nn.Sequential(*modules[6:7])
        self.trn2 = nn.Sequential(*list(modules[7])[:-1])

        # dense block 3
        self.db3 = nn.Sequential(*modules[8:9])
        self.trn3 = nn.Sequential(*list(modules[9])[:-1])

        # dense block 4
        self.db4 = nn.Sequential(*modules[10:])

        # avg layer for applying after dense block-1,2,3
        self.avg = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        # # final fc
        self.classifier = nn.Linear(1024, num_classes)
        self.classifier.weight.data.normal_(0, 0.01)
        self.classifier.bias.data.zero_()

        self.srm = SRM()
        
    def get_cls_loss(self, logits, targets):
        criterion_bce = nn.BCEWithLogitsLoss()             
        cls_loss = criterion_bce(logits, targets)
        return cls_loss

    def forward(self, x: torch.Tensor, targets=None, y: torch.Tensor=None):
        x = self.stem(x)

        x = self.db1(x)
        x = self.trn1(x)
        x = self.avg(x)

        x = self.db2(x)
        x = self.trn2(x)
        x = self.avg(x)

        # level 2
        if self.training:
            x = self.srm(x)

        x = self.db3(x)
        x = self.trn3(x)
        x = self.avg(x)
                
        x = self.db4(x)
        xg = F.relu(x)

        xg_pool = F.adaptive_avg_pool2d(xg, (1,1)).flatten(1)
        logits = self.classifier(xg_pool)
        cls_loss = self.get_cls_loss(logits, targets)

        if self.training:
            y = self.stem(y)

            y = self.db1(y)
            y = self.trn1(y)
            y = self.avg(y)

            y = self.db2(y)
            y = self.trn2(y)
            y = self.avg(y)

            y = self.db3(y)
            y = self.trn3(y)
            y = self.avg(y)

            y = self.db4(y)
            yg = F.relu(y)

            gm_x = generate_gram_matrix(xg)
            gm_y = generate_gram_matrix(yg)
            consistency_loss = F.mse_loss(xg, yg)+F.mse_loss(gm_x, gm_y) 

            gfe_loss = get_gfe_loss(xg)
 
            return {
                'logits': logits,
                'gfm': xg,
                'gfm_pool': xg_pool,
                'cls_loss': cls_loss,
                'consistency_loss': consistency_loss,
                'gfe_loss': gfe_loss,
                'loss': gfe_loss.mean() + consistency_loss.mean() + cls_loss.mean()
                } 
        else:
            return {
                'logits': logits,
                'gfm': xg,
                'gfm_pool': xg_pool,
                'loss': cls_loss
                }

def generate_gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features = F.normalize(features, dim=2, eps=1e-7)
    features_t = features.transpose(1, 2)
    gram_matrix = features.bmm(features_t)
    return gram_matrix

def get_gfe_loss(x):
    gfm = x.clone()
    criterion_mse = torch.nn.MSELoss()
    gm = generate_gram_matrix(gfm)  
    scores = torch.diagonal(gm, offset=0, dim1=-2, dim2=-1)
    gt = torch.ones_like(scores)
    gfe_loss = criterion_mse(scores, gt)
    return gfe_loss

class SRM(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-7

    def forward(self, x: torch.Tensor):
        N, C, H, W = x.size()

        # normalize
        x = x.view(N, C, -1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)      
        x = (x - mean) / (var + self.eps).sqrt()

        # swap styles
        idx_swap = torch.arange(N).flip(0)
        mean = mean[idx_swap]
        var = var[idx_swap]

        x = x * (var + self.eps).sqrt() + mean
        x = x.view(N, C, H, W)

        return x