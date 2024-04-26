import os
import torch
import torch.nn as nn
import torchvision
import copy

from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules.heads import BYOLProjectionHead, BYOLPredictionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum

class BYOL(nn.Module):
    def __init__(self, backbone, num_ftrs, proj_hidden_dim, pred_hidden_dim, out_dim):
        super().__init__()

        self.backbone = backbone.float()
        self.projection_head = BYOLProjectionHead(num_ftrs, proj_hidden_dim, out_dim).float()
        self.prediction_head = BYOLPredictionHead(out_dim, pred_hidden_dim, out_dim).float()
        #torch.save(BYOLProjectionHead(num_ftrs, proj_hidden_dim, out_dim), 'ModelLoader/BYOLProjectionHead')
        #torch.save(BYOLPredictionHead(out_dim, pred_hidden_dim, out_dim), 'ModelLoader/BYOLPredictionHead')

        self.backbone_momentum = copy.deepcopy(self.backbone).float()
        self.projection_head_momentum = copy.deepcopy(self.projection_head).float()

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return p

    def forward_momentum(self, x):
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z