import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from glob import glob
import albumentations as A
import cv2
import numpy as np
import tkinter as tk

class CFG:
    seed          = 101
    debug         = False # set debug=False for Full Training
    exp_name      = 'Baseline'
    comment       = 'unet-efficientnet_b1-224x224'
    model_name    = 'Unet'
    backbone      = 'efficientnet-b1'
    train_bs      = 64
    valid_bs      = train_bs*2
    img_size      = [224, 224]
    epochs        = 15
    lr            = 2e-3
    scheduler     = 'CosineAnnealingLR'
    min_lr        = 1e-6
    T_max         = int(30000/train_bs*epochs)+50
    T_0           = 25
    warmup_epochs = 0
    wd            = 1e-6
    n_accumulate  = max(1, 32//train_bs)
    n_fold        = 5
    num_classes   = 3
    device        = torch.device("cpu")
    thr           = 0.45
    ttas          = [0]

class SegmentationModel:
    def __init__(self, path=""):
        if path:
            self.model = self.load_model(path)
        else:
            self.model = self.build_model()
        self.transform = A.Compose([
        A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST),
        ], p=1.0)

    def build_model(self):
        model = smp.Unet(
            encoder_name=CFG.backbone,
            encoder_weights=None,
            in_channels=3,
            classes=CFG.num_classes,
            activation=None,
        )
        return model

    def load_model(self, path):
        model = self.build_model()
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        model.eval()
        return model

    def load_img(self, id):
        img = cv2.imread("input/" + str(id) + "_in.png", cv2.IMREAD_UNCHANGED)
        img = np.tile(img[...,None], [1, 1, 3]) # gray to rgb
        img = img.astype('float32') # original is uint16
        mx = np.max(img)
        if mx:
            img/=mx # scale image to [0, 1]
        return img

    def preprocess(self, id):
        img = self.load_img(id)
        h, w = img.shape[:2]
        data = self.transform(image=img)
        img = data['image']
        img_nchw = np.transpose(img, (2, 0, 1))
        return torch.tensor(img_nchw), img, id, h, w

    def inference(self, id):
        img, img_nhwc, id, h, w = self.preprocess(id)
        size = img.size()
        img = img.view(1, size[0], size[1], size[2])
        out = self.model(img)
        msk = nn.Sigmoid()(out)
        msk = (msk.permute((0,2,3,1)) > CFG.thr).to(torch.uint8).numpy()

        # print(img_nhwc)
        out = np.add(msk[0]*0.6, img_nhwc)*255

        # TODO: convert to rles for training
        return out