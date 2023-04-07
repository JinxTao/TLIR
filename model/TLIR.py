import model as Model
import torch.nn as nn
import os
import numpy as np
from model.ri3_modules.DSEM import data_self_enhancement_module
from .ri3_modules import unet
import torch
from collections import OrderedDict



class TLIR(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.device = torch.device(
            'cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.begin_step = 0
        self.begin_epoch = 0
        self.diffusion = Model.create_model(opt)
        # self.diffusion.
        print('subdiff finish')
        self.iterNum = opt['TLIR']['iterNum']
        self.R = opt['TLIR']['R']
        model_opt = opt['model']
        self.finaModel = unet.UNet(
            in_channel=model_opt['unet']['in_channel'],
            out_channel=model_opt['unet']['out_channel'],
            norm_groups=model_opt['unet']['norm_groups'],
            inner_channel=model_opt['unet']['inner_channel'],
            channel_mults=model_opt['unet']['channel_multiplier'],
            attn_res=model_opt['unet']['attn_res'],
            res_blocks=model_opt['unet']['res_blocks'],
            dropout=model_opt['unet']['dropout'],
            image_size=model_opt['diffusion']['image_size']
        )
        print('subU finish')
        namdaarr = np.full(self.iterNum,0.5)
        self.namda = torch.nn.Parameter(torch.FloatTensor(namdaarr), requires_grad=True).to(self.device)
        self.log_dict = OrderedDict()

    def set_device(self, x):
        if isinstance(x, dict):
            for key, item in x.items():
                if item is not None:
                    x[key] = item.to(self.device)
        elif isinstance(x, list):
            for item in x:
                if item is not None:
                    item = item.to(self.device)
        else:
            x = x.to(self.device)
        return x
    
    def forward(self, x, sino):
        x_ = x
        for i in range(self.iterNum):
            self.diffusion.feed_data(x_)
            self.diffusion.test(continous=False)
            visuals = self.diffusion.get_current_visuals()
            x_ = data_self_enhancement_module(visuals['RI'], sino, out_iter=i, namda=self.namda[i], R=self.R)
        y = self.finaModel(x_,x_)

        return y


    