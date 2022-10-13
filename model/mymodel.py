#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : model
# @Date : 2022-10-13
# @Project : pytorch_basic
# @Author : seungmin

import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


# candidate_model 폴더 안에 있는 후보 모델 파일 import
from .candidate_models import *
 

class MyModel(nn.Module):
    # 실험하고 싶은 model 종류가 여러가지 일 때, 유용한 클래스.
    # dictionary 를 구현하지 않고, class 별로 파일을 따로 만들어도 상관없음.
    def __init__(self, base_model):
        super(MyModel, self).__init__()
        self.model_dict = {
            'convnext_small' : models.convnext_small(pretrained=True),
            'convnext_base' : models.convnext_base(pretrained=True),
            'coca' : coca,
            'resnet_50' : models.resnet50(pretrained=True),
        }

        mymodel = self._get_basemodel(base_model)
        mymodel = self.freeze_layer(mymodel) 
        self.features = nn.Sequential(*list(mymodel.children())[:-1])  # get all layers except the last layer
        '''
        https://github.com/mortezamg63/Accessing-and-modifying-different-layers-of-a-pretrained-model-in-pytorch/blob/master/README.md
        To deal with layers of a pretrained model
        To get some layers which we want
        '''
        self.fc = nn.LazyLinear(out_features=10, bias=True)  # num of classes = 10
        # nn.init.kaiming_normal(self.fc.weight)

    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
            print("Feature Extractor: ", model_name)  # convoultional base
            return model
        except:
            raise ("Invalid model name. Check the config file.")
        
    def freeze_some_layers(self, model):
        ct=0
        num_layers = len(list(model.children()))
        for child in model.childern():
            ct +=1
            if ct < (num_layers*.7):         # lower layers (layers의 70%)는 업데이트되지 않도록 freeze
                for p in child.parameters():
                    p.required_grad = False
        return model

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x
