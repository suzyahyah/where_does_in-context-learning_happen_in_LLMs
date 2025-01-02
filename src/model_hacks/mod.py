#!/usr/bin/python3
# Author: Suzanna Sia

### Standard imports
import torch
import torch.nn as nn 
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

def init_trainable_mask(model, args_model):
    num_heads = model.transformer.h[0].attn.attention.num_heads
    num_layers = len(model.transformer.h)

    for layer in range(num_layers):
        if args_model.layer_mask.mask_init == "random":
            r0, r1 = 0, 2
            var = (r0 - r1) * torch.rand(1, num_heads, 1, 1) + r1
        params = nn.Parameter(var)
        if not DEVICE.type == "cpu":
            params = params.cuda()

        model.transformer.h[layer].attn.attention.head_mask = params 

    return model

