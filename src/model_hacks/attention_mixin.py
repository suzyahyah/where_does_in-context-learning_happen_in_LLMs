#!/usr/bin/python3
# Author: Suzanna Sia

import torch

class AttentionMaskMixin:
    def __init__(self):
        self.mask_prev_positions = False
        self.mask_till = None
        self.mask_from = None

    def _modify_mask(self, attention_mask):
        if self.mask_prev_positions:
            if attention_mask is None:
                raise Exception("Not implemented")
            for item in range(len(self.mask_till)):
                start = self.mask_from[item]
                end = self.mask_till[item]
                if start is None or end is None:
                    continue

                attention_mask[item, :, :, start:end] = torch.finfo(attention_mask.dtype).min
        return attention_mask
