#!/usr/bin/python3
# Author: Suzanna Sia

from transformers import BloomForCausalLM, BloomModel 
from transformers.models.bloom.modeling_bloom import BloomBlock, BloomAttention
from torch import nn

from src.model_hacks.attention_mixin import AttentionMaskMixin

class BloomForCausalLMHack(BloomForCausalLM):
    """Inheritance hack to manipulate attention layer"""

    def __init__(self, config):
        super().__init__(config)
        self.transformer = BloomModelHack(config)
        self.post_init()

class BloomModelHack(BloomModel):
    """Inheritance hack to manipulate attention layer"""

    def __init__(self, config):
        super().__init__(config)
        self.h = nn.ModuleList([BloomBlockHack(config, layer_idx=i) for i in
            range(config.num_hidden_layers)])
        self.post_init()

class BloomBlockHack(BloomBlock):
    """Inheritance hack to manipulate attention layer"""
    
    def __init__(self, config,  layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)
        self.self_attention = BloomAttentionHack(config, layer_idx)


class BloomAttentionHack(BloomAttention, AttentionMaskMixin):
    """Inheritance hack with AttentionMaskMixin to manipulate attention layer"""

    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)
        AttentionMaskMixin.__init__(self)
        self.layer_idx = layer_idx 

    def forward(self, *args, **kwargs):
        kwargs['attention_mask'] = self._modify_mask(kwargs['attention_mask'])
        return super().forward(*args, **kwargs)
