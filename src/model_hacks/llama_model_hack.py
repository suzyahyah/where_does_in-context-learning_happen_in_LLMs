#!/usr/bin/python3
# Author: Suzanna Sia

from transformers import LlamaForCausalLM, LlamaModel
from transformers.models.llama.modeling_llama import LlamaDecoderLayer,  LlamaAttention
from torch import nn

from src.model_hacks.attention_mixin import AttentionMaskMixin

class LlamaForCausalLMHack(LlamaForCausalLM):
    """Inheritance hack to manipulate attention layer"""

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModelHack(config)
        self.post_init()


class LlamaModelHack(LlamaModel):
    """Inheritance hack to manipulate attention layer"""

    def __init__(self, config): 
        super().__init__(config)
        self.layers = nn.ModuleList([LlamaDecoderLayerHack(config, layer_idx=i)\
                for i in range(config.num_hidden_layers)])

class LlamaDecoderLayerHack(LlamaDecoderLayer):
    """Inheritance hack to manipulate attention layer"""

    def __init__(self, config, layer_idx=0):
        super().__init__(config, layer_idx=layer_idx)
        self.self_attn = LlamaAttentionHack(config=config, layer_idx=layer_idx)

class LlamaAttentionHack(LlamaAttention, AttentionMaskMixin):
    """Inheritance hack with AttentionMaskMixin to manipulate attention layer"""

    def __init__(self, config=None, layer_idx=0):
        super().__init__(config=config, layer_idx=layer_idx)
        AttentionMaskMixin.__init__(self)

    def forward(self, *args, **kwargs):
        kwargs['attention_mask'] = self._modify_mask(kwargs['attention_mask'])
        return super().forward(*args, **kwargs)


