from hqq.core.quantize import HQQLinear
import torch
import torch.nn as nn

class _HQQ():
    def __init__(self, quant_config):
        self.quant_config = quant_config
    
    
    def linear(self,in_features, out_features, bias=False):
        return HQQLinear(
            nn.Linear(in_features, out_features, bias),
            quant_config=self.quant_config,
            del_orig=True,
            compute_dtype=torch.float16,
            initialize=True
        )