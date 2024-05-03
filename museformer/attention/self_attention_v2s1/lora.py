import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class LoRALinear(nn.Linear):

    def __init__(self,
                 # nn.Linear parameters
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 device=None,
                 dtype=None,
                 # LoRA parameters
                 lora_rank: int = 0,
                 lora_alpha: float = 0.0,
                 lora_dropout: float = 0.0,
                ) -> None:
        
        # Initialize the inherited class, nn.linear 
        super(LoRALinear, self).__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)

        self.has_weights_merged = False
        if lora_rank > 0:
            self.lora_dropout = nn.Dropout(lora_dropout)
            self.lora_scaling = lora_alpha / lora_rank

            self.lora_A = torch.nn.Parameter(torch.empty(lora_rank, in_features), requires_grad=True)
            self.lora_B = torch.nn.Parameter(torch.empty(out_features, lora_rank), requires_grad=True)

            self.lora_A.requires_grad = False
            self.lora_B.requires_grad = False

            self.reset_parameters()

    def is_lora(self) -> bool:
        return hasattr(self, 'lora_A')

    def reset_parameters(self) -> None:
        nn.Linear.reset_parameters(self)
        if self.is_lora():
            torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            torch.nn.init.zeros_(self.lora_B)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.is_lora() and not self.has_weights_merged:
            BA = self.lora_scaling * self.lora_B @ self.lora_A
            res = F.linear(input, self.weight, self.bias)
            res += F.linear(self.lora_dropout(input), BA)
            return res
        else:
            return F.linear(input, self.weight, self.bias)

    def train(self, mode: bool = True) -> "LoRALinear":
        if self.is_lora():
            if self.has_weights_merged:
                self.weights.data -= self.lora_B @ self.lora_A
            self.has_weights_merged = False
        return self

    def eval(self) -> "LoRALinear":
        if self.is_lora():
            if not self.has_weights_merged:
                self.weights.data += self.lora_B @ self.lora_A
            self.has_weights_merged = True
        return self
    
    def extra_repr(self) -> str:
        out = nn.Linear.extra_repr(self)
        if self.is_lora():
            out += f', lora_rank={self.lora_A.shape[0]}, lora_scaling={self.lora_scaling}, lora_dropout={self.lora_dropout.p}'
        return out

def mark_only_lora_as_trainable(model: nn.Module) -> nn.Module:
    print("(LoRA) Freezing Weights...")
    frozen_count = 0
    unfrozen_count = 0
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            unfrozen_count += param.numel()
            param.requires_grad = True
        else:
            frozen_count += param.numel()
            param.requires_grad = False
    print("(LoRA) Total Frozen Parameters: {}".format(frozen_count))
    print("(LoRA) Total Unfrozen Parameters: {} ({})".format(unfrozen_count, unfrozen_count/(frozen_count + unfrozen_count)))
    print("(LoRA) Total Parameters: {}",format(frozen_count + unfrozen_count))
    return model

