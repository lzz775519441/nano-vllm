import torch
from torch import nn

from nanovllm.cuda_ops import sample


class Sampler(nn.Module):

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        return sample(logits, temperatures)
