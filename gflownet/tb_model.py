import torch.nn as torch_nn
import torch as torch

class TBModel(torch_nn.Module):
  def __init__(self, num_hid):
    super().__init__()
    # The input dimension is 6 for the 6 patches.
    self.mlp = torch_nn.Sequential(torch_nn.Linear(6, num_hid), torch_nn.LeakyReLU(),
                             # We now output 12 numbers, 6 for P_F and 6 for P_B
                             torch_nn.Linear(num_hid, 12))
    # log Z is just a single number
    self.logZ = torch_nn.Parameter(torch.ones(1))

  def forward(self, x):
    logits = self.mlp(x)
    # Slice the logits, and mask invalid actions (since we're predicting
    # log-values), we use -100 since exp(-100) is tiny, but we don't want -inf)
    p_f = logits[..., :6] * (1 - x) + x * -100
    p_b = logits[..., 6:] * x + (1 - x) * -100
    return p_f, p_b