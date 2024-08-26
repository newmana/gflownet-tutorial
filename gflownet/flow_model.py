import torch.nn as torch_nn


class FlowModel(torch_nn.Module):
    def __init__(self, num_hid):
        super().__init__()
        # We encoded the current state as binary vector, for each patch the associated
        # dimension is either 0 or 1 depending on the absence or precense of that patch.
        # Therefore, the input dimension is 6 for the 6 patches.
        self.mlp = torch_nn.Sequential(torch_nn.Linear(6, num_hid), torch_nn.LeakyReLU(),
                                       # We also output 6 numbers, since there are up to
                                       # 6 possible actions (and thus child states), but we
                                       # will mask those outputs for patches that are
                                       # already there.
                                       torch_nn.Linear(num_hid, 6))

    def forward(self, x):
        # We take the exponential to get positive numbers, since flows must be positive,
        # and multiply by (1 - x) to give 0 flow to actions we know we can't take
        # (in this case, x[i] is 1 if a feature is already there, so we know we
        # can't add it again).
        f = self.mlp(x).exp() * (1 - x)
        return f
