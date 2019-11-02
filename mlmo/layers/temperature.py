from torch.nn import Module


class Temperature(Module):
    def __init__(self, alpha=1.):
        super(Temperature, self).__init__()
        self.alpha = alpha

    def forward(self, logits):
        return logits / self.alpha
