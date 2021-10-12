import torch

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.linear = torch.nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, x):
        out = self.linear(x)

        return out

