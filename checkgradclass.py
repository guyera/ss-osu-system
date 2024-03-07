import torch

class MulConstant(torch.autograd.Function):
    @staticmethod
    def forward(tensor, constant):
        return tensor 

    @staticmethod
    def setup_context(ctx, inputs, output):
        # ctx is a context object that can be used to stash information
        # for backward computation
        tensor, constant = inputs
        ctx.constant = constant

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        return grad_output * ctx.constant, None



tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
constant = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

result = MulConstant.apply(tensor, constant)
