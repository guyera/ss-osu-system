# This software was created by Oregon State University under Army Research
# Office (ARO) Award Number W911NF-22-2-0149. ARO, as the Federal awarding
# agency, reserves a royalty-free, nonexclusive and irrevocable right to
# reproduce, publish, or otherwise use this software for Federal purposes, and
# to authorize others to do so in accordance with 2 CFR 200.315(b).

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
