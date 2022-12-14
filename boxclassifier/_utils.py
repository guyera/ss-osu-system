def _state_dict(module):
    return {k: v.cpu() for k, v in module.state_dict().items()}

def _load_state_dict(module, state_dict):
    next_param = next(module.parameters())
    if next_param.is_cuda:
        cuda_state_dict = {k: v.to(next_param.device) for k, v in state_dict.items()}
        module.load_state_dict(cuda_state_dict)
    else:
        module.load_state_dict(state_dict)
