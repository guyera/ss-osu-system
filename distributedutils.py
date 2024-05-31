# This software was created by Oregon State University under Army Research
# Office (ARO) Award Number W911NF-22-2-0149. ARO, as the Federal awarding
# agency, reserves a royalty-free, nonexclusive and irrevocable right to
# reproduce, publish, or otherwise use this software for Federal purposes, and
# to authorize others to do so in accordance with 2 CFR 200.315(b).

import torch.distributed

# Generates a wrapper that broadcasts arguments via
# torch.distributed.broadcast_object_list, and then calls the specified function
# with those arguments. This should only be called by
# the broadcasting process (the process's rank should equal the argument
# specified for `src`, as forwarded to torch.distributed.broadcast_object_list).
# [args, kwargs] will be broadcasted, and so they must follow the requirements
# specified by torch.distributed.broadcast_object_list (including backend-
# specific requirements)
def gen_broadcast_call(f, src=0, group=None, device=None):
    def broadcast_call(*args, **kwargs):
        torch.distributed.broadcast_object_list(
            [args, kwargs],
            src=src,
            group=group,
            device=device
        )
        return f(*args, **kwargs)
    return broadcast_call

# Receives arguments via torch.distributed.broadcast_object_list, and then calls
# the specified function with those arguments. This should only be called by
# the receiving processes (the process's rank should NOT equal the argument
# specified for `src`, as forwarded to torch.distributed.broadcast_object_list).
# [args, kwargs] will be received and forwarded to the function call,
# f(*args, **kwargs). If args or kwargs are broadcasted as None, then the
# function call will be skipped.
# Returns:
# 1. The return value of the call to f given the received args and kwargs.
#    If the function call is skipped due to broadcasted Nonetype arguments,
#    then this value will be None.
# 2. A boolean specifying whether f was called (True) or skipped (False).
def receive_call(f, src=0, group=None, device=None):
    obj_list = [None, None] # args, kwargs
    torch.distributed.broadcast_object_list(
        obj_list,
        src=src,
        group=group,
        device=device
    )
    args = obj_list[0]
    kwargs = obj_list[1]
    if args is not None and kwargs is not None:
        # Received non-none arguments and keyword arguments. Call f and return.
        return f(*args, **kwargs), True
    else:
        # Received non-type arguments or keyword arguments. Skip the function
        # call and return False.
        return None, False

# Broadcasts args=None, kwargs=None to signify that a function call should be
# skipped on all processes. This can be used to define a loop exit condition.
# This should only be called by the broadcasting process.
def broadcast_skip(src=0, group=None, device=None):
    torch.distributed.broadcast_object_list(
        [None, None],
        src=src,
        group=group,
        device=device
    )
