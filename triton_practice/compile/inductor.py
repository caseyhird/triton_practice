import torch


def compiled(fn, fullgraph=False, dynamic=True, mode="max-autotune"):
    return torch.compile(fn, fullgraph=fullgraph, dynamic=dynamic, mode=mode)
