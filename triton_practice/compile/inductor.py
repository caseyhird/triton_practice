import torch


def compiled(fn, fullgraph=True, dynamic=True, mode="max-autotune"):
    return torch.compile(fn, fullgraph=fullgraph, dynamic=dynamic, mode=mode)
