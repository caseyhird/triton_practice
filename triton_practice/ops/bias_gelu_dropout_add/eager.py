import torch.nn.functional as F


def forward(x, bias, residual, p=0.0, training=True):
    y = x + bias if bias is not None else x
    y = F.gelu(y, approximate="tanh") # using tanh in triton so need a fair comparison
    if training and p > 0:
        y = F.dropout(y, p=p, training=True)
    return y + residual
