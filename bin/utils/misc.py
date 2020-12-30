import torch
import sys


def generate_subsequent_mask(sz, device="cpu"):
    return (torch.triu(
        torch.ones(sz, sz, dtype=torch.int, device=device)
    ) == 0).transpose(0, 1)


def print_progress(cur_val, max_val, max_len=50, prefix='',
                   pattern='=', suffix='', time_used=0):
    cur = int(cur_val / max_val * max_len)
    line = "\x1b[1K\r{} [{}] {}/{} {} - TIME_ELAPSED: {:.2f}s".format(
        prefix, cur * pattern + (max_len - cur) * ' ', \
        cur_val, max_val, suffix, time_used)
    end = ''
    if cur_val == max_val:
        end = '\n'
    print(line, end=end)

def count_params(module):
    """
    Compute number of trainable params of a module
    Args:
        module (nn.Module)
    Returns:
        (int) - Number of params
    """
    def size(x):
        n_params = 1
        for d in x.size():
            n_params *= d
        return n_params
    model_parameters = filter(lambda p: p.requires_grad, module.parameters())
    return sum([size(p) for p in model_parameters])
