import dataclasses

def to_cpu(x):
    return x.contiguous().detach().cpu()

def to_numpy(x):
    return to_cpu(x).numpy()


class AverageMeter:
    def __init__(self):
        self.step = 0
        self.total_val = 0.0

    def update(self, val):
        self.total_val += val
        self.step += 1

    def __call__(self):
        return self.total_val / float(self.step)

@dataclasses.dataclass
class FoldScore:
    fold: int
    epoch: int
    lr: float
    loss: float
    score: float