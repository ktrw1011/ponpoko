import torch
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


def collate_fn(data):
    def _pad_sequences(seqs):
        lens = [len(seq) for seq in seqs]
        max_len = max(lens)

        # 最初にPADDINGの配列用意
        padded_seqs = torch.zeros(len(seqs), max_len).long()
        for i, seq in enumerate(seqs):
            start = max_len - lens[i]
            padded_seqs[i, :lens[i]] = torch.LongTensor(seq)
        return padded_seqs
    
    # dataはList(datasetの__getitem__の戻り値)
    # list(zip(*data)) は転置操作
    # [[input_ids1, input_ids1], [attn_mask1, atten_mask2]] => [[input_ids1, atten_mask1], [input_ids2, atten_mask2]]
    # それを各要素のインデックスにまとめる
    transposed = list(zip(*data))
    index = transposed[0]
    input_ids, attention_mask = zip(*transposed[1])
    input_ids = _pad_sequences(input_ids)
    attention_mask = _pad_sequences(attention_mask)
    seqs = [input_ids, attention_mask]
    
    if len(transposed) == 2:
        return index, seqs
    
    return index, seqs, torch.FloatTensor(transposed[2])