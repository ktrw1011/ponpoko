from typing import Dict, List, Tuple
from collections import Counter, namedtuple

from gensim.corpora import Dictionary

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]"]

class GensimVocab:
    def __init__(self, special_tokens=SPECIAL_TOKENS):
        self.special_tokens = special_tokens
        self.dict = Dictionary([self.special_tokens])
        self.word_freq = None
        self.max_features = None

    def build(self, texts:list):
        self.dict.add_documents(texts)
        self.dict[0] #make id2token dict
        self.make_word_freq()
        self.max_features = len(self.dict)

    def filter(self, no_below=5, no_above=1, keep_n=100000):
        self.dict.filter_extremes(
            no_below=no_below,
            no_above=no_above,
            keep_n=100000,
            keep_tokens=self.special_tokens
            )

        self.dict[0] #make id2token dict
        self.make_word_freq()
        self.max_features = len(self.dict)

    def make_word_freq(self):
        self.dict[0] #make id2token dict
        self.word_freq = {self.dict.id2token[k]: v for k, v in self.dict.dfs.items()}

    @property
    def token2id(self):
        return self.dict.token2id

    @property
    def id2token(self):
        return self.dict.id2token

    @property
    def get_special_token_index(self):
        return [self.token2id[token] for token in self.special_tokens]


class SimpleVocab:
    def __init__(self, special_tokens=SPECIAL_TOKENS):
        self.token2id = {}
        self.id2token = {}
        self.word_freq = {}
        self.special_tokens = special_tokens
        self.max_features = None

    @classmethod
    def from_gensim_vocab(cls, gensim_vocab: GensimVocab, max_features: int=100000):
        """
        """
        simple_vocab = SimpleVocab(gensim_vocab.special_tokens)

        # special tokenを先頭に配置する
        for word in gensim_vocab.special_tokens:
            org_idx = gensim_vocab.token2id[word]

            simple_vocab.token2id[word] = org_idx
            simple_vocab.id2token[org_idx] = word
            simple_vocab.word_freq[word] = 1

        # 出現数順にソートされる
        idx = len(gensim_vocab.get_special_token_index)
        for org_idx, count in sorted(gensim_vocab.dict.dfs.items(), key=lambda x: x[1], reverse=True):
            if org_idx in gensim_vocab.get_special_token_index:
                continue

            word = gensim_vocab.id2token[org_idx]
            simple_vocab.token2id[word] = idx
            simple_vocab.id2token[idx] = word
            simple_vocab.word_freq[word] = count

            if max_features and len(simple_vocab.token2id) >= max_features:
                break

            idx += 1

        simple_vocab.max_features = len(simple_vocab.token2id)
        return simple_vocab

    @property
    def get_special_token_index(self):
        return [self.token2id[token] for token in self.special_tokens]

    def build(self, texts:List[str], max_features: int=100000):
        self.token2id.update({token:_id for _id, token in enumerate(self.special_tokens)})

        counter = Counter()
        for text in texts:
            counter.update(text.split())

        self.token2id.update({
            token: _id+len(self.special_tokens) for _id, (token, count) in enumerate(counter.most_common(max_features))
        })

        self.id2token = {v: k for k, v in self.token2id.items()}

        self.word_freq = {
            **{k:1 for k in self.special_tokens},
            **dict(counter.most_common(max_features))
        }
        
        self.max_features = len(self.token2id)


def convert_tokens2ids(tokens: List[str], token2id:Dict[str, int], unk_index=None, max_seq_len=200) -> List[int]:
    ids = [token2id.get(token, unk_index) for token in tokens[:max_seq_len]]
    return ids