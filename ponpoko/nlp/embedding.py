import warnings
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from tqdm.auto import tqdm

import gensim
from gensim.models.callbacks import CallbackAny2Vec

from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer

# from logging import getLogger, StreamHandler
# logger = getLogger(__name__)
# sh = StreamHandler()
# sh.
# logger.addHandler()

class GensimEpochProgress(CallbackAny2Vec):
    def __init__(self, num_epoch):
        self.now_epoch = 1
        self.num_epoch = num_epoch
        self.bar = tqdm(total=self.num_epoch)
        self.bar.set_description('Gensim Train Epoch')
        
    def on_epoch_end(self, model):
        self.bar.update(1)
        self.now_epoch += 1

        if self.now_epoch > self.num_epoch:
            self.bar.close()

def finetune_w2v(
    all_texts: Union[pd.Series, List[str]],
    vocab,
    embedding_matrix: np.ndarray,
    epoch:int = 3,
    embed_size: int=300,
    num_workers: int=1,
    return_model: bool=False,
    ) -> np.ndarray:

    word_freq = vocab.word_freq
    token2id = vocab.token2id
    special_tokens = vocab.special_tokens

    model = gensim.models.Word2Vec(min_count=1, workers=num_workers, iter=epoch, size=embed_size)

    # vocabを構築
    model.build_vocab_from_freq(word_freq)
    idxmap = np.array([token2id[w] for w in model.wv.index2entity])
    model.wv.vectors[:] = embedding_matrix[idxmap]
    model.trainables.syn1neg[:] = embedding_matrix[idxmap]

    # callback
    epoch_progress = GensimEpochProgress(num_epoch=epoch)
    # 語彙にないのが生じる？
    # 語彙をupdateするオプションを利用しない場合どうなるのか？
    model.train(all_texts, total_examples=len(all_texts), epochs=model.epochs, callbacks=[epoch_progress])

    emb_mean, emb_std = np.mean(model.wv.vectors), np.std(model.wv.vectors)

    embedding_matrix = np.zeros_like(embedding_matrix)
    for token, idx in token2id.items():
        if token in special_tokens:
            embedding_matrix[idx] = np.random.normal(emb_mean, emb_std, size=embedding_matrix.shape[1])
        else:
            embedding_matrix[idx] = model.wv.get_vector(token)

    if return_model:
        return embedding_matrix, model

    return embedding_matrix

def load_embed(embedding_path: Union[str, Path]) -> gensim.models.keyedvectors.KeyedVectors:
    """embeddingをロードするヘルパー関数

    Args:
        embedding_path (Union[str, Path]): [description]

    Raises:
        ValueError: [description]

    Returns:
        np.ndarray: [description]
    """
    if isinstance(embedding_path, str):
        embedding_path = Path(embedding_path)
    
    suffix = embedding_path.suffix
    embedding_path = str(embedding_path)

    if suffix == ".model":
        model = gensim.models.Word2Vec.load(embedding_path)
        wv = model.wv
        del model
        return wv

    elif suffix == ".bin":
        wv = gensim.models.KeyedVectors.load_word2vec_format(embedding_path, binary=True)
        return wv

    elif suffix == ".vec" or suffix == ".txt":
        wv = gensim.models.KeyedVectors.load_word2vec_format(embedding_path, binary=False)
        return wv

    elif suffix == '.kv':
        wv = gensim.models.KeyedVectors.load(embedding_path, mmap='r')
        return wv


class W2VBuilder:
    def __init__(self, embeding_or_path: Union[gensim.models.keyedvectors.KeyedVectors, dict, Path, str]):
        if isinstance(embeding_or_path, Path) or isinstance(embeding_or_path, str):
            embedding = load_embed(embeding_or_path)
        else:
            embedding = embeding_or_path

        self.embedding = embedding
        if isinstance(self.embedding, gensim.models.keyedvectors.KeyedVectors):
            self.type = "w2v"
        elif isinstance(self.embedding, dict):
            self.type = "dict"
        else:
            raise ValueError

        self._set_mean_std()

        self.ps = PorterStemmer()
        self.sb = SnowballStemmer('english')
        self.lc = LancasterStemmer()

    def _set_mean_std(self):
        if self.type == "w2v":
            self.emb_mean = np.mean(self.embedding.vectors)
            self.emb_std = np.std(self.embedding.vectors)

        elif self.type == "dict":
            self.emb_mean = np.mean(list(self.embedding.values()))
            self.emb_std = np.std(list(self.embedding.values()))

    def get_vector(self, word):

        if self.type == "w2v":
            if self.embedding.vocab.get(word):
                return self.embedding.get_vector(word)
            else:
                return None
            
        elif self.type == "dict":
            if word in self.embedding:
                return self.embedding[word]
            else:
                return None
            
    def build(self, vocab, max_features: Optional[int]=None, embed_size: int=300) -> np.ndarray:

        if max_features is None:
            max_features = vocab.max_features

        word_index = vocab.token2id
        special_token_index = vocab.get_special_token_index

        if vocab.max_features > max_features:
            warnings.warn('vocab features over max_features args')

        nb_words = min(max_features + len(special_token_index), len(word_index))

        if max(special_token_index) > nb_words:
            raise ValueError("Not include special tokens in Embedding")

        # 正規化
        embedding_matrix = np.random.normal(self.emb_mean, self.emb_std, (nb_words, embed_size))

        adding_tokens = []

        for key, i in tqdm(list(word_index.items())[:nb_words]):
            word = key
            embedding_vector = self.get_vector(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                adding_tokens.append(word)
                continue
            word = key.lower()
            embedding_vector = self.get_vector(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                adding_tokens.append(word)
                continue
            word = key.upper()
            embedding_vector = self.get_vector(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                adding_tokens.append(word)
                continue
            word = key.capitalize()
            embedding_vector = self.get_vector(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                adding_tokens.append(word)
                continue
            word = self.ps.stem(key)
            embedding_vector = self.get_vector(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                adding_tokens.append(word)
                continue
            word = self.lc.stem(key)
            embedding_vector = self.get_vector(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                adding_tokens.append(word)
                continue
            word = self.sb.stem(key)
            embedding_vector = self.get_vector(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                adding_tokens.append(word)
                continue

        return embedding_matrix