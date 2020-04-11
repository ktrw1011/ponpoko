from typing import Dict, List, Tuple, Optional
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import gensim
from gensim.models import Word2Vec

from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer

# from logging import getLogger, StreamHandler
# logger = getLogger(__name__)
# sh = StreamHandler()
# sh.
# logger.addHandler()


def train_w2v_embedding(docs: List[List[str]], window: int=5) -> Dict[str, np.ndarray]:
    """
    w2v
    """

    model = Word2Vec(docs, size=5, min_count=0, window=window)

    # vectorsプロパティはvocabのvalueのindex順に並んでいる
    vectors = model.wv.vectors
    
    key_embeddings = {k:vectors[v.index] for k, v in model.wv.vocab.items()}

    # df = pd.DataFrame.from_dict(key_embeddings, orient="index")
    return key_embeddings

def w2v_finetune(
    all_texts: List[str],
    vocab,
    embedding_matrix: np.ndarray,
    embed_size: int=300,
    return_model: bool=False
    ) -> np.ndarray:

    """
    fine tuning w2v

    example
    =======
    w2v_finetune(df['texts'], vocab, embedding_matrix)
    """

    word_freq = vocab.word_freq
    token2id = vocab.token2id
    special_tokens = vocab.special_tokens

    model = Word2Vec(min_count=1, workers=1, iter=3, size=embed_size)

    model.build_vocab_from_freq(word_freq)

    idxmap = np.array([token2id[w] for w in model.wv.index2entity])

    model.wv.vectors[:] = embedding_matrix[idxmap]

    model.trainables.syn1neg[:] = embedding_matrix[idxmap]

    model.train(all_texts, total_examples=len(all_texts), epochs=model.epochs)

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

class W2VEmbedding:
    def __init__(self, embedding_path: Optional[str], binary: bool=True):
        
        self.__embeddings_index = None
        self._type = "w2v"

        if embedding_path is not None:
            suffix = Path(embedding_path).suffix
            if suffix == ".vec":
                self.__embeddings_index = gensim.models.KeyedVectors.load_word2vec_format(embedding_path, binary=binary)
            elif suffix == ".pkl":
                self.__embeddings_index = joblib.load(embedding_path)
                self._type = "dict"

        self.ps = PorterStemmer()
        self.sb = SnowballStemmer('english')
        self.lc = LancasterStemmer()

    def _get_vector(self, word):
        if self._type == "w2v":
            if self.__embeddings_index.vocab.get(word):
                return self.__embeddings_index.get_vector(word)
            else:
                return None
            
        elif self._type == "dict":
            if word in self.__embeddings_index:
                return self.__embeddings_index[word]
            else:
                return None

    @property
    def embedding_matrix(self):
        return self.__embeddings_index

    @embedding_matrix.setter
    def embedding_matrix(self, value):
        self.__embeddings_index = value
            
    def __call__(
        self,
        vocab,
        max_features: int,
        embed_size: int=300,
        ) -> np.ndarray:

        word_index = vocab.token2id
        special_token_index = vocab.get_special_token_index

        if vocab.max_features > max_features:
            warnings.warn('vocab features over max_features args')

        nb_words = min(max_features + len(special_token_index), len(word_index))

        if max(special_token_index) > nb_words:
            raise ValueError("Not include special tokens in Embedding")

        emb_mean,emb_std = -0.0033469985, 0.109855495

        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

        adding_tokens = []

        for key, i in tqdm(list(word_index.items())[:nb_words]):
            word = key
            embedding_vector = self._get_vector(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                adding_tokens.append(word)
                continue
            word = key.lower()
            embedding_vector = self._get_vector(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                adding_tokens.append(word)
                continue
            word = key.upper()
            embedding_vector = self._get_vector(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                adding_tokens.append(word)
                continue
            word = key.capitalize()
            embedding_vector = self._get_vector(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                adding_tokens.append(word)
                continue
            word = self.ps.stem(key)
            embedding_vector = self._get_vector(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                adding_tokens.append(word)
                continue
            word = self.lc.stem(key)
            embedding_vector = self._get_vector(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                adding_tokens.append(word)
                continue
            word = self.sb.stem(key)
            embedding_vector = self._get_vector(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                adding_tokens.append(word)
                continue

        return embedding_matrix