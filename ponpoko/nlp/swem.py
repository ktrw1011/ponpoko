from typing import List, Dict
import numpy as np

class BaseSWEM:
    def get_word_embeddings(self, text):
        raise NotImplementedError

    def average_pooling(self, text):
        word_embeddings = self.get_word_embeddings(text)
        return np.mean(word_embeddings, axis=0)

    def max_pooling(self, text):
        word_embeddings = self.get_word_embeddings(text)
        return np.max(word_embeddings, axis=0)

    def concat_average_max_pooling(self, text):
        word_embeddings = self.get_word_embeddings(text)
        return np.r_[np.mean(word_embeddings, axis=0), np.max(word_embeddings, axis=0)]


class SWEM:
    """
    Simple Word-Embeddingbased Models (SWEM)
    https://arxiv.org/abs/1805.09843v1
    https://yag-ays.github.io/project/swem/
    """

    def __init__(self, w2v, tokenizer=None, oov_initialize_range=(-0.01, 0.01)):
        self.w2v = w2v
        self.tokenizer = tokenizer
        self.vocab = set(self.w2v.vocab.keys())
        self.embedding_dim = self.w2v.vector_size
        self.oov_initialize_range = oov_initialize_range

        if self.oov_initialize_range[0] > self.oov_initialize_range[1]:
            raise ValueError("Specify valid initialize range: "
                             f"[{self.oov_initialize_range[0]}, {self.oov_initialize_range[1]}]")

    def get_word_embeddings(self, text):
        vectors = []
        words = []
        
        if self.tokenizer:
            np.random.seed(abs(hash(text)) % (10 ** 8))
            words = self.tokenizer.tokenize(text)
        else:
            words = text
            np.random.seed(abs(hash("".join(words))) % (10 ** 8))

        for word in words:
            if word in self.vocab:
                vectors.append(self.w2v[word])
            else:
                vectors.append(np.random.uniform(self.oov_initialize_range[0],
                                                 self.oov_initialize_range[1],
                                                 self.embedding_dim))
        return np.array(vectors)

    def average_pooling(self, text):
        word_embeddings = self.get_word_embeddings(text)
        return np.mean(word_embeddings, axis=0)

    def max_pooling(self, text):
        word_embeddings = self.get_word_embeddings(text)
        return np.max(word_embeddings, axis=0)

    def concat_average_max_pooling(self, text):
        word_embeddings = self.get_word_embeddings(text)
        return np.r_[np.mean(word_embeddings, axis=0), np.max(word_embeddings, axis=0)]


class SWEMKeyEmbedding(BaseSWEM):
    def __init__(self, key_embedding: Dict[str, np.ndarray]) -> np.ndarray:
        self.vocab = list(key_embedding.keys())
        self.key_embeddings = key_embedding
        self.embed_dim = key_embedding[self.vocab[0]].shape[0]

        embedding = np.array(list(key_embedding.values()))
        self.mean, self.std = np.mean(embedding), np.std(embedding)

    def get_word_embeddings(self, text):
        words = text.split(" ")
        vectors = np.empty((0, self.embed_dim))

        for word in words:
            if word in self.vocab:
                vectors = np.vstack([vectors, self.key_embeddings[word]])
            else:
                embed = np.random.normal(self.mean, self.std, self.embed_dim)
                vectors = np.vstack([vectors, embed])

        return vectors

    


    
