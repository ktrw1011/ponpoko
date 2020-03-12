from typing import List
import numpy as np
from gensim import corpora, models, matutils

PASSES = 10
LDA_DIMENSION = 5
WORKERS = 2

def get_lda_vec(corp, lda):
    return matutils.sparse2full(lda.get_document_topics(corp), lda.num_topics)

def generate_lda_feature(docs : List[List[str]], gensim_dict) -> np.ndarray:
    """
    examples
    =======
    docs: [[tokenA, tokenB, tokenC], [tokenD, tokenE, ...]]
    """
    corpus = [gensim_dict.doc2bow(doc) for doc in docs]

    lda = models.LdaMulticore(
                corpus=corpus, id2word=gensim_dict, workers=WORKERS, random_state=42, passes=10,
                num_topics=LDA_DIMENSION, minimum_probability=0.001)

    lda_array = np.zeros((len(corpus), LDA_DIMENSION))
    for i, corp in enumerate(corpus):
        lda_array[i, :] = get_lda_vec(corp, lda)
    
    return lda_array