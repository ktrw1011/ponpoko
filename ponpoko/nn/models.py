from typing import List
from tqdm.autonotebook import tqdm
import numpy as np
import tensorflow as tf
# import tensorflow_hub as hub

class UniversalSentenceEncoder:
    def __init__(self, model_dir: str, batch_size: int=128):
        self.module = hub.load(str(model_dir))
        self.batch_size = batch_size
        
    def __call__(self, texts: List[str], mode: str) -> np.ndarray:
        
        embeddings = []
        
        for i in tqdm(range(0, len(texts), self.batch_size)):
            text = texts[i:(i + self.batch_size)]
            
            if mode == "question":
                h_embedding = self.module.signatures['question_encoder'](tf.constant(text))['outputs']
            else:
                h_embedding = self.module.signatures['response_encoder'](input=tf.constant(text),context=tf.constant(text))['outputs']
                
            h_embedding = h_embedding.numpy()
            embeddings.append(h_embedding)
            
        return embeddings