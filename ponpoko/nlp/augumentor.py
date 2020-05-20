import random
import re

from nltk.tokenize import word_tokenize

from nltk.corpus import wordnet

from albumentations.core.transforms_interface import DualTransform, BasicTransform

import nlpaug.augmenter.word as naw


stop_words = ['i', 'me', 'my', 'myself', 'we', 'our',
			'ours', 'ourselves', 'you', 'your', 'yours',
			'yourself', 'yourselves', 'he', 'him', 'his',
			'himself', 'she', 'her', 'hers', 'herself',
			'it', 'its', 'itself', 'they', 'them', 'their',
			'theirs', 'themselves', 'what', 'which', 'who',
			'whom', 'this', 'that', 'these', 'those', 'am',
			'is', 'are', 'was', 'were', 'be', 'been', 'being',
			'have', 'has', 'had', 'having', 'do', 'does', 'did',
			'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
			'because', 'as', 'until', 'while', 'of', 'at',
			'by', 'for', 'with', 'about', 'against', 'between',
			'into', 'through', 'during', 'before', 'after',
			'above', 'below', 'to', 'from', 'up', 'down', 'in',
			'out', 'on', 'off', 'over', 'under', 'again',
			'further', 'then', 'once', 'here', 'there', 'when',
			'where', 'why', 'how', 'all', 'any', 'both', 'each',
			'few', 'more', 'most', 'other', 'some', 'such', 'no',
			'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
			'very', 's', 't', 'can', 'will', 'just', 'don',
			'should', 'now', ''
            ]

def get_synonyms(word:str):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

class NLPBaseTransform(BasicTransform):
    TOKENIZER_REGEX = re.compile(r'(\W)')
    
    @property
    def targets(self):
        return {"data": self.apply}
    
    def update_params(self, params, **kwargs):
        if hasattr(self, "interpolation"):
            params["interpolation"] = self.interpolation
        if hasattr(self, "fill_value"):
            params["fill_value"] = self.fill_value
        return params

    @classmethod
    def _tokenizer(cls, text):
        tokens = cls.TOKENIZER_REGEX.split(text)
        return [t for t in tokens if len(t.strip()) > 0]

    @classmethod
    def _reverse_tokenizer(cls, tokens):
        return ' '.join(tokens)


class RandomSwapWordTransform(NLPBaseTransform):
    def __init__(self, tokenizer=None, reverse_tokenizer=None, always_apply=False, p=0.5):
        super().__init__(always_apply,  p)

        self.tokenizer = tokenizer or self._tokenizer
        self.reverse_tokenizer = reverse_tokenizer or self._reverse_tokenizer
    
    def apply(self, data, **params):
        sent = self.tokenizer(data)
        
        if len(sent) == 1:
            return data

        pos = random.sample(range(len(sent)), 2)
        pos1, pos2 = pos[0], pos[1]
        sent[pos1], sent[pos2] = sent[pos2], sent[pos1]
        
        return self.reverse_tokenizer(sent)

class RandomDeleionTransform(NLPBaseTransform):
    def __init__(self, aug_min=1, aug_max=2, aug_p=0.3, tokenizer=None, reverse_tokenizer=None, always_apply=False, p=0.5):
        super().__init__(always_apply,  p)

        self.tokenizer = tokenizer or self._tokenizer
        self.reverse_tokenizer = reverse_tokenizer or self._reverse_tokenizer

        self.aug_min = aug_min
        self.aug_max = aug_max
        self.aug_p = aug_p

    def apply(self, data, **params):
        words = self.tokenizer(data)
        
        if len(words) == 1:
            return data
        
        aug_counter = 0
        #randomly delete words with probability p
        new_words = []
        for word in words:
            r = random.uniform(0, 1)
            if r < self.aug_p and aug_counter < self.aug_max:
                aug_counter += 1
            else:
                new_words.append(word)

        while aug_counter < self.aug_min:
            randomint = random.randint(0, len(word)-1)
            del new_words[randomint]

        return self.reverse_tokenizer(new_words)


class RandomWordSplitTransform(NLPBaseTransform):
        def __init__(self, aug_max=2, aug_p=0.3, tokenizer=None, always_apply=False, p=0.5):
            super().__init__(always_apply,  p)
            
            self.aug = naw.SplitAug(aug_max=aug_max, aug_p=aug_p, tokenizer=tokenizer)
            
        def apply(self, data, **params):
            augmented_text = self.aug.augment(data)
            
            return augmented_text


class SynonymTransform(NLPBaseTransform):
    def __init__(self, aug_min=1, aug_max=10, aug_p=0.3, tokenizer=None, always_apply=False, p=0.5):
        super().__init__(always_apply, p)

        self.aug = naw.SynonymAug(
            aug_min=aug_min,
            aug_max=aug_max,
            aug_p=aug_p,
            tokenizer=tokenizer,
            )

    def apply(self, data, **params):
        augmented_text = self.aug.augment(data)
            
        return augmented_text