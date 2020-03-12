try:
    import MeCab
except:
    pass

import nltk
import re

class MeCabTokenizer:
    def __init__(self, mecab_args='-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd'):
        self.tagger = MeCab.Tagger(mecab_args)
        self.tagger.parse("")

        self.not_include_pos = ["助詞", "副詞"]

        self.stop_words = re.compile(r"[\!-&%\(\)\.\,/=\-，．。、\ ￥’”→↓←↑…「」（）ー\ ～『 \・～『』・；：※々ゞヶヵ％]")

    def tokenize(self, text):
        parsed = []
        lines = self.tagger.parse(text).split("\n")

        for line in lines[:-2]:
            items = re.split('[\t,]',line)
            
            if len(items) >= 2 and items[1] in self.not_include_pos:
                continue
            if re.match(self.stop_words, items[0]):
                continue

            parsed.append(items[0])

        return parsed


class NltkTokenizer:
    """
    特別なpunctは処理してくれないので先にクリーニングした方が良い
    """
    def __init__(self):
        pass

    def tokenize(self, text):
        return nltk.word_tokenize(text)
