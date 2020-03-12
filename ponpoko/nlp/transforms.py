import re
import string

misspell_dict = {"aren't": "are not", "can't": "can not", "couldn't": "could not",
                 "didn't": "did not", "doesn't": "does not", "don't": "do not",
                 "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                 "he'd": "he would", "he'll": "he will", "he's": "he is",
                 "i'd": "I had", "i'll": "I will", "i'm": "I am", "isn't": "is not",
                 "it's": "it is", "it'll": "it will", "i've": "I have", "let's": "let us",
                 "mightn't": "might not", "mustn't": "must not", "shan't": "shall not",
                 "she'd": "she would", "she'll": "she will", "she's": "she is",
                 "shouldn't": "should not", "that's": "that is", "there's": "there is",
                 "they'd": "they would", "they'll": "they will", "they're": "they are",
                 "they've": "they have", "we'd": "we would", "we're": "we are",
                 "weren't": "were not", "we've": "we have", "what'll": "what will",
                 "what're": "what are", "what's": "what is", "what've": "what have",
                 "where's": "where is", "who'd": "who would", "who'll": "who will",
                 "who're": "who are", "who's": "who is", "who've": "who have",
                 "won't": "will not", "wouldn't": "would not", "you'd": "you would",
                 "you'll": "you will", "you're": "you are", "you've": "you have",
                 "'re": " are", "wasn't": "was not", "we'll": " will", "tryin'": "trying"}

extra_punct = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']',
          '>', '%', '=', '#', '*', '+', '\\', '•', '~', '@', '£', '·', '_', '{', '}', '©', '^',
          '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â', '█',
          '½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶',
          '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼',
          '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲',
          'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»', '，', '♪',
          '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√']

class Compose:
    """
    各transfomerモジュールのインスタンスをリストで渡せばその順番でテキストの正規化を行う
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, text):
        for trans in self.transforms:
            text = trans(text)
        return text

    @property
    def composed(self):
        names = []
        for trans in self.transforms:
            names.append(trans.__class__.__name__)

        return names

class RemoveHtmlTag:
    def __init__(self):
        self.tag_pattern = re.compile(r'<(".*?"|\'.*?\'|[^\'"])*?>')

    def __call__(self, text):
        text = re.sub(self.tag_pattern, "", text)
        return text

class ReplaceTypicalMissSpelling:
    def __init__(self):
        self.pattern = re.compile('(%s)' % '|'.join(misspell_dict.keys()))
        
    def _replace(self, match):
        return misspell_dict[match.group(0)]

    def __call__(self, text:str) -> str:
        return self.pattern.sub(self._replace, text)

class CleanNumbers:
    def __init__(self):
        self.pattern = re.compile(r'\d+')

    def __call__(self, text: str) -> str:
        return re.sub(self.pattern, ' ', text)


class SpacingPunctuation:
    def __init__(self):
        self.all_punct = list(set(list(string.punctuation)+ extra_punct))
        self.all_punct.remove('-')
        self.all_punct.remove('.')

    def __call__(self, text):
        for punct in self.all_punct:
            if punct in text:
                text = text.replace(punct, f' {punct} ')
        return text


class CleanRepeatWords:

    def __call__(self, text):
        text = re.sub(r"(I|i)(I|i)+ng", "ing", text)
        text = re.sub(r"(L|l)(L|l)(L|l)+y", "lly", text)
        text = re.sub(r"(A|a)(A|a)(A|a)+", "a", text)
        text = re.sub(r"(C|c)(C|c)(C|c)+", "cc", text)
        text = re.sub(r"(D|d)(D|d)(D|d)+", "dd", text)
        text = re.sub(r"(E|e)(E|e)(E|e)+", "ee", text)
        text = re.sub(r"(F|f)(F|f)(F|f)+", "ff", text)
        text = re.sub(r"(G|g)(G|g)(G|g)+", "gg", text)
        text = re.sub(r"(I|i)(I|i)(I|i)+", "i", text)
        text = re.sub(r"(K|k)(K|k)(K|k)+", "k", text)
        text = re.sub(r"(L|l)(L|l)(L|l)+", "ll", text)
        text = re.sub(r"(M|m)(M|m)(M|m)+", "mm", text)
        text = re.sub(r"(N|n)(N|n)(N|n)+", "nn", text)
        text = re.sub(r"(O|o)(O|o)(O|o)+", "oo", text)
        text = re.sub(r"(P|p)(P|p)(P|p)+", "pp", text)
        text = re.sub(r"(Q|q)(Q|q)+", "q", text)
        text = re.sub(r"(R|r)(R|r)(R|r)+", "rr", text)
        text = re.sub(r"(S|s)(S|s)(S|s)+", "ss", text)
        text = re.sub(r"(T|t)(T|t)(T|t)+", "tt", text)
        text = re.sub(r"(V|v)(V|v)+", "v", text)
        text = re.sub(r"(Y|y)(Y|y)(Y|y)+", "y", text)
        text = re.sub(r"(Z|z)(Z|z)(Z|z)+", "zz", text)
        return text