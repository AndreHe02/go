from nltk.tokenize import word_tokenize
import pickle as pkl
import numpy as np

class TextProcessor:

    def __init__(self, max_sent_length, itos=None):
        self.max_sent_length = max_sent_length
        if itos is None:
            self.itos = pkl.load(open('vocab/itos.pkl', 'rb'))
        else:
            self.itos = itos
        self.stoi = {s:idx for idx, s in enumerate(self.itos)}
        self.vocab_size = len(self.stoi)

    def text2tokenarr(self, text):
        tokens = word_tokenize(text.lower())
        token_idxes = [self.stoi['<bos>']] + [self.token2idx(token) for token in tokens] + [self.stoi['<eos>']]
        return token_idxes[:self.max_sent_length]

    def token2idx(self, token):
        if token in self.stoi:
            return self.stoi[token]
        else:
            return self.stoi['<unk>']

    def npify_tokenarrs(self, sents):
        lengths = [len(sent) for sent in sents]
        max_length = np.max(lengths)
        returned = []
        for sent in sents:
            sent = sent[:] + [self.stoi['<pad>']] * (max_length - len(sent))
            returned.append(sent)
        return np.array(returned), np.array(lengths)

    def npifytext(self, texts):
        tokenarrs = [self.text2tokenarr(text) for text in texts]
        return self.npify_tokenarrs(tokenarrs)