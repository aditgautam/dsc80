# project.py


import pandas as pd
import numpy as np
from pathlib import Path
import re
import requests
import time


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------
START_MARKER = re.compile(
    r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK .* \*\*\*",
    flags=re.IGNORECASE
)
END_MARKER = re.compile(
    r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK .* \*\*\*",
    flags=re.IGNORECASE
)

#helper tpo dea;l with  \n

def _normalize_newlines(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")

def _slice_text(text: str) -> str:
    start_m = START_MARKER.search(text)
    end_m = END_MARKER.search(text)
    if start_m and end_m and end_m.start() > start_m.end():
        return text[start_m.end(): end_m.start()]
    return text

def get_book(url):
    time.sleep(0.5)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    text = resp.content.decode(resp.encoding or "utf-8", errors="replace")
    text = _normalize_newlines(text)
    return _slice_text(text)


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


START = '\x02'
END   = '\x03'

_WORD_OR_PUNCT = re.compile(r"\w+|[^\s\w]", flags=re.UNICODE)
_PARA_SEP      = re.compile(r"(?:[^\S\n]*\n){2,}[^\S\n]*")

def tokenize(book_string):
    s = (book_string
         .replace("\r\n", "\n")
         .replace("\r", "\n")
         .replace("\u2028", "\n")
         .replace("\u2029", "\n")
         .replace("\u0085", "\n"))

    paras = [p for p in _PARA_SEP.split(s) if re.search(r"\S", p)]
    if not paras:
        return [START, END]

    out = []
    for p in paras:
        out.append(START)
        out.extend(_WORD_OR_PUNCT.findall(p))
        out.append(END)
    return out


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


class UniformLM(object):


    def __init__(self, tokens):

        self.mdl = self.train(tokens)
        
    def train(self, tokens):
        vocab = pd.Index(pd.unique(tokens))
        probs = np.full(len(vocab), 1.0/len(vocab))
        return pd.Series(probs, index=vocab)
    
    def probability(self, words):
        p =  self.mdl.reindex(words).to_numpy()
        if np.any(pd.isna(p)):
            return 0.0
        return float(np.prod(p))
        
    def sample(self, M):
        vocab = self.mdl.index.to_numpy()
        picks = np.random.choice(vocab, size=int(M), replace=True)
        return " ".join(picks)


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


class UnigramLM(object):
    
    def __init__(self, tokens):
        self.mdl = self.train(tokens)
    
    def train(self, tokens):
        counts = pd.Series(tokens, dtype="object").value_counts()
        probs = counts / counts.sum()
        probs.sort_index(inplace=False)
        return probs
    
    def probability(self, words):
        p = self.mdl.reindex(words).to_numpy()
        if np.any(pd.isna(p)):
            return 0.0
        return float(np.prod(p))
        
    def sample(self, M):
        vocab = self.mdl.index.to_numpy()
        p = self.mdl.to_numpy()
        picks = np.random.choice(vocab, size=int(M), replace=True, p=p)
        return " ".join(picks)


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


class NGramLM(object):
    
    def __init__(self, N, tokens):
        # You don't need to edit the constructor,
        # but you should understand how it works!
        
        self.N = N

        ngrams = self.create_ngrams(tokens)

        self.ngrams = ngrams
        self.mdl = self.train(ngrams)

        if N < 2:
            raise Exception('N must be greater than 1')
        elif N == 2:
            self.prev_mdl = UnigramLM(tokens)
        else:
            self.prev_mdl = NGramLM(N-1, tokens)

    def create_ngrams(self, tokens):
        N = self.N
        return tuple(tuple(tokens[i:i+N]) for i in range(len(tokens)-N+1))
        
    def train(self, ngrams):
        N1 = [ng[:-1] for ng in ngrams]
        c_ng = pd.Series(ngrams, dtype="object").value_counts()
        c_n1 = pd.Series(N1, dtype="object").value_counts()
        df = c_ng.rename("cnt").to_frame().reset_index().rename(columns={"index":"ngram"})
        df["n1gram"] = df["ngram"].apply(lambda t: t[:-1])
        df["prob"] = df["cnt"] / df["n1gram"].map(c_n1)
        return df[["ngram","n1gram","prob"]]
    
    def probability(self, words):
        w = tuple(words)
        if len(w) < self.N:
            return self.prev_mdl.probability(w)
        p = self.prev_mdl.probability(w[:self.N-1])
        if p == 0.0:
            return 0.0
        idx = self.mdl.set_index("ngram")["prob"]
        for i in range(self.N-1, len(w)):
            ng = w[i-(self.N-1):i+1]
            q = idx.get(ng, np.nan)
            if pd.isna(q):
                return 0.0
            p *= float(q)
        return float(p)
    

    def sample(self, M):
        def dist(model, context):
            if isinstance(model, NGramLM):
                if len(context) == model.N-1:
                    rows = model.mdl[model.mdl["n1gram"] == context]
                    if rows.empty:
                        return np.array(['\x03']), np.array([1.0])
                    toks = rows["ngram"].apply(lambda t: t[-1]).to_numpy()
                    probs = rows["prob"].to_numpy()
                    probs = probs / probs.sum()
                    return toks, probs
                return dist(model.prev_mdl, context)
            vocab = model.mdl.index.to_numpy()
            probs = model.mdl.to_numpy()
            probs = probs / probs.sum()
            return vocab, probs

        out = ['\x02']
        while len(out) - 1 < M:
            k = min(self.N-1, len(out))
            context = tuple(out[-k:]) if k > 0 else tuple()
            toks, probs = dist(self, context)
            nxt = np.random.choice(toks, p=probs)
            out.append(nxt)
        return " ".join(out)
