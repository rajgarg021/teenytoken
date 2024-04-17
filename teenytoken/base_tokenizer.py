"""
Contains the base tokenizer class
"""

def get_stats(ids, countDict=None):
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [5, 6, 3, 5, 6] -> {(5, 6): 2, (6, 3): 1, (3, 5): 1}
    Optionally allows to update an existing dictionary of counts
    """
    countDict = {} if countDict is None else countDict
    for pair in zip(ids, ids[1:]): # iterate consecutive elements
        countDict[pair] = countDict.get(pair, 0) + 1
    return countDict

def merge(ids, pair, idx):
    """
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx
    Example: ids=[5, 6, 3, 5, 6], pair=(5, 6), idx=4 -> [4, 3, 4]
    """
    newids = []
    i = 0
    while i < len(ids):
        # if not at the very last position AND the pair matches, replace it
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

class Tokenizer:
    """Base class for tokenizers"""

    def __init__(self):
        # default: vocab size of 256 (all bytes), no merges and no patterns
        self.merges = {} # (int, int) -> int
        self.pattern = "" # str
        self.special_tokens = {} # str -> int, e.g. {'<|endoftext|>': 100257}
        self.vocab = self._build_vocab() # int -> bytes

    def train(self, text, vocab_size, verbose=False):
        # Tokenizer can train a vocabulary of size vocab_size from text
        raise NotImplementedError

    def encode(self, text):
        # Tokenizer can encode a string into a list of integers
        raise NotImplementedError

    def decode(self, ids):
        # Tokenizer can decode a list of integers into a string
        raise NotImplementedError
    
    def _build_vocab(self):
        # vocab is simply and deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab