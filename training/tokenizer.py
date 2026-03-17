"""
fortza-super-tiny
Character tokenizer — built from dataset, saved alongside weights.
"""

import json


SPECIAL = {
    "<PAD>": 0,
    "<START>": 1,
    "<END>": 2,
    "<UNK>": 3,
}


class Tokenizer:
    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}

    def build(self, text):
        chars = sorted(set(text))
        self.char2idx = dict(SPECIAL)
        for ch in chars:
            if ch not in self.char2idx:
                self.char2idx[ch] = len(self.char2idx)
        self.idx2char = {v: k for k, v in self.char2idx.items()}

    def encode(self, text):
        unk = SPECIAL["<UNK>"]
        return [self.char2idx.get(ch, unk) for ch in text]

    def decode(self, indices):
        special_vals = set(SPECIAL.values())
        return "".join(
            self.idx2char.get(i, "?")
            for i in indices
            if i not in special_vals
        )

    @property
    def vocab_size(self):
        return len(self.char2idx)

    @property
    def start_idx(self):
        return SPECIAL["<START>"]

    @property
    def end_idx(self):
        return SPECIAL["<END>"]

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.char2idx, f, ensure_ascii=False, indent=2)

    def load(self, path):
        with open(path) as f:
            self.char2idx = json.load(f)
        self.idx2char = {v: k for k, v in self.char2idx.items()}
