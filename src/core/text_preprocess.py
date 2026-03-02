import string
import torch.nn as nn
import torch
from .vocabulary import Vocabulary
from rich import print


def clean_file(filename: str) -> str:
    with open(filename, "r") as f:
        return clean_text(f.read())

def clean_text(text: str) -> str:
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator).lower()

def add_to_vocabulary_db(cleaned_text: str, vocab: Vocabulary):
    words = cleaned_text.split()
    indices = []
    for word in words:
        idx = vocab.add_word(word)
        indices.append(idx)
        
    return indices

def text_to_fixed_tensor(text: str, vocab: Vocabulary, train_mode: bool, max_len: int = 512) -> torch.Tensor:
    words = clean_text(text).split()
    if train_mode:
        indices = [vocab.add_word(w) or 0 for w in words]
    else:
        indices = [vocab.search_word(w) or 0 for w in words]
    
    if len(indices) < max_len:
        # magic index 1 means <PAD>
        indices += [1] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
        
    return torch.tensor(indices, dtype=torch.long)
