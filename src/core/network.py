from pathlib import Path
from typing import Any

import torch
from torch import nn

from core.vocabulary import Vocabulary
from core.text_preprocess import *

from rich import print


class TextEmotionClassificationNetwork(nn.Module):
    """
    Use LSTM classification model

    Args:
        nn (_type_): _description_
    """
    def __init__(self, vocabulary: Vocabulary, embed_dim: int, hidden_dim_1: int, num_class: int) -> None:
        super().__init__()
        
        self.vocabulary_size = vocabulary.size()
        
        # 1. Embedding
        self.l_embedding = nn.Embedding(self.vocabulary_size, embed_dim)
        
        # 2. LSTM (Bidirectional)
        self.l_lstm = nn.LSTM(embed_dim, hidden_dim_1, batch_first=True, num_layers=1, bidirectional=True)
        
        self.l_fc = nn.Linear(hidden_dim_1 * 2, num_class)
    
    def forward(self, text_vector: torch.Tensor):
        # Embedding layer
        embedded = self.l_embedding(text_vector) 
        
        # LSTM layer
        output, (hn, cn) = self.l_lstm(embedded)
        
        cat_hidden = torch.cat((hn[0], hn[1]), dim=1)
        
        # 4. Final Linear Layer
        return self.l_fc(cat_hidden)
    
    
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    model_filepath = project_root / "data" / "model"
    
    vocab = Vocabulary()
    
    state_dict = torch.load(model_filepath, weights_only=True)
    example_word_vec = text_to_fixed_tensor("bad", vocab, False).unsqueeze(0)
    
    net = TextEmotionClassificationNetwork(vocab, 128, 256, 2)
    
    net.load_state_dict(state_dict)
    net.eval()
    
    with torch.no_grad():
        result = net(example_word_vec)
        print(f"Reasoning result: {result}")