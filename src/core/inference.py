from pathlib import Path
from core.network import TextEmotionClassificationNetwork
from core.vocabulary import Vocabulary
from core.text_preprocess import *

import torch
import torch.nn as nn

from rich import print

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class Inference:
    def __init__(self) -> None:
        self.vocabulary = Vocabulary()
        self.model = TextEmotionClassificationNetwork(self.vocabulary, 128, 256, 2)
        self.model.load_state_dict(torch.load(PROJECT_ROOT / "data" / "model"))
        
    def infer(self, text: str) -> dict:
        # mind to disable train mode in inference mode
        text_tensor = ttext_tensor = text_to_fixed_tensor(clean_text(text), self.vocabulary, False).unsqueeze(0)
        result = self.model.forward(text_tensor)
        print(f"raw result {result}")
        # TODO not completed
        print("[cyan]result have been calculated![/cyan]")
        return get_emotion_analysis(result)


def get_emotion_analysis(logits_tensor: torch.Tensor):
    """
    Analyzes the logits to determine emotion category and intensity.
    Input: tensor([[val_neg, val_pos]])
    """
    probs = torch.softmax(logits_tensor, dim=1)
    neg_prob = probs[0][0].item()
    pos_prob = probs[0][1].item()
    
    # We use the raw logit difference to determine "Extreme" vs "Moderate"
    diff = abs(logits_tensor[0][0].item() - logits_tensor[0][1].item())
    
    if neg_prob > pos_prob:
        label = "NEGATIVE"
        score = neg_prob
    else:
        label = "POSITIVE"
        score = pos_prob

    # Logic degree
    if diff < 0.5:
        degree = "Ambiguous / Neutral"
    elif 0.5 <= diff < 2.0:
        degree = f"Moderate {label}"
    else:
        degree = f"Extremely {label}"
        
    return {
        "label": label,
        "degree": degree,
        "confidence": f"{score * 100:.2f}%"
    }