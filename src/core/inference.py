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
        
    def infer(self, text: str):
        # mind to disable train mode in inference mode
        text_tensor = text_to_fixed_tensor(clean_text(text), self.vocabulary, False)
        result = self.model.forward(text_tensor)
        print(result)
        # TODO not completed
        print("[cyan]result have been calculated![/cyan]")
