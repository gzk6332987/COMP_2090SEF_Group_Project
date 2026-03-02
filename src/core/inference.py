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
        text_tensor = text_to_fixed_tensor(clean_text(text), self.vocabulary, False).unsqueeze(0)
        result = self.model.forward(text_tensor)
        print(f"raw result {result}")
        # TODO not completed
        print("[cyan]result have been calculated![/cyan]")
        return result, get_emotion_analysis(result)


def get_emotion_analysis(logits_tensor: torch.Tensor):
    """
    Analyzes the logits to determine emotion category and intensity.
    Input: tensor([[val_neg, val_pos]])
    """
    probs = torch.softmax(logits_tensor, dim=1)
    neg_prob = probs[0][0].item()
    pos_prob = probs[0][1].item()
    
    # use the raw logit difference to determine "Extreme" vs "Moderate"  (FEATURE Might be a problem only rely on confidence rate)
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


# lib for print_pretty_analysis only
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

def print_pretty_analysis(analysis_data: dict, raw_tensor: torch.Tensor):
    console = Console()

    color = "white"
    if "POSITIVE" in analysis_data["label"]:
        color = "bold green" if "Extremely" in analysis_data["degree"] else "green"
    elif "NEGATIVE" in analysis_data["label"]:
        color = "bold red" if "Extremely" in analysis_data["degree"] else "red"
    elif "Ambiguous" in analysis_data["degree"]:
        color = "yellow"

    table = Table(show_header=False, box=box.SIMPLE_HEAD, expand=True)
    table.add_row("[cyan]Sentiment Index[/cyan]", f"[{color}]{analysis_data['label']}[/{color}]")
    table.add_row("[cyan]Emotion Intensity[/cyan]", f"[{color}]{analysis_data['degree']}[/{color}]")
    table.add_row("[cyan]Model Confidence[/cyan]", f"{analysis_data['confidence']}")
    table.add_row("[cyan]Raw Logits[/cyan]", f"[dim]{raw_tensor.tolist()}[/dim]")

    panel = Panel(
        table,
        title="[bold white]🧠 Inference Results[/bold white]",
        border_style=color,
        subtitle="[dim]Text Emotion Classification Engine[/dim]",
        padding=(1, 2)
    )

    console.print("\n", panel, "\n")
