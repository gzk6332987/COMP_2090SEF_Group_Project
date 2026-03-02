import torch.nn as nn
import torch

DEFAULT_MAX_BAG_SIZE: int = 10010

class TextPreprocess:
    
    def __init__(self, filename: str):
        self.filename = filename
        with open(filename, "r") as f:
            self.content = f.read()
        
        # create an empty word bag
        self.ebag = nn.EmbeddingBag(DEFAULT_MAX_BAG_SIZE, 3, )
    
    def perform(self):
        
            
    