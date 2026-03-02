from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from core.network import TextEmotionClassificationNetwork
from core.vocabulary import Vocabulary
from core.text_preprocess import *
import pandas as pd
import tqdm
from rich import print

import matplotlib.pyplot as plt


# avoid memory allocate error in new python version (test version: 3.14.3 with Arch Linux Operation System)
torch.backends.cudnn.enabled = False
torch.backends.cuda.matmul.allow_tf32 = False 

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CSV_DATASET_FILENAME = "training_dataset/processed.csv"

class MovieCommentsDataset(torch.utils.data.Dataset):
    # TODO mind the test mode here! do not turn it on in latest release version
    def __init__(self, vocab: Vocabulary, test_mode: bool = False):
        super().__init__()
        
        if test_mode:
            print("[red]please mind that you have enable test_mode![/red]")
            self.csv_dataset = pd.read_csv(PROJECT_ROOT / CSV_DATASET_FILENAME, nrows=5000)
        else:
            self.csv_dataset = pd.read_csv(PROJECT_ROOT / CSV_DATASET_FILENAME)
        self.vocabulary = vocab
        
    
    def __len__(self) -> int:
        return len(self.csv_dataset)

    def __getitem__(self, index):
        row = self.csv_dataset.iloc[index]
        text = str(row["text"])
        emotion = int(row["emotion"])
        
        # There will auto add new words to vocabulary if this is true
        text_tensor = text_to_fixed_tensor(text, self.vocabulary, False)
        
        return text_tensor, torch.tensor(emotion, dtype=torch.long)
    

def train(optimizer_type: str, training_epoch: int = 512, skip_build_vocabulary=False, show_total_loss_map: bool=False, test_mode: bool = False):
    total_loss_list = []
    # select GPU training if possible (NVIDIA GPU only)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocabulary = Vocabulary()
    
    # prepare dataset
    print("[red]Prepare dataset, This might take a long time (about 10 minutes if not skip build vocabulary)![/red]")
    print("[red]Prepare dataset, This might take a long time (about 10 minutes if not skip build vocabulary)![/red]")
    print("[red]Prepare dataset, This might take a long time (about 10 minutes if not skip build vocabulary)![/red]")
    
    # TODO mind the test mode here! do not turn it on in latest release version
    train_dataset = MovieCommentsDataset(vocabulary, test_mode)
    
    if not skip_build_vocabulary:
        # add all text's words into vocabulary database
        df = pd.read_csv(PROJECT_ROOT / CSV_DATASET_FILENAME)
        for text in tqdm.tqdm(df["text"]):
            add_to_vocabulary_db(clean_text(text), vocabulary)
        
        del df
    else:
        print("[red]warning: You choose skip build vocabulary, there might be some errors in training![/red]")
    
    # use single main thread in dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    
    # create model, optimizer and criterion
    model = TextEmotionClassificationNetwork(vocabulary, 128, 256, 2).to(device)
    if optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=0.01)
    elif optimizer_type == "adadelta":
        optimizer = optim.Adadelta(model.parameters())
    else:
        print(f"[red]warning: we can't recognize optimizer_type {optimizer_type}! we will use adadelta in default[/red]")
        optimizer = optim.Adadelta(model.parameters())
    
    criterion = nn.CrossEntropyLoss()
    
    epoch_pbar = tqdm.tqdm(range(training_epoch), desc="Total progress", unit="epoch")
    for epoch in epoch_pbar:
        epoch_total_loss = 0
        batch_pbar = tqdm.tqdm(train_loader, 
                          desc=f"Epoch {epoch+1}", 
                          leave=False, 
                          colour="cyan")
        
        for tensor_text, label in batch_pbar:
            # send data from memory to GPU (If possible)
            tensor_text, label = tensor_text.to(device), label.to(device)
            
            # initialize this batch training parameters and perform training
            optimizer.zero_grad()
            output = model(tensor_text)
            loss = criterion(output, label)
            epoch_total_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        epoch_pbar.set_postfix(loss=str(epoch_total_loss))
        total_loss_list.append(epoch_total_loss)
        
        print(f"[blue]Save model with loss {epoch_total_loss}[/blue]")
        save_model(model, f"data/model_with_loss_{epoch_total_loss}.model")
        save_model(model, f"data/model")  # save to model in the same time
        
    
    # draw loss chart
    if show_total_loss_map:
        draw_loss_chart(total_loss_list)

def draw_loss_chart(losses: list):
    plt.figure()
    plt.plot(losses, label='Training Loss', color='teal', linewidth=2)
    plt.title("Model Training Progress", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss Value", fontsize=12)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()
    
        
def save_model(network: nn.Module, dest_filename: str):
    project_root = Path(__file__).resolve().parent.parent.parent
    save_path = project_root / dest_filename
    
    torch.save(network.state_dict(), str(save_path))
    
    
    
if __name__ == "__main__":
    train("adadelta", 10, True)