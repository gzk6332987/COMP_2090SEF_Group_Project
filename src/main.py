import typer
from rich import print

from core.vocabulary import Vocabulary
from core.training import train as inner_train
from pathlib import Path

app = typer.Typer(rich_markup_mode="rich")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
data_dir = PROJECT_ROOT / "data"
data_dir.mkdir(exist_ok=True)

@app.command()
def add(
    word: str = typer.Argument(..., help="The word you want to add"), 
):
    """
    [bold green]Add a new word to library (persistence)[/bold green]
    """
    vocab = Vocabulary()
    index = vocab.add_word(word)
    print(f"The index of word {word} is {index}.")
    
@app.command()
def dbsize():
    """
    [bold green]Get the total word database size[/bold green]
    """
    vocab = Vocabulary()
    size = vocab.size()
    print(f"The size of vocabulary library is {size}")

@app.command()
def search(word: str):
    """
    [bold blue]Search an index of a word[/bold blue]
    """
    vocabulary = Vocabulary()
    print(f"Result: {vocabulary.search_word(word)}")
    
@app.command()
def train(
    epoch: int = typer.Option(512, help="Training Epoch you want"),
    skip_vocab: bool = typer.Option(False, "--skip-vocab", help="If skip build vocabulary database")
):
    """
    [bold green]Start training. (This is time consuming and your computer must heat up!)[/bold green]
    """
    print(f"[cyan]Start training: {epoch} epochs, skip_vocab={skip_vocab}[/cyan]")
    inner_train(epoch, skip_vocab)

if __name__ == "__main__":
    app()

else:
    print("WARNING: you can not run or import src/main.py in other python file")
