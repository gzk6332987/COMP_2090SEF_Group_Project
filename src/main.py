import typer
from rich import print

from core.vocabulary import Vocabulary
from core.training import train as inner_train
from core.inference import Inference
from core.utils import check_cuda_status
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
    optimizer_type: str = typer.Option("adadelta", help="The optimizer you want to use"),
    epoch: int = typer.Option(512, help="Training Epoch you want"),
    skip_vocab: bool = typer.Option(False, "--skip-vocab", help="If skip build vocabulary database"),
    show_chart: bool = typer.Option(False, "--show-chart", help="Enable to show total loss chart at the end of training"),
    test_mode: bool = typer.Option(False, "--test-mode", help="Enable to reduce training dataset")
):
    """
    [bold green]Start training. (This is time consuming and your computer must heat up!)[/bold green]
    """
    print(f"[cyan]Start training: {epoch} epochs, skip_vocab={skip_vocab}[/cyan]")
    inner_train(optimizer_type, epoch, skip_vocab, show_chart, test_mode)

@app.command()
def infer(
    text = typer.Option(None, "--text", help="The text you want to infer emotion (MUST be english)"),
    file = typer.Option(None, "--file", help="The file you want to infer emotion (MUST be english)")
):
    if text is None and file is None:
        print("[bold red]--text or --file can not be empty![/bold red]")
        return
    if file is not None and text is not None:
        print("[bold red]--text and --file conflict with each other![/bold red]")
        return
    
    if file is not None:
        try:
            with open(file, "r") as f:
                text = f.read()
        except FileNotFoundError:
            print(f"[bold red]No Such file with path {file}[/bold red]")

    print(f"[gree]WE have received your text, start performing inference[/gree]")
    inference = Inference()
    inference.infer(text)

@app.command()
def cudatest():
    """
    [bold purple]Check your computer CUDA (GPU acceleration) support status[/bold purple]
    """
    check_cuda_status()

if __name__ == "__main__":
    
    app()

else:
    print("WARNING: you can not run or import src/main.py in other python file")
