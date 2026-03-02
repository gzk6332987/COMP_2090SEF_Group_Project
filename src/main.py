import typer
from rich import print

from core.vocabulary import Vocabulary

app = typer.Typer(rich_markup_mode="rich")

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
    print(f"Result: {word}")

if __name__ == "__main__":
    app()
