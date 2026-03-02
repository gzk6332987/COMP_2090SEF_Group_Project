import shelve
from pathlib import Path

class Vocabulary:
    def __init__(self) -> None:
        root_dir = Path(__file__).resolve().parent.parent.parent
        db_path = root_dir / "vocabulary" / "vocab_db"
        
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = shelve.open(str(db_path))
    
    def add_word(self, word: str) -> int:
        """_summary_

        Args:
            word (str): A word you want to add

        Returns:
            int: If word is not valid, return -1. If adding successfully or already exist, return the index of the word.
        """
        # transform to lower case and remove empty character
        word = word.strip().lower()
        
        if not word:
            return -1
        
        if word not in self.db:
            current_index = len(self.db.keys())
            self.db[word] = current_index
            return current_index
        
        return self.db[word]
    
    def search_word(self, word: str) -> int:
        """_summary_

        Args:
            word (str): A word you want to get index

        Returns:
            int: If word exist, return the index, otherwise return -1.
        """
        word = word.strip().lower()
        if word not in self.db.keys():
            return -1
        else:
            return self.db[word]
    
    def size(self) -> int:
        return self.db.keys().__len__()
            
    def close(self):
        self.db.close()


# test only
if __name__ == "__main__":
    vocab = Vocabulary()