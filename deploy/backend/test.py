from random import randint
import time

def random_snippet() -> str:
    """
    Returns a random snippet from the dataset
    """
    random_line = randint(0, 270_000)
    with open('snippets.csv', 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == random_line:
                return line

start = time.time()
print(random_snippet())
print(time.time() - start)