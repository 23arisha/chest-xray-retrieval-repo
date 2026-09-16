"""Small filesystem utility shared by the trainer and the training entry script."""
import os


def mkdir_p(path):
    os.makedirs(path, exist_ok=True)
